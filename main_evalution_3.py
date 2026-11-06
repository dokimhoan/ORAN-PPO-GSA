"""Thay duy nhất kết quả PPO-MLP, giữ nguyên MILP và PPO-GSA.

Ví dụ::

    python evaluate_mlp_only.py
    python evaluate_mlp_only.py --train-results results_train_4 \
        --eval-results result_eval_3

Script đánh giá checkpoint MLP trên đúng các cặp run/seed đã có
trong kết quả cũ. Chỉ sau khi mọi episode thành công, các row/file
PPO-MLP cũ mới bị thay thế theo kiểu nguyên tử.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.PPO_MLP import PPOAgent
from env.RLEnv import RLEnv
from main_train import PPOStateAdapter


MLP_METHOD = "PPO_MLP"
MLP_ALIASES = {"ppo_mlp", "ppo-mlp"}
GSA_ALIASES = {"ppo_gsa", "ppo-gsa"}
METHOD_ORDER = (
    "Long_Term", "Short_Term", "Nearest_RU", "Random_RU",
    "RoundRobin_RU", "PPO_GSA", MLP_METHOD,
)
CONFIGS = (
    (20, 135, 7, 5, 5),
    (30, 135, 7, 5, 5),
    (40, 135, 7, 5, 5),
    (50, 135, 7, 5, 5),
    (60, 135, 7, 5, 5),
    (70, 135, 7, 5, 5),
)
CSV_FIELDS = (
    "config", "num_UEs", "num_RBs", "num_RUs", "num_DUs", "num_CUs",
    "run", "seed", "method", "score", "accepted", "acceptance_rate",
    "total_throughput_bps", "total_throughput_mbps", "total_latency_s",
    "mean_latency_s", "mean_latency_ms", "allocated_rbs", "power_w",
    "runtime_s", "inference_runtime_s",
)
SUMMARY_METRICS = (
    "score", "accepted", "acceptance_rate", "total_throughput_mbps",
    "total_latency_s", "mean_latency_ms", "allocated_rbs", "power_w",
    "runtime_s", "inference_runtime_s",
)


def config_name(config):
    ue, rb, ru, du, cu = config
    return f"UE{ue}_RB{rb}_RU{ru}_DU{du}_CU{cu}"


def is_mlp(row):
    return str(row.get("method", "")).strip().lower() in MLP_ALIASES


def read_rows(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def atomic_write_rows(path, rows, fields=CSV_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_retained_rows(root, name):
    """Nạp mọi kết quả cũ ngoại trừ PPO-MLP."""
    combined = root / f"{name}_all_results.csv"
    rows = [row for row in read_rows(combined) if not is_mlp(row)]
    if rows:
        return rows

    rows = []
    for method in METHOD_ORDER[:-1]:
        rows.extend(read_rows(root / f"{name}_{method}_results.csv"))
    rows = [row for row in rows if not is_mlp(row)]
    if not rows:
        raise FileNotFoundError(
            f"Không tìm thấy kết quả MILP/PPO-GSA cũ của {name} trong {root}."
        )
    return rows


def evaluation_cases(rows, name):
    """Lấy run/seed từ MILP/heuristic, không phụ thuộc PPO cũ."""
    by_method = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method.strip().lower() in GSA_ALIASES:
            continue
        try:
            pair = (int(row["run"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Dòng CSV không hợp lệ của {name}: {row}") from error
        by_method.setdefault(method, set()).add(pair)

    if not by_method:
        raise ValueError(f"{name} không có run/seed để evaluation.")
    reference_method = next(iter(by_method))
    reference = by_method[reference_method]
    mismatch = {
        method: sorted(pairs)
        for method, pairs in by_method.items()
        if pairs != reference
    }
    if mismatch:
        raise ValueError(
            f"Các method của {name} không dùng cùng run/seed: {mismatch}"
        )
    return sorted(reference)


def cuda_sync(agent):
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)


def info_value(info, *names, default=0.0):
    for name in names:
        if name in info:
            return info[name]
    return default


@torch.inference_mode()
def evaluate_episode(agent, env, seed, run, precise_timing=False):
    state = env.reset(seed=seed)
    num_ues = len(state["state_UEs_req"])
    totals = {
        "score": 0.0, "accepted": 0, "throughput": 0.0,
        "latency": 0.0, "rbs": 0, "power": 0.0,
    }
    inference_time = 0.0 if precise_timing else np.nan

    cuda_sync(agent)
    episode_start = time.perf_counter()
    for step in range(num_ues + 1):
        if env.episode_done:
            break
        if step == num_ues:
            raise RuntimeError(f"{MLP_METHOD}: episode vượt {num_ues} bước.")

        if precise_timing:
            cuda_sync(agent)
            inference_start = time.perf_counter()
        policy = agent.act([state], deterministic=True)
        if precise_timing:
            cuda_sync(agent)
            inference_time += time.perf_counter() - inference_start

        action = policy["env_actions"][0]
        state, reward, _, info = env.step(action)
        totals["score"] += float(reward)
        accepted = bool(info_value(info, "pi_k", "accepted", "success"))
        if accepted:
            totals["accepted"] += 1
            totals["throughput"] += float(
                info_value(info, "R_k", "throughput_bps")
            )
            totals["latency"] += float(
                info_value(info, "L_k", "latency_s")
            )
            totals["power"] += float(info_value(
                info, "num_P_i_k", "power_alloc", default=action[4]
            ))
            totals["rbs"] += int(info_value(
                info, "num_z_i_k", "num_RB_alloc", default=action[5]
            ))

    cuda_sync(agent)
    runtime = time.perf_counter() - episode_start
    accepted = totals["accepted"]
    return {
        "run": run,
        "seed": seed,
        "method": MLP_METHOD,
        "score": totals["score"],
        "accepted": accepted,
        "acceptance_rate": accepted / max(num_ues, 1),
        "total_throughput_bps": totals["throughput"],
        "total_throughput_mbps": totals["throughput"] / 1e6,
        "total_latency_s": totals["latency"],
        "mean_latency_s": totals["latency"] / max(accepted, 1),
        "mean_latency_ms": totals["latency"] / max(accepted, 1) * 1e3,
        "allocated_rbs": totals["rbs"],
        "power_w": totals["power"],
        "runtime_s": runtime,
        "inference_runtime_s": inference_time,
    }


def attach_config(row, config):
    ue, rb, ru, du, cu = config
    return {
        **row,
        "config": config_name(config),
        "num_UEs": ue,
        "num_RBs": rb,
        "num_RUs": ru,
        "num_DUs": du,
        "num_CUs": cu,
    }


def summary_rows(rows):
    fields = ["method", "num_valid_runs"]
    for metric in SUMMARY_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"))

    present = {row.get("method", "") for row in rows}
    methods = [method for method in METHOD_ORDER if method in present]
    methods.extend(sorted(present - set(methods)))
    output = []
    for method in methods:
        method_rows = [
            row for row in rows
            if row.get("method") == method and finite(row.get("score"))
        ]
        summary = {"method": method, "num_valid_runs": len(method_rows)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([
                float(row[metric])
                for row in method_rows
                if finite(row.get(metric))
            ], dtype=float)
            if values.size:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if values.size > 1 else 0.0
                ci95 = float(1.96 * std / np.sqrt(values.size))
            else:
                mean = std = ci95 = np.nan
            summary.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95": ci95,
            })
        output.append(summary)
    return fields, output


def sort_rows(rows):
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (
        int(row.get("run", 0)),
        order.get(row.get("method"), len(order)),
        row.get("method", ""),
    ))


def remove_old_mlp_artifacts(root):
    """Chỉ xóa file riêng PPO-MLP; không đụng PPO-GSA/MILP."""
    removed = []
    for path in root.iterdir():
        if path.is_file() and "ppo_mlp" in path.name.lower():
            path.unlink()
            removed.append(path.name)
    return removed


def strip_mlp_from_untargeted_csv(root, targeted):
    """Xóa row MLP cũ ngoài CONFIGS, giữ nguyên mọi method khác."""
    for path in root.glob("*_all_results.csv"):
        if path.name == "all_evaluation_results.csv" or path in targeted:
            continue
        rows = read_rows(path)
        if any(is_mlp(row) for row in rows):
            atomic_write_rows(path, [row for row in rows if not is_mlp(row)])

    for path in root.glob("*_summary.csv"):
        if path in targeted:
            continue
        rows = read_rows(path)
        if any(is_mlp(row) for row in rows):
            fields = list(rows[0]) if rows else ["method", "num_valid_runs"]
            atomic_write_rows(
                path, [row for row in rows if not is_mlp(row)], fields
            )


def save_results(root, evaluated):
    """Commit sau khi evaluation xong; chỉ thay dữ liệu PPO-MLP."""
    target_all = {root / f"{name}_all_results.csv" for name in evaluated}
    target_summary = {root / f"{name}_summary.csv" for name in evaluated}
    removed = remove_old_mlp_artifacts(root)
    strip_mlp_from_untargeted_csv(root, target_all | target_summary)

    for name, rows in evaluated.items():
        rows = sort_rows(rows)
        atomic_write_rows(root / f"{name}_all_results.csv", rows)
        fields, summaries = summary_rows(rows)
        atomic_write_rows(root / f"{name}_summary.csv", summaries, fields)

        mlp_rows = [row for row in rows if row.get("method") == MLP_METHOD]
        atomic_write_rows(root / f"{name}_{MLP_METHOD}_results.csv", mlp_rows)
        runtime_path = root / f"{name}_{MLP_METHOD}_runtime.txt"
        temporary = runtime_path.with_name(runtime_path.name + ".tmp")
        temporary.write_text(
            "".join(f"{row['runtime_s']}\n" for row in mlp_rows),
            encoding="utf-8",
        )
        temporary.replace(runtime_path)

    global_path = root / "all_evaluation_results.csv"
    target_names = set(evaluated)
    retained_global = [
        row for row in read_rows(global_path)
        if row.get("config") not in target_names and not is_mlp(row)
    ]
    rebuilt = retained_global + [
        row for rows in evaluated.values() for row in rows
    ]
    rebuilt.sort(key=lambda row: (
        int(row.get("num_UEs", 0)),
        int(row.get("run", 0)),
        METHOD_ORDER.index(row["method"])
        if row.get("method") in METHOD_ORDER else len(METHOD_ORDER),
    ))
    atomic_write_rows(global_path, rebuilt)
    return removed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-results", type=Path, default=Path("results_train_4")
    )
    parser.add_argument(
        "--eval-results", type=Path, default=Path("result_eval_3")
    )
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument(
        "--precise-timing", action="store_true",
        help="CUDA synchronize từng action để đo inference time chính xác.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.mlp_checkpoint or (
        args.train_results / "PPO_MLP" / "PPOMLP_model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint MLP: {checkpoint}")
    if not args.eval_results.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy folder evaluation cũ: {args.eval_results}"
        )

    # Chỉ đọc dữ liệu. Không file nào bị sửa nếu evaluation lỗi.
    retained_by_config = {}
    cases_by_config = {}
    for config in CONFIGS:
        name = config_name(config)
        rows = load_retained_rows(args.eval_results, name)
        retained_by_config[name] = rows
        cases_by_config[name] = evaluation_cases(rows, name)

    agent, metadata = PPOAgent.load(
        checkpoint, device=device, restore_rng=False
    )
    agent.model.eval()

    print("=" * 88)
    print("PPO-MLP EVALUATION — GIỮ NGUYÊN MILP VÀ PPO-GSA")
    print("=" * 88)
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Metadata   : {metadata}")

    evaluated = {}
    for config in CONFIGS:
        name = config_name(config)
        pairs = cases_by_config[name]
        env = PPOStateAdapter(RLEnv(*config))

        # Warm-up CUDA không được tính vào runtime.
        warmup_state = env.reset(seed=pairs[0][1])
        agent.act([warmup_state], deterministic=True)
        cuda_sync(agent)

        mlp_rows = []
        print(f"\n{name}: {len(pairs)} snapshot")
        for index, (run, seed) in enumerate(pairs, 1):
            row = evaluate_episode(
                agent, env, seed, run, args.precise_timing
            )
            mlp_rows.append(attach_config(row, config))
            print(
                f"  {index:02d}/{len(pairs):02d} seed={seed} | "
                f"PPO_MLP={row['score']:.4f}"
            )

        retained = [
            attach_config(row, config)
            for row in retained_by_config[name]
        ]
        evaluated[name] = retained + mlp_rows

    removed = save_results(args.eval_results, evaluated)
    print("\n" + "=" * 88)
    print("HOÀN TẤT")
    print(f"Folder       : {args.eval_results.resolve()}")
    print(f"MLP files cũ: đã xóa {len(removed)} file")
    print("MILP/PPO-GSA : giữ nguyên")
    print("=" * 88)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    main()
