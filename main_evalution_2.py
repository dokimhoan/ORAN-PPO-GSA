"""Đánh giá lại hai PPO và tái sử dụng nguyên kết quả MILP cũ.

Quy trình:
    1. Đọc các hàng MILP từ results_eval_6/all_evaluation_results.csv.
    2. Không đọc hoặc sao chép bất kỳ hàng PPO cũ nào.
    3. Load PPO-GSA và PPO-MLP từ results_train_7.
    4. Chạy PPO mới trên đúng config, run và seed của dữ liệu MILP cũ.
    5. Tạo lại toàn bộ CSV/summary/runtime trong results_eval_7.

Chạy tại thư mục gốc project:
    python evaluate_ppo_reuse_milp.py
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from env.RLEnv import RLEnv
from algorithms.PPO_GSA_MLP import PPOAgent as PPOGSAAgent


# ======================================================================
# 1. CẤU HÌNH
# ======================================================================

OLD_RESULTS_CANDIDATES = (
    Path("./results_eval_6"),
    Path("./result_eval_6"),  # Tên đang xuất hiện trong file evaluation cũ.
)
NEW_RESULTS_ROOT = Path("./results_eval_7")
TRAIN_RESULTS_ROOT = Path("./results_train_7")

PPO_GSA_DIR = TRAIN_RESULTS_ROOT / "PPO_GSA_MLP"
PPO_MLP_DIR = TRAIN_RESULTS_ROOT / "PPO_MLP"

# Checkpoint PPO-MLP trong results_train_7 được train bằng file này.
# Phải load bằng đúng class/kiến trúc đã dùng khi train.
PPO_MLP_MODULE = "algorithms.PPO_MLP_2"
PPOMLPAgent = importlib.import_module(PPO_MLP_MODULE).PPOAgent

PPO_GSA_CHECKPOINT_NAMES = ("PPOGSA_model.pt", "PPOGSA_latest.pt")
PPO_MLP_CHECKPOINT_NAMES = ("PPOMLP_model.pt", "PPOMLP_latest.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PPO_DETERMINISTIC = True

# False đo runtime toàn episode chính xác với một lần CUDA sync ở đầu/cuối.
# True còn đo riêng inference từng action nhưng sẽ làm evaluation chậm hơn.
PRECISE_PPO_INFERENCE_TIMING = False

MILP_METHODS = (
    "Long_Term",
    "Short_Term",
    "Nearest_RU",
    "Random_RU",
    "RoundRobin_RU",
)
PPO_METHODS = ("PPO_GSA", "PPO_MLP")
METHODS = (*MILP_METHODS, *PPO_METHODS)

CSV_FIELDS = [
    "config", "num_UEs", "num_RBs", "num_RUs", "num_DUs", "num_CUs",
    "run", "seed", "method",
    "score", "accepted", "acceptance_rate",
    "total_throughput_bps", "total_throughput_mbps",
    "total_latency_s", "mean_latency_s", "mean_latency_ms",
    "allocated_rbs", "power_w", "runtime_s", "inference_runtime_s",
]

SUMMARY_METRICS = (
    "score",
    "accepted",
    "acceptance_rate",
    "total_throughput_mbps",
    "total_latency_s",
    "mean_latency_ms",
    "allocated_rbs",
    "power_w",
    "runtime_s",
    "inference_runtime_s",
)


# ======================================================================
# 2. ĐỌC VÀ KIỂM TRA KẾT QUẢ CŨ
# ======================================================================

def find_old_results_root():
    """Tìm đúng thư mục cũ, hỗ trợ cả result_eval_6 và results_eval_6."""
    matches = [
        path for path in OLD_RESULTS_CANDIDATES
        if (path / "all_evaluation_results.csv").is_file()
    ]
    if not matches:
        candidates = "\n".join(
            f"  - {path.resolve()}" for path in OLD_RESULTS_CANDIDATES
        )
        raise FileNotFoundError(
            "Không tìm thấy all_evaluation_results.csv tại:\n" + candidates
        )
    if len(matches) > 1:
        print(
            "Cảnh báo: tìm thấy cả hai thư mục cũ; sử dụng "
            f"{matches[0].resolve()}"
        )
    return matches[0]


def resolve_checkpoint(directory, names):
    """Ưu tiên model.pt; chỉ dùng latest.pt khi model.pt không tồn tại."""
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    expected = "\n".join(f"  - {(directory / name).resolve()}" for name in names)
    raise FileNotFoundError("Không tìm thấy checkpoint:\n" + expected)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV không có header: {path}")
        missing = [field for field in CSV_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} thiếu cột: {missing}")
        return [dict(row) for row in reader]


def integer(row, field):
    try:
        return int(float(row[field]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Giá trị {field} không hợp lệ trong config={row.get('config')}: "
            f"{row.get(field)!r}"
        ) from error


def config_tuple(row):
    return (
        integer(row, "num_UEs"),
        integer(row, "num_RBs"),
        integer(row, "num_RUs"),
        integer(row, "num_DUs"),
        integer(row, "num_CUs"),
    )


def group_old_milp_rows(all_rows):
    """Loại hoàn toàn PPO cũ và nhóm các hàng MILP theo config."""
    grouped = defaultdict(list)

    for row in all_rows:
        if row["method"] in MILP_METHODS:
            grouped[row["config"]].append(row)

    if not grouped:
        raise ValueError("CSV cũ không chứa kết quả MILP hợp lệ.")

    for name, rows in grouped.items():
        configs = {config_tuple(row) for row in rows}
        if len(configs) != 1:
            raise ValueError(
                f"Config {name!r} chứa nhiều bộ tham số mạng: {configs}"
            )

        keys_by_method = {}
        for method in MILP_METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            if not method_rows:
                raise ValueError(f"Config {name} thiếu kết quả {method}.")

            keys = [
                (integer(row, "run"), integer(row, "seed"))
                for row in method_rows
            ]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Config {name}, method {method} có run/seed trùng.")
            keys_by_method[method] = set(keys)

        reference = keys_by_method[MILP_METHODS[0]]
        for method, keys in keys_by_method.items():
            if keys != reference:
                missing = sorted(reference - keys)
                extra = sorted(keys - reference)
                raise ValueError(
                    f"Run/seed của {name}/{method} không khớp MILP gốc. "
                    f"Thiếu={missing[:5]}, thừa={extra[:5]}"
                )

    return grouped


# ======================================================================
# 3. PPO EVALUATION
# ======================================================================

def cuda_sync(agent):
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)


@torch.inference_mode()
def warmup_ppo(agent, env, seed):
    state = env.reset(seed=seed)
    cuda_sync(agent)
    agent.act([state], deterministic=True)
    cuda_sync(agent)


@torch.inference_mode()
def run_ppo(agent, env, seed, method, run):
    state = env.reset(seed=seed)
    num_ues = len(state["state_UEs_req"])

    score = 0.0
    throughput = 0.0
    total_latency = 0.0
    power = 0.0
    accepted = 0
    allocated_rbs = 0
    steps = 0
    done = False

    inference_runtime = 0.0 if PRECISE_PPO_INFERENCE_TIMING else np.nan
    cuda_sync(agent)
    episode_start = time.perf_counter()

    while not done:
        if PRECISE_PPO_INFERENCE_TIMING:
            cuda_sync(agent)
            inference_start = time.perf_counter()

        policy = agent.act([state], deterministic=PPO_DETERMINISTIC)

        if PRECISE_PPO_INFERENCE_TIMING:
            cuda_sync(agent)
            inference_runtime += time.perf_counter() - inference_start

        action = policy["env_actions"][0]
        state, reward, done, info = env.step(action)

        score += float(reward)
        accepted_now = int(
            info.get("pi_k", info.get("accepted", info.get("success", 0)))
        )

        if accepted_now:
            accepted += 1
            throughput += float(info.get("R_k", info.get("throughput_bps", 0.0)))
            total_latency += float(info.get("L_k", info.get("latency_s", 0.0)))
            power += float(action[4])
            allocated_rbs += int(action[5])

        steps += 1
        if steps > num_ues:
            raise RuntimeError(
                f"{method}: số step ({steps}) vượt số UE ({num_ues}); "
                "kiểm tra logic deactivate UE trong RLEnv."
            )

    cuda_sync(agent)
    runtime = time.perf_counter() - episode_start

    return {
        "run": run,
        "seed": seed,
        "method": method,
        "score": score,
        "accepted": accepted,
        "acceptance_rate": accepted / max(num_ues, 1),
        "total_throughput_bps": throughput,
        "total_throughput_mbps": throughput / 1e6,
        "total_latency_s": total_latency,
        "mean_latency_s": total_latency / max(accepted, 1),
        "mean_latency_ms": total_latency / max(accepted, 1) * 1e3,
        "allocated_rbs": allocated_rbs,
        "power_w": power,
        "runtime_s": runtime,
        "inference_runtime_s": inference_runtime,
    }


# ======================================================================
# 4. GHI KẾT QUẢ
# ======================================================================

def clean_output_directory(path):
    """Chỉ xóa các artifact do chính script này sinh ra."""
    path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*_runtime.txt", "evaluation_manifest.json"):
        for target in path.glob(pattern):
            if target.is_file():
                target.unlink()


def atomic_write_csv(path, rows, fields=CSV_FIELDS):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_runtime(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(f"{row['runtime_s']}\n")
    temporary.replace(path)


def finite_values(rows, metric):
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def write_summary(rows, path):
    fields = ["method", "num_valid_runs"]
    for metric in SUMMARY_METRICS:
        fields.extend((
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_ci95",
        ))

    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        valid_rows = [
            row for row in method_rows
            if finite_values([row], "score").size == 1
        ]
        summary = {
            "method": method,
            "num_valid_runs": len(valid_rows),
        }

        for metric in SUMMARY_METRICS:
            values = finite_values(valid_rows, metric)
            if values.size == 0:
                mean = std = ci95 = np.nan
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if values.size > 1 else 0.0
                ci95 = float(1.96 * std / np.sqrt(values.size))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci95"] = ci95

        summary_rows.append(summary)

    atomic_write_csv(path, summary_rows, fields)


def complete_row(row, config_name, config):
    num_ues, num_rbs, num_rus, num_dus, num_cus = config
    completed = dict(row)
    completed.update({
        "config": config_name,
        "num_UEs": num_ues,
        "num_RBs": num_rbs,
        "num_RUs": num_rus,
        "num_DUs": num_dus,
        "num_CUs": num_cus,
    })
    return completed


def sort_rows(rows):
    order = {method: index for index, method in enumerate(METHODS)}
    return sorted(
        rows,
        key=lambda row: (
            integer(row, "run"),
            order.get(row["method"], len(order)),
        ),
    )


def save_config_outputs(config_name, rows):
    rows = sort_rows(rows)
    atomic_write_csv(
        NEW_RESULTS_ROOT / f"{config_name}_all_results.csv",
        rows,
    )
    write_summary(
        rows,
        NEW_RESULTS_ROOT / f"{config_name}_summary.csv",
    )

    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        atomic_write_csv(
            NEW_RESULTS_ROOT / f"{config_name}_{method}_results.csv",
            method_rows,
        )
        write_runtime(
            NEW_RESULTS_ROOT / f"{config_name}_{method}_runtime.txt",
            method_rows,
        )
    return rows


# ======================================================================
# 5. MAIN
# ======================================================================

def main():
    old_root = find_old_results_root()
    gsa_checkpoint = resolve_checkpoint(
        PPO_GSA_DIR,
        PPO_GSA_CHECKPOINT_NAMES,
    )
    mlp_checkpoint = resolve_checkpoint(
        PPO_MLP_DIR,
        PPO_MLP_CHECKPOINT_NAMES,
    )

    old_all_csv = old_root / "all_evaluation_results.csv"
    old_rows = read_csv(old_all_csv)
    grouped = group_old_milp_rows(old_rows)

    # Không bao giờ sửa/xóa results_eval_6. results_eval_7 được tạo lại từ đầu.
    clean_output_directory(NEW_RESULTS_ROOT)

    gsa_agent, gsa_metadata = PPOGSAAgent.load(
        gsa_checkpoint,
        device=DEVICE,
        restore_rng=False,
    )
    mlp_agent, mlp_metadata = PPOMLPAgent.load(
        mlp_checkpoint,
        device=DEVICE,
        restore_rng=False,
    )

    print("=" * 100)
    print("REUSE MILP RESULTS + EVALUATE NEW PPO CHECKPOINTS")
    print("=" * 100)
    print(f"MILP source : {old_root.resolve()}")
    print(f"PPO source  : {TRAIN_RESULTS_ROOT.resolve()}")
    print(f"Output      : {NEW_RESULTS_ROOT.resolve()}")
    print(f"Device      : {DEVICE}")
    print(f"PPO-GSA     : {gsa_checkpoint.resolve()}")
    print(f"PPO-MLP     : {mlp_checkpoint.resolve()}")
    print(f"PPO-MLP code: {PPO_MLP_MODULE}")
    print(f"GSA metadata: {gsa_metadata}")
    print(f"MLP metadata: {mlp_metadata}")

    all_new_rows = []

    for config_name in sorted(grouped):
        milp_rows = grouped[config_name]
        config = config_tuple(milp_rows[0])

        reference_method = MILP_METHODS[0]
        run_seeds = sorted({
            (integer(row, "run"), integer(row, "seed"))
            for row in milp_rows
            if row["method"] == reference_method
        })

        print("\n" + "#" * 100)
        print(f"CONFIG {config_name}: {config} | runs={len(run_seeds)}")
        print("#" * 100)

        gsa_env = RLEnv(*config)
        mlp_env = RLEnv(*config)
        warmup_ppo(gsa_agent, gsa_env, run_seeds[0][1])
        warmup_ppo(mlp_agent, mlp_env, run_seeds[0][1])

        ppo_rows = []
        for index, (run, seed) in enumerate(run_seeds, start=1):
            gsa_row = run_ppo(
                gsa_agent,
                gsa_env,
                seed,
                "PPO_GSA",
                run,
            )
            mlp_row = run_ppo(
                mlp_agent,
                mlp_env,
                seed,
                "PPO_MLP",
                run,
            )
            ppo_rows.extend((gsa_row, mlp_row))

            print(
                f"  PPO {index:02d}/{len(run_seeds):02d} | "
                f"run={run:02d}, seed={seed} | "
                f"GSA={gsa_row['score']:.3f} ({gsa_row['runtime_s']:.4f}s) | "
                f"MLP={mlp_row['score']:.3f} ({mlp_row['runtime_s']:.4f}s)"
            )

        # MILP được lấy nguyên giá trị; chỉ PPO được tạo mới.
        combined = [dict(row) for row in milp_rows]
        combined.extend(
            complete_row(row, config_name, config) for row in ppo_rows
        )

        saved_rows = save_config_outputs(config_name, combined)
        all_new_rows.extend(saved_rows)

        print(
            f"Đã lưu {len(milp_rows)} hàng MILP cũ + "
            f"{len(ppo_rows)} hàng PPO mới."
        )

    all_new_rows.sort(
        key=lambda row: (
            row["config"],
            integer(row, "run"),
            METHODS.index(row["method"]),
        )
    )
    atomic_write_csv(
        NEW_RESULTS_ROOT / "all_evaluation_results.csv",
        all_new_rows,
    )

    manifest = {
        "milp_source": str(old_root.resolve()),
        "ppo_training_source": str(TRAIN_RESULTS_ROOT.resolve()),
        "ppo_gsa_checkpoint": str(gsa_checkpoint.resolve()),
        "ppo_mlp_checkpoint": str(mlp_checkpoint.resolve()),
        "ppo_mlp_module": PPO_MLP_MODULE,
        "device": DEVICE,
        "deterministic": PPO_DETERMINISTIC,
        "configs": sorted(grouped),
        "methods_reused": list(MILP_METHODS),
        "methods_evaluated": list(PPO_METHODS),
    }
    with (NEW_RESULTS_ROOT / "evaluation_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("HOÀN THÀNH")
    print(f"Kết quả mới: {(NEW_RESULTS_ROOT / 'all_evaluation_results.csv').resolve()}")
    print("results_eval_6 được giữ nguyên; không chạy lại bất kỳ MILP nào.")
    print("=" * 100)


if __name__ == "__main__":
    main()
