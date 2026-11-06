"""Evaluate BLO, MILP baselines and PPO agents on identical RAN snapshots.

The program runs sequentially so wall-clock runtime is not biased by CPU
contention.  A seed is saved only when every method returns a complete finite
result, guaranteeing the same number of paired valid samples for all methods.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env.RLEnv import RLEnv
import algorithms.MILP_BLO as MILP_BLO
import algorithms.MILP_Nearest as MILP_NEAREST
import algorithms.MILP_Random as MILP_RANDOM
import algorithms.MILP_RoundRobin as MILP_ROUNDROBIN
from utils.data_converter_unslice import (
    data_to_long_term,
    data_to_nearest_ru,
    data_to_random_ru,
    data_to_roundrobin_ru,
    data_to_short_term,
    state_to_MILP,
)


RESULTS_ROOT = Path("results_eval_20")
TRAIN_ROOT = Path("results_train_10")
PPO_GSA_CHECKPOINT = TRAIN_ROOT / "PPO_GSA_MLP" / "PPOGSA_best.pt"
PPO_MLP_CHECKPOINT = TRAIN_ROOT / "PPO_MLP" / "PPOMLP_best.pt"

CONFIGS = (
    (20, 30, 5, 3, 3),
    (30, 135, 7, 5, 5),
    (40, 135, 7, 5, 5),
    (50, 135, 7, 5, 5),
)
METHODS = ("BLO", "PPO-GSA", "PPO-MLP", "Nearest-RU", "RoundRobin", "Random-RU")
METRICS = (
    "score",
    "accepted",
    "acceptance_rate",
    "total_throughput_mbps",
    "total_latency_ms",
    "allocated_rbs",
    "power_w",
    "runtime_s",
    "inference_runtime_s",
)
RUN_FIELDS = (
    "config", "num_UEs", "num_RBs", "num_RUs", "num_DUs", "num_CUs",
    "run", "seed", "method", "status", "success", "error",
    *METRICS,
    "long_term_runtime_s", "short_term_runtime_s",
)


def config_name(config: tuple[int, ...]) -> str:
    ue, rb, ru, du, cu = config
    return f"UE{ue}_RB{rb}_RU{ru}_DU{du}_CU{cu}"


def valid_milp(result) -> bool:
    return result is not None and len(result) == 13 and result[0] is not None


def timed_call(function, data):
    start = time.perf_counter()
    result = function(**data) if isinstance(data, dict) else function(*data)
    return result, time.perf_counter() - start


def configure_mosek(mip_threads: int, mip_time: float, mip_gap: float) -> None:
    for module in (MILP_BLO, MILP_NEAREST, MILP_RANDOM, MILP_ROUNDROBIN):
        for value in vars(module).values():
            if not isinstance(value, dict):
                continue
            if "MSK_DPAR_MIO_MAX_TIME" in value:
                value["MSK_DPAR_MIO_MAX_TIME"] = float(mip_time)
                value["MSK_DPAR_MIO_TOL_REL_GAP"] = float(mip_gap)
                value["MSK_IPAR_NUM_THREADS"] = int(mip_threads)
            elif "MSK_DPAR_OPTIMIZER_MAX_TIME" in value:
                value["MSK_DPAR_OPTIMIZER_MAX_TIME"] = 30.0
                value["MSK_IPAR_NUM_THREADS"] = 1


def base_row(config, run: int, seed: int, method: str) -> dict:
    ue, rb, ru, du, cu = config
    row = {
        "config": config_name(config),
        "num_UEs": ue,
        "num_RBs": rb,
        "num_RUs": ru,
        "num_DUs": du,
        "num_CUs": cu,
        "run": run,
        "seed": seed,
        "method": method,
        "status": "not_run",
        "success": 0,
        "error": "",
    }
    row.update({metric: np.nan for metric in METRICS})
    row.update({"long_term_runtime_s": np.nan, "short_term_runtime_s": np.nan})
    return row


def metrics_from_milp(result, num_ues: int) -> dict:
    accepted = int(round(float(result[7])))
    throughput_bps = float(result[8])
    return {
        "score": float(result[6]),
        "accepted": accepted,
        "acceptance_rate": accepted / max(num_ues, 1),
        "total_throughput_mbps": throughput_bps / 1e6,
        "total_latency_ms": float(result[10]) * 1e3,
        "allocated_rbs": int(round(float(result[11]))),
        "power_w": float(result[12]),
    }


def evaluate_blo(data, config, run: int, seed: int) -> dict:
    row = base_row(config, run, seed, "BLO")
    try:
        print("      Long-Term is running...", flush=True)
        long_result, long_time = timed_call(
            MILP_BLO.Long_Term,
            data_to_long_term(copy.deepcopy(data)),
        )
        row["long_term_runtime_s"] = long_time
        row["runtime_s"] = long_time
        print(
            f"      Long-Term finished: {long_time:.3f} s "
            f"| valid={valid_milp(long_result)}",
            flush=True,
        )
        if not valid_milp(long_result):
            row.update({
                "status": "invalid_long_term",
                "error": "Long-Term did not return a feasible incumbent.",
            })
            return row

        print("      Short-Term is running...", flush=True)
        short_result, short_time = timed_call(
            MILP_BLO.Short_Term,
            data_to_short_term(copy.deepcopy(data), long_result),
        )
        row["short_term_runtime_s"] = short_time
        row["runtime_s"] = long_time + short_time
        print(
            f"      Short-Term finished: {short_time:.3f} s "
            f"| valid={valid_milp(short_result)}",
            flush=True,
        )
        if not valid_milp(short_result):
            row.update({
                "status": "invalid_short_term",
                "error": "Short-Term did not return a valid solution.",
            })
            return row

        row.update(metrics_from_milp(short_result, config[0]))
        row.update({"status": "ok", "success": 1})
    except Exception as error:
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    return row


def evaluate_milp(data, config, run, seed, method, function, converter) -> dict:
    row = base_row(config, run, seed, method)
    try:
        np.random.seed(int(seed) % (2**32 - 1))
        result, runtime = timed_call(function, converter(copy.deepcopy(data)))
        row["runtime_s"] = runtime
        if not valid_milp(result):
            row.update({
                "status": "invalid_result",
                "error": "Solver did not return a valid solution.",
            })
            return row
        row.update(metrics_from_milp(result, config[0]))
        row.update({"status": "ok", "success": 1})
    except Exception as error:
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    return row


def cuda_sync(agent) -> None:
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)


@torch.inference_mode()
def evaluate_ppo(agent, env, config, run, seed, method, precise_timing) -> dict:
    row = base_row(config, run, seed, method)
    try:
        state = env.reset(seed=seed)
        num_ues = len(state["state_UEs_req"])
        totals = {
            "score": 0.0,
            "throughput": 0.0,
            "latency": 0.0,
            "power": 0.0,
            "rbs": 0,
            "accepted": 0,
        }
        inference_time = 0.0 if precise_timing else np.nan

        cuda_sync(agent)
        start = time.perf_counter()
        for _ in range(num_ues + 1):
            if precise_timing:
                cuda_sync(agent)
                inference_start = time.perf_counter()

            policy = agent.act([state], deterministic=True)

            if precise_timing:
                cuda_sync(agent)
                inference_time += time.perf_counter() - inference_start

            state, reward, done, info = env.step(policy["env_actions"][0])
            totals["score"] += float(reward)

            if int(info.get("pi_k", info.get("accepted", 0))) == 1:
                totals["accepted"] += 1
                totals["throughput"] += float(info["R_k"])
                totals["latency"] += float(info["L_k"])
                totals["power"] += float(info["num_P_i_k"])
                totals["rbs"] += int(info["num_z_i_k"])

            if done:
                break
        else:
            raise RuntimeError("Episode exceeded the number of UEs.")

        cuda_sync(agent)
        row.update({
            "score": totals["score"],
            "accepted": totals["accepted"],
            "acceptance_rate": totals["accepted"] / max(num_ues, 1),
            "total_throughput_mbps": totals["throughput"] / 1e6,
            "total_latency_ms": totals["latency"] * 1e3,
            "allocated_rbs": totals["rbs"],
            "power_w": totals["power"],
            "runtime_s": time.perf_counter() - start,
            "inference_runtime_s": inference_time,
            "status": "ok",
            "success": 1,
        })
    except Exception as error:
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    return row


@torch.inference_mode()
def warmup(agent, env, seed: int) -> None:
    state = env.reset(seed=seed)
    cuda_sync(agent)
    agent.act([state], deterministic=True)
    cuda_sync(agent)


def append_rows(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_result(row: dict, num_ues: int) -> None:
    method = str(row["method"])
    runtime = float(row["runtime_s"])
    if int(row["success"]) != 1:
        message = (
            f"    [FAILED] {method:<12} | status={row['status']:<20} "
            f"| runtime={runtime:9.3f} s"
        )
        if row.get("error"):
            message += f" | {row['error']}"
        print(message, flush=True)
        return

    accepted = int(row["accepted"])
    total_latency = float(row["total_latency_ms"])
    mean_latency = total_latency / accepted if accepted else np.nan
    print(
        f"    [OK] {method:<12} "
        f"| score={float(row['score']):11.4f} "
        f"| accepted={accepted:3d}/{num_ues:<3d} "
        f"({float(row['acceptance_rate']):6.2%}) "
        f"| throughput={float(row['total_throughput_mbps']):11.3f} Mbps "
        f"| total_latency={total_latency:10.3f} ms "
        f"| mean_latency={mean_latency:8.3f} ms "
        f"| RBs={int(row['allocated_rbs']):4d} "
        f"| power={float(row['power_w']):9.4f} W "
        f"| runtime={runtime:9.3f} s",
        flush=True,
    )
    if method == "BLO":
        print(
            f"         BLO runtime = "
            f"{float(row['long_term_runtime_s']):.3f} s (Long-Term) + "
            f"{float(row['short_term_runtime_s']):.3f} s (Short-Term) = "
            f"{runtime:.3f} s",
            flush=True,
        )


def valid_row(row: dict) -> bool:
    """A saved row must contain a complete, finite, feasible result."""
    if int(row.get("success", 0)) != 1 or row.get("status") != "ok":
        return False
    required = (
        "score", "accepted", "acceptance_rate", "total_throughput_mbps",
        "total_latency_ms", "allocated_rbs", "power_w", "runtime_s",
    )
    try:
        values = np.asarray([float(row[name]) for name in required], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    if not np.all(np.isfinite(values)):
        return False
    accepted = int(round(float(row["accepted"])))
    return (
        0 <= accepted <= int(row["num_UEs"])
        and 0.0 <= float(row["acceptance_rate"]) <= 1.0
        and float(row["total_throughput_mbps"]) >= 0.0
        and float(row["total_latency_ms"]) >= 0.0
        and float(row["allocated_rbs"]) >= 0.0
        and float(row["power_w"]) >= 0.0
        and float(row["runtime_s"]) >= 0.0
    )


def describe(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(values.size)
    if n == 0:
        empty = {key: np.nan for key in ("mean", "std", "ci95", "median", "q25", "q75", "min", "max")}
        return empty | {"n": 0}
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(values.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(n)),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_summary(raw_csv: Path, summary_csv: Path) -> None:
    data = pd.read_csv(raw_csv)
    rows = []
    columns = ["config", "num_UEs", "num_RBs", "num_RUs", "num_DUs", "num_CUs", "method"]
    for keys, group in data.groupby(columns, sort=False, dropna=False):
        row = dict(zip(columns, keys))
        success = pd.to_numeric(group["success"], errors="coerce").fillna(0).astype(bool)
        row.update({
            "num_runs": int(len(group)),
            "num_success": int(success.sum()),
            "success_rate": float(success.mean()),
        })
        for metric in METRICS:
            source = group[metric] if metric in {"runtime_s", "inference_runtime_s"} else group.loc[success, metric]
            for statistic, value in describe(source).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


def print_summary(raw_csv: Path, config=None) -> None:
    data = pd.read_csv(raw_csv)
    if config is not None:
        data = data[data["config"] == config_name(config)]
        title = f"SUMMARY: {config_name(config)}"
    else:
        title = "FINAL SUMMARY: ALL CONFIGURATIONS"

    print(f"\n{'-' * 132}\n{title}\n{'-' * 132}", flush=True)
    print(
        f"{'Method':<14}{'Success':>10}{'Score':>14}{'Accept':>11}"
        f"{'Throughput':>16}{'Latency':>14}{'RBs':>10}{'Power':>12}{'Runtime':>13}",
        flush=True,
    )
    print(
        f"{'':<14}{'':>10}{'':>14}{'rate':>11}"
        f"{'Mbps':>16}{'ms':>14}{'':>10}{'W':>12}{'s':>13}",
        flush=True,
    )
    print("-" * 132, flush=True)

    for method in METHODS:
        group = data[data["method"] == method]
        if group.empty:
            continue
        success = pd.to_numeric(group["success"], errors="coerce").fillna(0).astype(bool)
        valid = group.loc[success]
        success_text = f"{int(success.sum())}/{len(group)}"
        runtime = pd.to_numeric(group["runtime_s"], errors="coerce").mean()

        if valid.empty:
            print(f"{method:<14}{success_text:>10}{'nan':>14}{'nan':>11}{'nan':>16}{'nan':>14}{'nan':>10}{'nan':>12}{runtime:>13.3f}", flush=True)
            continue

        mean = lambda field: pd.to_numeric(valid[field], errors="coerce").mean()
        print(
            f"{method:<14}{success_text:>10}{mean('score'):>14.4f}"
            f"{mean('acceptance_rate'):>10.2%}{mean('total_throughput_mbps'):>16.3f}"
            f"{mean('total_latency_ms'):>14.3f}{mean('allocated_rbs'):>10.2f}"
            f"{mean('power_w'):>12.4f}{runtime:>13.3f}",
            flush=True,
        )
    print("-" * 132, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect paired, valid MILP and PPO evaluations.")
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--runs", type=int, default=50, help="Valid paired runs required per configuration.")
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mosek-threads", type=int, default=4)
    parser.add_argument("--mip-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=5e-2)
    parser.add_argument("--precise-ppo-timing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if args.max_attempts < args.runs:
        raise ValueError("--max-attempts must be at least --runs.")
    if args.mosek_threads <= 0:
        raise ValueError("--mosek-threads must be positive.")
    if args.mip_time_limit <= 0.0:
        raise ValueError("--mip-time-limit must be positive.")
    if not 0.0 < args.mip_gap < 1.0:
        raise ValueError("--mip-gap must be in (0, 1).")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    for checkpoint in (PPO_GSA_CHECKPOINT, PPO_MLP_CHECKPOINT):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    args.results.mkdir(parents=True, exist_ok=True)
    raw_csv = args.results / "evaluation_runs.csv"
    summary_csv = args.results / "evaluation_summary.csv"
    metadata_json = args.results / "evaluation_metadata.json"
    existing = [path for path in (raw_csv, summary_csv, metadata_json) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Results already exist; use --overwrite or a new output directory.")
    for path in existing:
        path.unlink()

    from algorithms.PPO_GSA_MLP import PPOAgent as PPOGSAAgent
    from algorithms.PPO_MLP import PPOAgent as PPOMLPAgent

    gsa, gsa_meta = PPOGSAAgent.load(PPO_GSA_CHECKPOINT, device=device, restore_rng=False)
    mlp, mlp_meta = PPOMLPAgent.load(PPO_MLP_CHECKPOINT, device=device, restore_rng=False)
    configure_mosek(args.mosek_threads, args.mip_time_limit, args.mip_gap)

    started = datetime.now(timezone.utc)
    metadata = {
        "started_utc": started.isoformat(),
        "num_runs": args.runs,
        "seed_start": args.seed,
        "configs": [list(config) for config in CONFIGS],
        "methods": list(METHODS),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "mosek_threads": args.mosek_threads,
        "mip_time_limit_s": args.mip_time_limit,
        "mip_relative_gap": args.mip_gap,
        "max_attempts_per_config": args.max_attempts,
        "ppo_deterministic": True,
        "precise_ppo_timing": args.precise_ppo_timing,
        "ppo_gsa_checkpoint": str(PPO_GSA_CHECKPOINT.resolve()),
        "ppo_mlp_checkpoint": str(PPO_MLP_CHECKPOINT.resolve()),
        "ppo_gsa_metadata": gsa_meta,
        "ppo_mlp_metadata": mlp_meta,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"\nDevice: {device} | CPU: {os.cpu_count()} | MOSEK threads: {args.mosek_threads}\n"
        f"Valid paired runs/config: {args.runs} | Seed start: {args.seed}\n"
        f"MIP limit: {args.mip_time_limit:g} s | MIP gap: {args.mip_gap:.2%}\n"
        f"Output: {args.results.resolve()}",
        flush=True,
    )

    for config_index, config in enumerate(CONFIGS, start=1):
        name = config_name(config)
        num_ues = config[0]
        print(
            f"\n{'=' * 132}\nCONFIG {config_index}/{len(CONFIGS)}: {name}\n{'=' * 132}",
            flush=True,
        )

        gsa_env, mlp_env = RLEnv(*config), RLEnv(*config)
        print("Warming up PPO agents...", flush=True)
        warmup(gsa, gsa_env, args.seed)
        warmup(mlp, mlp_env, args.seed)

        valid_count = 0
        attempt = 0
        while valid_count < args.runs and attempt < args.max_attempts:
            attempt += 1
            run = valid_count + 1
            seed = args.seed + attempt - 1
            print(
                f"\n{'-' * 132}\nRUN {run:02d}/{args.runs} | seed={seed} | {name}\n{'-' * 132}",
                flush=True,
            )

            state = RLEnv(*config).reset(seed=seed)
            milp_data = state_to_MILP(state)
            # MILPs run first: if one is invalid, reject this snapshot before
            # spending time on the remaining methods.  PPO is evaluated only
            # after every MILP produced a valid result.
            evaluations = (
                ("Nearest-RU", lambda: evaluate_milp(milp_data, config, run, seed, "Nearest-RU", MILP_NEAREST.Nearest_RU, data_to_nearest_ru)),
                ("RoundRobin", lambda: evaluate_milp(milp_data, config, run, seed, "RoundRobin", MILP_ROUNDROBIN.RoundRobin_RU, data_to_roundrobin_ru)),
                ("Random-RU", lambda: evaluate_milp(milp_data, config, run, seed, "Random-RU", MILP_RANDOM.Random_RU, data_to_random_ru)),
                ("BLO", lambda: evaluate_blo(milp_data, config, run, seed)),
                ("PPO-GSA", lambda: evaluate_ppo(gsa, gsa_env, config, run, seed, "PPO-GSA", args.precise_ppo_timing)),
                ("PPO-MLP", lambda: evaluate_ppo(mlp, mlp_env, config, run, seed, "PPO-MLP", args.precise_ppo_timing)),
            )

            completed = []
            failed_method = None
            for method, evaluator in evaluations:
                print(f"  [START] {method}", flush=True)
                row = evaluator()
                completed.append(row)
                print_result(row, num_ues)
                if not valid_row(row):
                    failed_method = method
                    break

            if failed_method is not None:
                print(
                    f"  [REJECTED] seed={seed}: {failed_method} invalid. "
                    f"Valid paired runs remain {valid_count}/{args.runs}.",
                    flush=True,
                )
                continue

            by_method = {row["method"]: row for row in completed}
            accepted_rows = [by_method[method] for method in METHODS]
            append_rows(raw_csv, accepted_rows)
            write_summary(raw_csv, summary_csv)
            valid_count += 1
            print(
                f"  [ACCEPTED] paired run {valid_count:02d}/{args.runs} "
                f"from seed={seed}; all {len(METHODS)} methods are valid.",
                flush=True,
            )

        if valid_count < args.runs:
            raise RuntimeError(
                f"{name}: collected only {valid_count}/{args.runs} valid paired runs "
                f"after {attempt} attempts. Fix MILP incumbent handling or increase "
                "--max-attempts; blindly retrying is not efficient."
            )

        print_summary(raw_csv, config)

    write_summary(raw_csv, summary_csv)
    print_summary(raw_csv)

    finished = datetime.now(timezone.utc)
    metadata.update({
        "finished_utc": finished.isoformat(),
        "elapsed_s": (finished - started).total_seconds(),
        "raw_results": str(raw_csv.resolve()),
        "summary": str(summary_csv.resolve()),
    })
    metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(
        f"\n{'=' * 100}\nEVALUATION FINISHED\n"
        f"Elapsed: {(finished - started).total_seconds():.3f} s\n"
        f"Raw results: {raw_csv.resolve()}\n"
        f"Summary:     {summary_csv.resolve()}\n"
        f"Metadata:    {metadata_json.resolve()}\n{'=' * 100}",
        flush=True,
    )


if __name__ == "__main__":
    main()
