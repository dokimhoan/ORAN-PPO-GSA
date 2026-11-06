"""Plot training curves and saved scientific evaluation summaries.

Performance bars use mean +/- 95% confidence interval. Runtime bars use
median and the interquartile range because solver time is strongly skewed by
hard instances and timeouts. No metric is recomputed from raw evaluation data.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_ROOT = Path("results_train_10")
EVAL_ROOT = Path("results_eval_20")
OUTPUT_ROOT = Path("results_plot_20")
METHODS = ("BLO", "PPO-GSA", "PPO-MLP", "Nearest-RU", "RoundRobin", "Random-RU")
LABEL = {method: method for method in METHODS}
STYLE = {
    "BLO": ("#2b9638", ".."),
    "PPO-GSA": ("#c58b00", "//"),
    "PPO-MLP": ("#5594cc", "xx"),
    "Nearest-RU": ("#1426a0", "oo"),
    "RoundRobin": ("#b51de8", ".."),
    "Random-RU": ("#111111", "\\"),
}


@dataclass(frozen=True)
class Metric:
    name: str
    file_name: str
    ylabel: str
    statistic: str = "mean"
    error: str = "ci95"
    log_scale: bool = False


METRICS = (
    Metric("acceptance_rate", "acceptance_rate", "Acceptance rate"),
    Metric("total_throughput_mbps", "total_throughput", "Avg. total throughput (Mbps)"),
    Metric("total_latency_ms", "average_total_latency", "Average total latency (ms)"),
    Metric("score", "reward", "Average objective value"),
    Metric("runtime_s", "runtime", "Computing time (s)", "median", "iqr", True),
)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "text.usetex": shutil.which("latex") is not None,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.labelsize": 21,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.2,
        "hatch.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def save_figure(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def training_files(root: Path) -> dict[str, Path]:
    candidates = {
        "PPO-GSA": (
            root / "PPO_GSA_MLP" / "PPOGSA_train_reward_hist.txt",
            root / "PPO_GSA" / "PPOGSA_train_reward_hist.txt",
        ),
        "PPO-MLP": (
            root / "PPO_MLP" / "PPOMLP_train_reward_hist.txt",
            root / "PPO_MLP_SMALL" / "PPOMLPSMALL_train_reward_hist.txt",
        ),
    }
    files = {}
    for method, paths in candidates.items():
        files[method] = next((path for path in paths if path.is_file()), None)
        if files[method] is None:
            raise FileNotFoundError(f"Không tìm thấy training history của {method}.")
    return files


def plot_training(root: Path, output: Path, window: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {"PPO-GSA": "#c58b00", "PPO-MLP": "#7c2ae8"}
    max_episode = 1
    for method, path in training_files(root).items():
        reward = pd.to_numeric(pd.read_csv(path, header=None).iloc[:, 0], errors="coerce").dropna().to_numpy(float)
        episode = np.arange(1, reward.size + 1)
        smooth = pd.Series(reward).rolling(max(1, min(window, reward.size)), min_periods=1).mean()
        ax.plot(episode, reward, color=colors[method], alpha=0.12, linewidth=0.7)
        ax.plot(episode, smooth, color=colors[method], linewidth=2.2, label=method)
        max_episode = max(max_episode, int(episode[-1]))
    ax.set(xlabel="Episode", ylabel="Total reward", xlim=(1, max_episode))
    ax.margins(x=0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "training_reward_curve")


def load_summary(root: Path) -> pd.DataFrame:
    path = root / "evaluation_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    data = pd.read_csv(path)
    required = {"config", "num_UEs", "method", "num_runs", "num_success", "success_rate"}
    for metric in METRICS:
        required.add(f"{metric.name}_{metric.statistic}")
        required.update(
            {f"{metric.name}_q25", f"{metric.name}_q75"}
            if metric.error == "iqr" else {f"{metric.name}_{metric.error}"}
        )
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("evaluation_summary.csv thiếu cột: " + ", ".join(sorted(missing)))
    data["num_UEs"] = pd.to_numeric(data["num_UEs"], errors="coerce")
    return data


def y_limits(metric: Metric, single: bool, lower: float, upper: float) -> tuple[float, float]:
    if metric.name in {"acceptance_rate", "success_rate"}:
        return 0.0, 1.10 if single else 1.35
    lower = min(0.0, lower)
    span = max(upper - lower, 1e-12)
    scale = {
        "total_throughput_mbps": (1.30, 1.35),
        "total_latency_ms": (1.25, 1.35),
        "score": (1.30, 1.40),
    }.get(metric.name, (1.20, 1.25))[0 if single else 1]
    return lower, lower + scale * span


def plot_bars(data: pd.DataFrame, metric: Metric, output: Path, ues: tuple[int, ...]) -> None:
    subset = data[data["num_UEs"].isin(ues)].copy()
    if subset.empty:
        return
    single = len(ues) == 1
    configs = list(dict.fromkeys(subset.sort_values("num_UEs")["config"].astype(str)))
    methods = [method for method in METHODS if method in set(subset["method"])]
    x = np.arange(len(configs), dtype=float)
    group_width, all_lower, all_upper = 0.82, [], []
    width = group_width / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(6.6 if single else max(8.0, 1.9 * len(configs) + 3.0), 5.8))

    for index, method in enumerate(methods):
        values, low_errors, high_errors = [], [], []
        for config in configs:
            rows = subset[(subset["config"] == config) & (subset["method"] == method)]
            if rows.empty:
                values.append(np.nan); low_errors.append(0.0); high_errors.append(0.0)
                continue
            row = rows.iloc[0]
            value = float(row[f"{metric.name}_{metric.statistic}"])
            if metric.error == "iqr":
                low = max(0.0, value - float(row[f"{metric.name}_q25"]))
                high = max(0.0, float(row[f"{metric.name}_q75"]) - value)
            else:
                low = high = max(0.0, float(row[f"{metric.name}_{metric.error}"]))
            values.append(value); low_errors.append(low); high_errors.append(high)

        values = np.asarray(values, float)
        low_errors, high_errors = np.asarray(low_errors), np.asarray(high_errors)
        valid = np.isfinite(values)
        if metric.log_scale:
            valid &= values > 0
        if not np.any(valid):
            continue
        if metric.log_scale:
            low_errors = np.minimum(low_errors, values * 0.95)
        positions = x + (index - (len(methods) - 1) / 2) * width
        color, hatch = STYLE[method]
        ax.bar(
            positions[valid], values[valid], width=width * 0.92,
            yerr=np.vstack([low_errors[valid], high_errors[valid]]),
            facecolor="white", edgecolor=color, linewidth=1.8, hatch=hatch,
            error_kw={"ecolor": color, "elinewidth": 1.2, "capsize": 4, "capthick": 1.2},
            label=LABEL[method], zorder=3,
        )
        all_lower.extend((values[valid] - low_errors[valid]).tolist())
        all_upper.extend((values[valid] + high_errors[valid]).tolist())

    if single:
        ax.set_xticks([])
    else:
        ax.set_xticks(x, [rf"$\mathcal{{K}}={int(subset[subset['config'] == c]['num_UEs'].iloc[0])}$" for c in configs])
        ax.set_xlabel(r"Number of UEs ($|\mathcal{K}|$)")
    ax.set_ylabel(metric.ylabel)
    ax.set_xlim(-group_width / 2, len(configs) - 1 + group_width / 2)
    ax.margins(x=0)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    finite_lower = np.asarray(all_lower)[np.isfinite(all_lower)]
    finite_upper = np.asarray(all_upper)[np.isfinite(all_upper)]
    if metric.log_scale:
        positive = finite_lower[finite_lower > 0]
        if positive.size and finite_upper.size:
            ax.set_yscale("log")

            y_min = 10 ** np.floor(np.log10(positive.min()))
            y_scale = 8.0 if single else 9.0
            y_max = float(finite_upper.max()) * y_scale

            ax.set_ylim(y_min, y_max)
            
            
    elif finite_upper.size:
        ax.set_ylim(*y_limits(metric, single, float(finite_lower.min()), float(finite_upper.max())))
        if metric.name == "acceptance_rate":
            ax.set_yticks(np.linspace(0, 1, 6))

    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=15 if single else 16,
              columnspacing=1.25, handlelength=1.8, handletextpad=0.55)
    fig.tight_layout()
    suffix = "K20" if single else "K30_40_50"
    save_figure(fig, output, f"evaluation_{metric.file_name}_{suffix}_bar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved scientific evaluation statistics.")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--smooth-window", type=int, default=200)
    parser.add_argument("--evaluation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    if not args.evaluation_only:
        plot_training(args.train_root, args.output_root, args.smooth_window)
    data = load_summary(args.eval_root)
    for metric in METRICS:
        plot_bars(data, metric, args.output_root, (20,))
        plot_bars(data, metric, args.output_root, (30, 40, 50))
    print(f"Saved figures to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
