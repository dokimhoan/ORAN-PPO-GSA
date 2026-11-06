"""Vẽ learning curve và grouped bar chart cho kết quả evaluation.

- Bỏ Long_Term khỏi toàn bộ hình evaluation.
- Short_Term được hiển thị với tên BLO.
- Runtime của Short_Term đã là tổng Long-Term + Short-Term từ main_evaluation.
- Tất cả metric evaluation được vẽ bằng grouped bar chart.
- Runtime giữ đơn vị giây và dùng trục y logarithmic.
- Nếu CSV có các reward component thì tự động vẽ thêm từng term.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# 1. CẤU HÌNH
# ======================================================================

TRAIN_ROOT = Path("./results_train_10")
EVAL_ROOT = Path("./results_eval_10")
OUTPUT_ROOT = Path("./results_plot_10")

SMOOTH_WINDOW = 200
PNG_DPI = 300

METHOD_ORDER = (
    "Short_Term",
    "PPO_GSA",
    "PPO_MLP",
    "Nearest_RU",
    "RoundRobin_RU",
    "Random_RU",
)

METHOD_LABEL = {
    "Short_Term": "BLO",
    "PPO_GSA": "PPO-GSA",
    "PPO_MLP": "PPO-MLP",
    "Nearest_RU": "Nearest-RU",
    "RoundRobin_RU": "RoundRobin",
    "Random_RU": "Random-RU",
}

METHOD_STYLE = {
    # edge color, hatch; mặt cột để trắng giống các hình mẫu.
    "Short_Term": ("#2b9638", ".."),
    "PPO_GSA": ("#c58b00", "//"),
    "PPO_MLP": ("#5594cc", "xx"),
    "Nearest_RU": ("#1426a0", "oo"),
    "RoundRobin_RU": ("#b51de8", ".."),
    "Random_RU": ("#111111", "\\"),
}

TRAIN_STYLE = {"PPO-GSA": "#c58b00", "PPO-MLP": "#7c2ae8"}

CONFIG_COLUMNS = ("num_UEs", "num_RBs", "num_RUs", "num_DUs", "num_CUs")
CONFIG_LABEL = {
    "num_UEs": "UEs",
    "num_RBs": "RBs",
    "num_RUs": "RUs",
    "num_DUs": "DUs",
    "num_CUs": "CUs",
}


@dataclass(frozen=True)
class MetricSpec:
    column: str
    file_name: str
    ylabel: str
    unit_interval: bool = False
    log_scale: bool = False


METRICS = (
    MetricSpec("acceptance_rate", "acceptance_rate", "Acceptance rate", unit_interval=True),
    MetricSpec("total_throughput_mbps", "total_throughput", "Avg. total throughput (Mbps)"),
    MetricSpec("total_latency_s", "average_total_latency", "Average total latency (ms)"),
    MetricSpec("score", "reward", "Average objective value"),
    MetricSpec("runtime_s", "runtime", "Computing time (s)", log_scale=True),
)

# Chỉ vẽ reward term khi CSV thực sự có dữ liệu tương ứng.
REWARD_TERM_ALIASES = {
    "acceptance_reward_term": (
        "acceptance_reward_term", "acceptance_term", "acc_term", "reward_acc", "acceptance_reward"
    ),
    "throughput_reward_term": (
        "throughput_reward_term", "throughput_term", "thr_term", "reward_thr", "throughput_reward"
    ),
    "latency_penalty_term": (
        "latency_penalty_term", "latency_term", "lat_pen", "reward_lat", "latency_penalty"
    ),
}

REWARD_TERM_METRICS = (
    MetricSpec("acceptance_reward_term", "reward_acceptance_term", "Acceptance reward term"),
    MetricSpec("throughput_reward_term", "reward_throughput_term", "Throughput reward term"),
    MetricSpec("latency_penalty_term", "reward_latency_term", "Latency penalty term"),
)


# ======================================================================
# 2. HÀM DÙNG CHUNG
# ======================================================================

def configure_matplotlib() -> None:
    latex_available = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.labelsize": 21,
        "axes.titlesize": 21,
        "legend.fontsize": 17,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.2,
        "axes.unicode_minus": False,
        "hatch.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": PNG_DPI,
    })
    if not latex_available:
        print("[plot] Không tìm thấy latex.exe; dùng MathText Computer Modern.")


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_root / f"{stem}{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def natural_key(value: object) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def ordered_methods(methods: Iterable[str]) -> list[str]:
    available = {str(method) for method in methods if str(method) != "Long_Term"}
    known = [method for method in METHOD_ORDER if method in available]
    unknown = sorted(available.difference(METHOD_ORDER), key=natural_key)
    return known + unknown


def style_for(method: str, index: int) -> tuple[str, str]:
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    hatches = ("//", "xx", "oo", "\\")
    return colors[index % len(colors)], hatches[index % len(hatches)]


# ======================================================================
# 3. TRAINING CURVE
# ======================================================================

def resolve_training_files(train_root: Path) -> dict[str, Path]:
    expected = {
        "PPO-GSA": (
            train_root / "PPO_GSA_MLP" / "PPOGSA_train_reward_hist.txt",
            train_root / "PPO_GSA" / "PPOGSA_train_reward_hist.txt",
        ),
        "PPO-MLP": (
            train_root / "PPO_MLP_SMALL" / "PPOMLPSMALL_train_reward_hist.txt",
            train_root / "PPO_MLP" / "PPOMLP_train_reward_hist.txt",
        ),
    }

    resolved = {}
    for algorithm, candidates in expected.items():
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Không tìm thấy reward history của {algorithm}.")
        resolved[algorithm] = path
    return resolved


def read_numeric_history(path: Path) -> np.ndarray:
    values = pd.to_numeric(pd.read_csv(path, header=None, comment="#").iloc[:, 0], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"File không chứa dữ liệu hợp lệ: {path}")
    return values


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), values.size))
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy(float)


def plot_training_curve(train_root: Path, output_root: Path, smooth_window: int) -> list[Path]:
    files = resolve_training_files(train_root)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    max_episode = 1

    for algorithm, path in files.items():
        reward = read_numeric_history(path)
        episode = np.arange(1, reward.size + 1)
        max_episode = max(max_episode, int(episode[-1]))
        color = TRAIN_STYLE[algorithm]
        ax.plot(episode, reward, color=color, alpha=0.12, linewidth=0.7, label="_nolegend_")
        ax.plot(episode, moving_average(reward, smooth_window), color=color, linewidth=2.2, label=algorithm)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_xlim((1, max_episode) if max_episode > 1 else (0.5, 1.5))
    ax.margins(x=0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_root, "training_reward_curve")
    return list(files.values())


# ======================================================================
# 4. ĐỌC KẾT QUẢ EVALUATION
# ======================================================================

def load_evaluation_results(eval_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    master = eval_root / "all_evaluation_results.csv"
    files = [master] if master.is_file() else sorted(eval_root.glob("*_all_results.csv"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy CSV evaluation trong {eval_root}")

    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data.columns = [str(col).strip() for col in data.columns]

    aliases = {
        "algorithm": "method",
        "reward": "score",
        "throughput_mbps": "total_throughput_mbps",
        "throughput_bps": "total_throughput_bps",
        "latency_ms": "mean_latency_ms",
        "time_s": "runtime_s",
    }
    for old, new in aliases.items():
        if old in data and new not in data:
            data[new] = data[old]

    if "method" not in data:
        raise ValueError("CSV evaluation thiếu cột 'method'.")

    data["method"] = data["method"].astype(str).str.strip()
    # Hỗ trợ CSV sinh bởi bản baseline có tên thư mục/method cũ.
    data["method"] = data["method"].replace({"PPO_MLP_SMALL": "PPO_MLP"})

    # main_evaluation đã lưu Short_Term.runtime_s = Long_Term + Short_Term.
    # Không cộng lại ở đây vì sẽ làm runtime BLO bị tính hai lần.
    data = data[data["method"] != "Long_Term"].copy()

    if "acceptance_rate" not in data and {"accepted", "num_UEs"} <= set(data):
        data["acceptance_rate"] = pd.to_numeric(data["accepted"], errors="coerce") / pd.to_numeric(data["num_UEs"], errors="coerce").clip(lower=1)

    if "total_throughput_mbps" not in data and "total_throughput_bps" in data:
        data["total_throughput_mbps"] = pd.to_numeric(data["total_throughput_bps"], errors="coerce") / 1e6

    if "mean_latency_ms" not in data:
        if "mean_latency_s" in data:
            data["mean_latency_ms"] = pd.to_numeric(data["mean_latency_s"], errors="coerce") * 1e3
        elif {"total_latency_s", "accepted"} <= set(data):
            accepted = pd.to_numeric(data["accepted"], errors="coerce").clip(lower=1)
            data["mean_latency_ms"] = pd.to_numeric(data["total_latency_s"], errors="coerce") / accepted * 1e3

    for canonical, candidates in REWARD_TERM_ALIASES.items():
        if canonical not in data:
            source = next((name for name in candidates if name in data), None)
            if source is not None:
                data[canonical] = data[source]

    required = {metric.column for metric in METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("CSV evaluation thiếu cột: " + ", ".join(sorted(missing)))

    numeric = required.union(CONFIG_COLUMNS).union({"run", "seed"}).union(REWARD_TERM_ALIASES)
    for column in numeric.intersection(data.columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "config" not in data:
        present = [c for c in CONFIG_COLUMNS if c in data]
        if present:
            data["config"] = data[present].apply(
                lambda row: "_".join(f"{c.replace('num_', '')}{int(v)}" for c, v in row.items() if pd.notna(v)), axis=1
            )
        else:
            data["config"] = "default"

    data["config"] = data["config"].fillna("default").astype(str).str.strip()
    data = data[np.isfinite(data["score"].to_numpy(float))].copy()

    dedup_keys = [key for key in ("config", "run", "seed", "method") if key in data]
    if len(dedup_keys) >= 2:
        data = data.drop_duplicates(subset=dedup_keys, keep="last")

    if data.empty:
        raise ValueError("Không có evaluation run hợp lệ.")

    return data, files


# ======================================================================
# 5. TRỤC CONFIGURATION
# ======================================================================

def ordered_configs(data: pd.DataFrame) -> list[str]:
    configs = list(data["config"].unique())
    present = [c for c in CONFIG_COLUMNS if c in data]
    if not present:
        return sorted(configs, key=natural_key)

    values = {}
    for config in configs:
        group = data[data["config"] == config]
        key = []
        for column in present:
            finite = group[column].dropna()
            key.append(float(finite.iloc[0]) if not finite.empty else np.inf)
        values[config] = tuple(key)
    return sorted(configs, key=lambda c: values[c])


def configuration_axis(data: pd.DataFrame, configs: list[str]) -> tuple[list[str], str]:
    # Trục x chỉ biểu diễn số UE, không đưa RB/RU/DU/CU vào nhãn.
    if "num_UEs" in data:
        labels = []
        for config in configs:
            values = data.loc[data["config"] == config, "num_UEs"].dropna()
            labels.append(
                rf"$\mathcal{{K}}={int(values.iloc[0])}$"
                if not values.empty else config
            )
        return labels, r"Number of UEs ($|\mathcal{K}|$)"

    return configs, "Configuration"


# ======================================================================
# 6. GROUPED BAR CHART EVALUATION
# ======================================================================

def metric_summary(values: np.ndarray, error_kind: str) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0.0

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    error = std if error_kind == "std" else 1.96 * std / np.sqrt(values.size)
    return mean, float(error)


def _plot_metric_bars(
    data: pd.DataFrame,
    metric: MetricSpec,
    error_kind: str,
    output_root: Path,
    suffix: str,
    single_group: bool,
) -> None:
    if metric.column not in data:
        return

    methods = ordered_methods(data["method"].unique())
    configs = ordered_configs(data)
    labels, xlabel = configuration_axis(data, configs)
    x = np.arange(len(configs), dtype=float)

    if not methods or not configs:
        return

    group_width = 0.82
    bar_width = group_width / len(methods)
    fig_width = 6.6 if single_group else max(8.0, 1.9 * len(configs) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    all_means, all_lowers, all_uppers = [], [], []

    for index, method in enumerate(methods):
        means, errors = [], []
        for config in configs:
            values = data.loc[
                (data["method"] == method) & (data["config"] == config),
                metric.column,
            ].to_numpy(float)
            if metric.column == "total_latency_s":
                values = values * 1e3
            if metric.log_scale:
                values = values[values > 0.0]
            mean, error = metric_summary(values, error_kind)
            means.append(mean)
            errors.append(error)

        means = np.asarray(means, dtype=float)
        errors = np.asarray(errors, dtype=float)
        valid = np.isfinite(means)
        if not np.any(valid):
            continue

        color, hatch = style_for(method, index)
        positions = x + (index - (len(methods) - 1) / 2.0) * bar_width

        # Trục log không chấp nhận error bar đi qua 0.
        lower_error = errors.copy()
        if metric.log_scale:
            lower_error = np.minimum(lower_error, means * 0.95)
            yerr = np.vstack([lower_error[valid], errors[valid]])
        else:
            yerr = errors[valid]

        ax.bar(
            positions[valid], means[valid], width=bar_width * 0.92,
            yerr=yerr,
            facecolor="white", edgecolor=color, linewidth=1.8,
            hatch=hatch,
            error_kw={
                "ecolor": color, "elinewidth": 1.2,
                "capsize": 4, "capthick": 1.2,
            },
            label=METHOD_LABEL.get(method, method.replace("_", "-")),
            zorder=3,
        )
        all_means.extend(means[valid])
        all_lowers.extend((means[valid] - lower_error[valid]).tolist())
        all_uppers.extend((means[valid] + errors[valid]).tolist())

    if single_group:
        ax.set_xticks([])
        ax.set_xlabel("")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, linespacing=1.15)
        ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.ylabel)
    ax.set_xlim(-group_width / 2.0, len(configs) - 1 + group_width / 2.0)
    ax.margins(x=0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.75, alpha=0.65)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.55, alpha=0.45)
    ax.set_axisbelow(True)

    if metric.unit_interval:
        # Scale riêng theo nhóm K, legend vẫn nằm bên trong trục.
        ax.set_ylim(0.0, 1.10 if single_group else 1.35)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
    elif metric.log_scale:
        positive_lower = np.asarray(all_lowers, dtype=float)
        positive_lower = positive_lower[np.isfinite(positive_lower) & (positive_lower > 0)]
        positive_upper = np.asarray(all_uppers, dtype=float)
        positive_upper = positive_upper[np.isfinite(positive_upper) & (positive_upper > 0)]
        if positive_lower.size and positive_upper.size:
            lower = 10.0 ** np.floor(np.log10(positive_lower.min()))
            upper = 10.0 ** np.ceil(np.log10(positive_upper.max()))
            if lower == upper:
                lower /= 10.0
                upper *= 10.0
            ax.set_yscale("log")
            ax.set_ylim(lower, upper)
    elif all_means:
        lower = min(0.0, float(np.nanmin(all_lowers)))
        upper = float(np.nanmax(all_uppers))
        span = max(upper - lower, 1e-12)
        if metric.column == "total_latency_s":
            scale = 1.25 if single_group else 1.3
            ax.set_ylim(lower, lower + scale * span)
        elif metric.column == "total_throughput_mbps":
            # Tăng riêng khoảng phía trên để legend không chạm cột/error bar.
            scale = 1.30 if single_group else 1.35
            ax.set_ylim(lower, lower + scale * span)
        elif metric.column == "score":
            scale = 1.30 if single_group else 1.3
            ax.set_ylim(lower, lower + scale * span)
        else:
            # Chừa chỗ cho ba hàng legend nằm trong trục.
            ax.set_ylim(lower, upper + 0.26 * span)

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        fontsize=15 if single_group else 16,
        columnspacing=1.25, handlelength=1.8, handletextpad=0.55,
    )
    fig.tight_layout()
    save_figure(fig, output_root, f"evaluation_{metric.file_name}_{suffix}_bar")


def plot_metric_bars(
    data: pd.DataFrame,
    metric: MetricSpec,
    error_kind: str,
    output_root: Path,
) -> None:
    """Tách K=20 và K=30/40/50 thành hai figure độc lập."""
    if "num_UEs" not in data:
        raise ValueError("CSV evaluation thiếu cột 'num_UEs'.")

    num_ues = pd.to_numeric(data["num_UEs"], errors="coerce")
    groups = (
        (data[num_ues == 20].copy(), "K20", True),
        (data[num_ues.isin([30, 40, 50])].copy(), "K30_40_50", False),
    )
    for subset, suffix, single_group in groups:
        if not subset.empty:
            _plot_metric_bars(
                subset,
                metric,
                error_kind,
                output_root,
                suffix,
                single_group,
            )


def plot_evaluation(eval_root: Path, output_root: Path, error_kind: str) -> tuple[list[str], list[Path], list[str]]:
    data, source_files = load_evaluation_results(eval_root)

    # Xóa các figure cũ đã bị loại khỏi bộ kết quả hiện tại.
    obsolete = (
        "accepted_ues", "allocated_power", "allocated_rbs",
        "average_latency", "total_latency",
    )
    for stem in obsolete:
        for path in output_root.glob(f"evaluation_{stem}*_bar.*"):
            if path.suffix.lower() in {".png", ".pdf"}:
                path.unlink()

    for metric in METRICS:
        plot_metric_bars(data, metric, error_kind, output_root)

    plotted_terms = []
    for metric in REWARD_TERM_METRICS:
        if metric.column in data and np.isfinite(pd.to_numeric(data[metric.column], errors="coerce")).any():
            plot_metric_bars(data, metric, error_kind, output_root)
            plotted_terms.append(metric.column)

    return ordered_configs(data), source_files, plotted_terms


# ======================================================================
# 7. COMMAND LINE
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vẽ training curve và grouped bar chart evaluation."
    )
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--error", choices=("std", "ci95"), default="std")
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help="Chỉ vẽ các bar chart evaluation, không đọc history training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window phải > 0.")

    configure_matplotlib()
    args.output_root.mkdir(parents=True, exist_ok=True)

    training_files = [] if args.evaluation_only else plot_training_curve(
        args.train_root, args.output_root, args.smooth_window
    )
    configs, evaluation_files, reward_terms = plot_evaluation(args.eval_root, args.output_root, args.error)

    print("Hoàn thành vẽ hình.")
    print(
        "  Training files : "
        + (", ".join(map(str, training_files)) if training_files else "bỏ qua")
    )
    print(f"  Evaluation     : {', '.join(map(str, evaluation_files))}")
    print(f"  Configs        : {', '.join(configs)}")
    print(f"  Reward terms   : {', '.join(reward_terms) if reward_terms else 'không có trong CSV'}")
    print(f"  Output         : {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
