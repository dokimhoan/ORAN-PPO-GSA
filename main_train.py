"""Huấn luyện tuần tự PPO-GSA-MLP và PPO-MLP trên nhiều kích thước RAN."""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SEED = 10
RESULTS_ROOT = Path("results_train_10")
RESUME = False

# Phù hợp RTX 5070 Ti 16 GB và CPU 32 luồng.
NUM_ENVS = 64
CPU_WORKERS = 16
ROLLOUT_STEPS = 8192
MINIBATCH_SIZE = 1024
TORCH_THREADS = 4

# Hai module PPO đọc các giá trị này khi được import.
os.environ.setdefault("PPO_NUM_ENVS", str(NUM_ENVS))
os.environ.setdefault("PPO_CPU_WORKERS", str(CPU_WORKERS))
os.environ.setdefault("PPO_STEPS_PER_UPDATE", str(ROLLOUT_STEPS))
os.environ.setdefault("PPO_MINIBATCH_SIZE", str(MINIBATCH_SIZE))

from algorithms import PPO_GSA_MLP, PPO_MLP  # noqa: E402
from env.RLEnv import RLEnv  # noqa: E402


@dataclass(frozen=True)
class SizeRange:
    ue: tuple[int, int] = (20, 50)
    rb: tuple[int, int] = (50, 135)
    ru: tuple[int, int] = (5, 7)
    du: tuple[int, int] = (3, 5)
    cu: tuple[int, int] = (3, 5)


SIZES = SizeRange()


def configure_runtime():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(TORCH_THREADS)

    if not torch.cuda.is_available():
        print("CUDA không khả dụng; huấn luyện bằng CPU.")
        return

    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 2**30:.1f} GB | "
        f"CUDA: {torch.version.cuda}"
    )


def make_env(num_ues: int, episode_index: int = 0):
    """Sinh một snapshot RAN mới, xác định bởi seed của episode."""
    seed = SEED * 1_000_003 + int(episode_index or 0)
    rng = np.random.default_rng(seed)

    def draw(bounds):
        return int(rng.integers(bounds[0], bounds[1] + 1))

    dimensions = (
        int(num_ues),
        draw(SIZES.rb),
        draw(SIZES.ru),
        draw(SIZES.du),
        draw(SIZES.cu),
    )

    # Môi trường dùng NumPy RNG toàn cục; không để nó làm lệch RNG của PPO.
    numpy_state = np.random.get_state()
    np.random.seed(seed % (2**32 - 1))
    try:
        return RLEnv(*dimensions)
    finally:
        np.random.set_state(numpy_state)


def train(module):
    output_dir = RESULTS_ROOT / module.RESULT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(SIZES.ue[0])

    print(f"\n{'=' * 80}\nTRAIN {module.DISPLAY_NAME}\n{'=' * 80}")
    if RESUME:
        checkpoint = output_dir / f"{module.CHECKPOINT_PREFIX}_latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        agent, _ = module.PPOAgent.load(
            checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            restore_rng=True,
        )
    else:
        agent = module.PPOAgent.from_state(env.get_state())

    history = module.train_agent(
        agent=agent,
        env=env,
        results_root=output_dir,
        env_factory=make_env,
        ue_range=SIZES.ue,
    )

    del agent, env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def main_train():
    configure_runtime()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        module.DISPLAY_NAME: train(module)
        for module in (PPO_GSA_MLP, PPO_MLP)
    }


if __name__ == "__main__":
    main_train()
