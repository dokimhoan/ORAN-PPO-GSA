"""PPO với GraphSAGE encoder cho bài toán cấp phát tài nguyên RAN."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import speed_of_light_mps


# ======================================================================
# 1. CẤU HÌNH HUẤN LUYỆN
# ======================================================================

DISPLAY_NAME = "PPO-GSA-MLP"
RESULT_DIR_NAME = "PPO_GSA_MLP"
CHECKPOINT_PREFIX = "PPOGSA"

MAX_EPISODES = int(os.getenv("PPO_MAX_EPISODES", "50000"))
TRAIN_SEED = int(os.getenv("PPO_TRAIN_SEED", "100000"))
PRINT_EVERY = int(os.getenv("PPO_PRINT_EVERY", "10"))
CHECKPOINT_EVERY = int(os.getenv("PPO_CHECKPOINT_EVERY", "1000"))
VALIDATE_EVERY = int(os.getenv("PPO_VALIDATE_EVERY", "1000"))
VALIDATION_EPISODES = int(os.getenv("PPO_VALIDATION_EPISODES", "32"))
VALIDATION_SEED_START = int(os.getenv("PPO_VALIDATION_SEED_START", "900000"))

GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
DEFAULT_MINIBATCH = (
    2_048 if GPU_MEMORY_GB >= 20.0 else
    1_024 if GPU_MEMORY_GB >= 10.0 else
    512 if GPU_MEMORY_GB > 0.0 else
    256
)
DEFAULT_NUM_ENVS = (
    128 if GPU_MEMORY_GB >= 20.0 else
    64 if GPU_MEMORY_GB >= 10.0 else
    32 if GPU_MEMORY_GB > 0.0 else
    min(16, max(1, os.cpu_count() or 1))
)

STEPS_PER_UPDATE = int(os.getenv("PPO_STEPS_PER_UPDATE", "4096"))
REQUESTED_MINIBATCH = int(os.getenv("PPO_MINIBATCH_SIZE", str(DEFAULT_MINIBATCH)))
MINIBATCH_SIZE = int(os.getenv(
    "PPO_GSA_MINIBATCH_SIZE",
    str(min(REQUESTED_MINIBATCH, 512 if GPU_MEMORY_GB < 20.0 else 1_024)),
))
UPDATE_EPOCHS = int(os.getenv("PPO_GSA_UPDATE_EPOCHS", os.getenv("PPO_UPDATE_EPOCHS", "4")))
NUM_ENVS = int(os.getenv("PPO_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
CPU_ENV_WORKERS = int(os.getenv(
    "PPO_CPU_WORKERS",
    str(min(32, DEFAULT_NUM_ENVS, max(1, (os.cpu_count() or 1) - 1))),
))

LEARNING_RATE = float(os.getenv("PPO_GSA_LR", "2e-4"))
MIN_LEARNING_RATE = float(os.getenv("PPO_GSA_MIN_LR", "2e-5"))
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5

GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.20
VALUE_CLIP = 0.20
VALUE_COEF = 0.50
MAX_GRAD_NORM = 0.50
TARGET_KL = float(os.getenv("PPO_GSA_TARGET_KL", "0.015"))

# Entropy của từng head được chuẩn hóa theo log(số action hợp lệ), sau đó
# lấy trung bình 6 head. Vì vậy hệ số này ổn định khi topology thay đổi.
ENTROPY_COEF_START = float(os.getenv("PPO_GSA_ENTROPY_START", "0.01"))
ENTROPY_COEF_END = float(os.getenv("PPO_GSA_ENTROPY_END", "0.001"))
ENTROPY_ANNEAL_FRACTION = float(os.getenv("PPO_GSA_ENTROPY_FRACTION", "0.45"))

HIDDEN_DIM = int(os.getenv("PPO_GSA_HIDDEN_DIM", "192"))
NUM_LAYERS = int(os.getenv("PPO_GSA_NUM_LAYERS", "2"))
UE_SIZE_SAMPLING = os.getenv("PPO_UE_SIZE_SAMPLING", "transition_balanced")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
PIN_MEMORY = True
DEBUG_MASKS = False

# Hệ số chuẩn hóa feature.
RATE_LOG_SCALE = 22.0
SINR_SCALE = 30.0
LATENCY_US_LOG_SCALE = 14.0
PACKET_LOG_SCALE = 12.0
LAMBDA_LOG_SCALE = 10.0
RU_RESOURCE_LOG_SCALE = 4.0
CPU_RESOURCE_LOG_SCALE = 24.0
UE_COUNT_LOG_SCALE = math.log1p(100.0)
# ======================================================================
# 2. HÀM HỖ TRỢ
# ======================================================================

def init_linear(module, gain=math.sqrt(2.0)):
    """Orthogonal initialization cho Linear."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def bucketed_minibatches(states, minibatch_size):
    """Sort theo số UE rồi chia batch đầy để giảm padding mà không tăng update."""
    if minibatch_size <= 0:
        raise ValueError("minibatch_size phải > 0.")
    indices = np.arange(len(states), dtype=np.int64)
    np.random.shuffle(indices)  # random thứ tự giữa các state có cùng kích thước
    sizes = np.asarray([int(states[i].sizes[0]) for i in indices])
    indices = indices[np.argsort(sizes, kind="stable")]
    batches = [
        indices[start:start + minibatch_size]
        for start in range(0, indices.size, minibatch_size)
    ]
    np.random.shuffle(batches)
    return batches


def masked_pool(embeddings, valid_mask, weights=None):
    """Weighted mean + max pooling, an toàn khi mask rỗng."""
    valid_mask = valid_mask.bool()
    if weights is None:
        weights = valid_mask.to(embeddings.dtype)
    else:
        weights = weights.to(embeddings.dtype) * valid_mask.to(embeddings.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean = (embeddings * weights.unsqueeze(-1)).sum(dim=1) / denominator
    maximum = embeddings.masked_fill(
        ~valid_mask.unsqueeze(-1), -torch.inf
    ).amax(dim=1)
    maximum = torch.where(
        torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
    )
    return torch.cat([mean, maximum], dim=-1)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Action domain và hằng số vật lý được đọc hoàn toàn từ state."""

    power_levels: tuple[float, ...]
    max_rbs_per_ue: int
    bandwidth_per_rb: float
    k_du: float
    k_cu: float
    speed_of_light: float

    @classmethod
    def from_state(cls, state):
        cfg = state["env_config"]
        ran = state["state_RAN"]
        power = np.asarray(cfg["P_ib_k_val"], dtype=np.float64)
        if power.ndim != 1 or power.size == 0 or not np.isfinite(power).all():
            raise ValueError("Power codebook phải là vector 1D hữu hạn, không rỗng.")
        if np.any(power < 0.0):
            raise ValueError("Power codebook không được chứa mức âm.")
        if np.any(np.diff(power) <= 0.0):
            raise ValueError("Power codebook phải tăng nghiêm ngặt và không trùng.")
        # RLEnv.check_action() cấm power <= 0. Mức 0 trong config không phải action.
        power = power[power > 0.0]
        if power.size == 0:
            raise ValueError("Power codebook phải có ít nhất một mức dương.")

        num_rbs = int(ran["num_RBs"])
        max_rbs = int(cfg["max_RBs_per_UE"])
        bandwidth = float(cfg["W_b"])
        k_du = float(cfg["k_DU"])
        k_cu = float(cfg["k_CU"])
        speed_of_light = float(speed_of_light_mps)

        if num_rbs <= 0 or not 0 < max_rbs <= num_rbs:
            raise ValueError("Cần 0 < max_RBs_per_UE <= num_RBs.")
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("Bandwidth mỗi RB phải dương và hữu hạn.")
        if not all(math.isfinite(x) and x >= 0.0 for x in (k_du, k_cu)):
            raise ValueError("k_DU và k_CU phải không âm và hữu hạn.")
        if not math.isfinite(speed_of_light) or speed_of_light <= 0.0:
            raise ValueError("speed_of_light_mps phải dương và hữu hạn.")

        return cls(
            power_levels=tuple(float(x) for x in power),
            max_rbs_per_ue=max_rbs,
            bandwidth_per_rb=bandwidth,
            k_du=k_du,
            k_cu=k_cu,
            speed_of_light=speed_of_light,
        )


def reset_fixed_env(env, seed=TRAIN_SEED):
    """Reset cùng network snapshot nhưng không làm thay đổi RNG dùng để sample policy."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    state = env.reset(seed=seed)

    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return state


# ======================================================================
# 3. STATE
# ======================================================================

@dataclass
class TensorState:
    """Một state đã tensor hóa và lưu trên CPU."""

    node_features: torch.Tensor           # [N, NODE_DIM]
    adjacency_matrix: torch.Tensor        # [N, N]
    ue_features: torch.Tensor             # [K, UE_DIM]
    ue_ru_features: torch.Tensor          # [K, R, UE_RU_DIM]
    ue_ru_gain: torch.Tensor              # [K, R], gain đã chuẩn hóa theo noise/RB
    ue_ru_distance: torch.Tensor          # [K, R], mét
    ue_r_min: torch.Tensor                # [K], throughput QoS [bit/s]
    ue_l_max: torch.Tensor                # [K], latency QoS [s]
    ue_packet_bits: torch.Tensor          # [K]
    ue_lambda_pps: torch.Tensor           # [K]
    ue_eta: torch.Tensor                  # [K]
    ue_ids: torch.Tensor                  # [K]
    active_ue_mask: torch.Tensor          # [K]
    ru_remaining: torch.Tensor            # [R]
    du_remaining: torch.Tensor            # [D]
    cu_remaining: torch.Tensor            # [C]
    ru_du: torch.Tensor                   # [R, D]
    du_cu: torch.Tensor                   # [D, C]
    rb_remaining: int
    rb_ratio: float
    max_rbs_allowed: int
    bandwidth_per_rb: float
    k_du: float
    k_cu: float
    speed_of_light: float

    @property
    def sizes(self):
        return (
            self.ue_ids.numel(),
            self.ru_remaining.numel(),
            self.du_remaining.numel(),
            self.cu_remaining.numel(),
        )


@dataclass
class StateBatch:
    """Batch các state sau padding."""

    node_features: torch.Tensor
    adjacency_matrix: torch.Tensor
    valid_node_mask: torch.Tensor
    ue_features: torch.Tensor
    ue_ru_features: torch.Tensor
    ue_ru_gain: torch.Tensor
    ue_ru_distance: torch.Tensor
    ue_r_min: torch.Tensor
    ue_l_max: torch.Tensor
    ue_packet_bits: torch.Tensor
    ue_lambda_pps: torch.Tensor
    ue_eta: torch.Tensor
    ue_ids: torch.Tensor
    valid_ue_mask: torch.Tensor
    active_ue_mask: torch.Tensor
    ru_indices: torch.Tensor
    du_indices: torch.Tensor
    cu_indices: torch.Tensor
    valid_ru_mask: torch.Tensor
    valid_du_mask: torch.Tensor
    valid_cu_mask: torch.Tensor
    ru_remaining: torch.Tensor
    du_remaining: torch.Tensor
    cu_remaining: torch.Tensor
    ru_du: torch.Tensor
    du_cu: torch.Tensor
    rb_remaining: torch.Tensor
    rb_ratio: torch.Tensor
    rb_domain_mask: torch.Tensor
    rb_mask: torch.Tensor
    bandwidth_per_rb: torch.Tensor
    k_du: torch.Tensor
    k_cu: torch.Tensor
    speed_of_light: torch.Tensor

    def to(self, device, non_blocking=True):
        return StateBatch(**{
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in vars(self).items()
        })


class StateBuilder:
    """Chuyển public state của RLEnv thành tensor cho PPO."""

    # resource, capacity, node type, normalized degree và log-degree.
    NODE_DIM = 8

    # [R_min, SINR_min, L_max, packet_size, lambda, eta, w_acc, w_thr, w_lat]
    UE_DIM = 9

    # [absolute_gain, relative_gain, log_distance, distance]
    UE_RU_DIM = 4

    def build(self, state: dict) -> TensorState:
        ran = state["state_RAN"]
        ues = state["state_UEs_req"]
        spec = EnvironmentSpec.from_state(state)

        num_rus = int(ran["num_RUs"])
        num_dus = int(ran["num_DUs"])
        num_cus = int(ran["num_CUs"])
        if min(num_rus, num_dus, num_cus) <= 0:
            raise ValueError("num_RUs, num_DUs và num_CUs phải > 0.")

        ru_rem = self._read_vector(ran, "P_i_rem", num_rus)
        du_rem = self._read_vector(ran, "A_j_rem", num_dus)
        cu_rem = self._read_vector(ran, "A_m_rem", num_cus)
        max_ru = self._read_positive(ran, "max_P_i")
        max_du = self._read_positive(ran, "max_A_j")
        max_cu = self._read_positive(ran, "max_A_m")
        ru_cap = np.full(num_rus, max_ru, dtype=np.float32)
        du_cap = np.full(num_dus, max_du, dtype=np.float32)
        cu_cap = np.full(num_cus, max_cu, dtype=np.float32)
        ru_du = self._read_binary(ran, "l_ru_du", (num_rus, num_dus))
        du_cu = self._read_binary(ran, "l_du_cu", (num_dus, num_cus))

        # ------------------------ RAN node features ------------------------
        num_nodes = num_rus + num_dus + num_cus
        du_start = num_rus
        cu_start = num_rus + num_dus
        node_features = np.zeros((num_nodes, self.NODE_DIM), dtype=np.float32)

        node_features[:num_rus, 0] = np.clip(ru_rem / ru_cap, 0.0, 1.0)
        node_features[:num_rus, 1] = np.log1p(ru_rem) / RU_RESOURCE_LOG_SCALE
        node_features[:num_rus, 2] = np.log1p(ru_cap) / RU_RESOURCE_LOG_SCALE
        node_features[:num_rus, 3] = 1.0

        node_features[du_start:cu_start, 0] = np.clip(du_rem / du_cap, 0.0, 1.0)
        node_features[du_start:cu_start, 1] = np.log1p(du_rem) / CPU_RESOURCE_LOG_SCALE
        node_features[du_start:cu_start, 2] = np.log1p(du_cap) / CPU_RESOURCE_LOG_SCALE
        node_features[du_start:cu_start, 4] = 1.0

        node_features[cu_start:, 0] = np.clip(cu_rem / cu_cap, 0.0, 1.0)
        node_features[cu_start:, 1] = np.log1p(cu_rem) / CPU_RESOURCE_LOG_SCALE
        node_features[cu_start:, 2] = np.log1p(cu_cap) / CPU_RESOURCE_LOG_SCALE
        node_features[cu_start:, 5] = 1.0

        degrees = np.concatenate([
            ru_du.sum(axis=1),
            ru_du.sum(axis=0) + du_cu.sum(axis=1),
            du_cu.sum(axis=0),
        ]).astype(np.float32)
        possible = np.concatenate([
            np.full(num_rus, max(num_dus, 1)),
            np.full(num_dus, max(num_rus + num_cus, 1)),
            np.full(num_cus, max(num_dus, 1)),
        ]).astype(np.float32)
        node_features[:, 6] = degrees / possible
        node_features[:, 7] = np.log1p(degrees) / max(math.log1p(num_nodes), 1.0)

        # ----------------------------- UE state -----------------------------
        ue_items = sorted(ues.items(), key=lambda item: int(item[0]))
        ue_ids = np.asarray([int(ue_id) for ue_id, _ in ue_items], dtype=np.int64)
        active = np.asarray([bool(ue["status"].get("active", 0)) for _, ue in ue_items], dtype=bool)

        if ue_items:
            ue_features = np.asarray([self._ue_features(ue) for _, ue in ue_items], dtype=np.float32)
            ue_r_min = np.asarray([float(ue["R_min_k"]) for _, ue in ue_items], dtype=np.float32)
            ue_l_max = np.asarray([float(ue["L_max_k"]) for _, ue in ue_items], dtype=np.float32)
            ue_packet_bits = np.asarray([float(ue["packet_size_bits"]) for _, ue in ue_items], dtype=np.float32)
            ue_lambda_pps = np.asarray([float(ue["lambda_pps"]) for _, ue in ue_items], dtype=np.float32)
            ue_eta = np.asarray([float(ue["eta_k"]) for _, ue in ue_items], dtype=np.float32)
        else:
            ue_features = np.empty((0, self.UE_DIM), dtype=np.float32)
            ue_r_min = np.empty(0, dtype=np.float32)
            ue_l_max = np.empty(0, dtype=np.float32)
            ue_packet_bits = np.empty(0, dtype=np.float32)
            ue_lambda_pps = np.empty(0, dtype=np.float32)
            ue_eta = np.empty(0, dtype=np.float32)

        gains, distances, relations = [], [], []
        for _, ue in ue_items:
            gain, distance = self._read_ue_ru_arrays(ue, num_rus)
            gains.append(gain)
            distances.append(distance)
            relations.append(self._ue_ru_features(gain, distance))

        ue_ru_gain = np.asarray(gains, dtype=np.float32) if gains else np.empty((0, num_rus), dtype=np.float32)
        ue_ru_distance = np.asarray(distances, dtype=np.float32) if distances else np.empty((0, num_rus), dtype=np.float32)
        ue_ru_features = np.asarray(relations, dtype=np.float32) if relations else np.empty((0, num_rus, self.UE_RU_DIM), dtype=np.float32)

        # ------------------------------ RB state -----------------------------
        num_rbs = int(ran["num_RBs"])
        rb_remaining = int(ran["RB_rem"])
        max_rbs_allowed = spec.max_rbs_per_ue
        if num_rbs <= 0 or not 0 <= rb_remaining <= num_rbs:
            raise ValueError("num_RBs hoặc RB_rem không hợp lệ.")
        if max_rbs_allowed <= 0:
            raise ValueError("max_RBs_per_UE phải > 0.")

        return TensorState(
            node_features=torch.from_numpy(node_features),
            adjacency_matrix=torch.from_numpy(self._build_adjacency(ru_du, du_cu, num_rus, num_dus, num_cus)),
            ue_features=torch.from_numpy(ue_features),
            ue_ru_features=torch.from_numpy(ue_ru_features),
            ue_ru_gain=torch.from_numpy(ue_ru_gain),
            ue_ru_distance=torch.from_numpy(ue_ru_distance),
            ue_r_min=torch.from_numpy(ue_r_min),
            ue_l_max=torch.from_numpy(ue_l_max),
            ue_packet_bits=torch.from_numpy(ue_packet_bits),
            ue_lambda_pps=torch.from_numpy(ue_lambda_pps),
            ue_eta=torch.from_numpy(ue_eta),
            ue_ids=torch.from_numpy(ue_ids),
            active_ue_mask=torch.from_numpy(active),
            ru_remaining=torch.from_numpy(ru_rem.copy()),
            du_remaining=torch.from_numpy(du_rem.copy()),
            cu_remaining=torch.from_numpy(cu_rem.copy()),
            ru_du=torch.from_numpy(ru_du.copy()),
            du_cu=torch.from_numpy(du_cu.copy()),
            rb_remaining=rb_remaining,
            rb_ratio=rb_remaining / num_rbs,
            max_rbs_allowed=max_rbs_allowed,
            bandwidth_per_rb=spec.bandwidth_per_rb,
            k_du=spec.k_du,
            k_cu=spec.k_cu,
            speed_of_light=spec.speed_of_light,
        )

    def collate(self, states: Sequence[TensorState], max_rbs: int, pin_memory=False) -> StateBatch:
        """Padding nhiều state khác kích thước thành một batch."""
        if not states:
            raise ValueError("states không được rỗng.")

        batch_size = len(states)
        max_ues = max(state.sizes[0] for state in states)
        max_rus = max(state.sizes[1] for state in states)
        max_dus = max(state.sizes[2] for state in states)
        max_cus = max(state.sizes[3] for state in states)
        max_nodes = max(sum(state.sizes[1:]) for state in states)

        node_features = torch.zeros(batch_size, max_nodes, self.NODE_DIM)
        adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
        valid_node = torch.zeros(batch_size, max_nodes, dtype=torch.bool)

        ue_features = torch.zeros(batch_size, max_ues, self.UE_DIM)
        ue_ru_features = torch.zeros(batch_size, max_ues, max_rus, self.UE_RU_DIM)
        ue_ru_gain = torch.zeros(batch_size, max_ues, max_rus)
        ue_ru_distance = torch.zeros(batch_size, max_ues, max_rus)
        ue_r_min = torch.zeros(batch_size, max_ues)
        ue_l_max = torch.zeros(batch_size, max_ues)
        ue_packet_bits = torch.zeros(batch_size, max_ues)
        ue_lambda_pps = torch.zeros(batch_size, max_ues)
        ue_eta = torch.zeros(batch_size, max_ues)
        ue_ids = torch.full((batch_size, max_ues), -1, dtype=torch.long)
        valid_ue = torch.zeros(batch_size, max_ues, dtype=torch.bool)
        active_ue = torch.zeros(batch_size, max_ues, dtype=torch.bool)

        ru_indices = torch.zeros(batch_size, max_rus, dtype=torch.long)
        du_indices = torch.zeros(batch_size, max_dus, dtype=torch.long)
        cu_indices = torch.zeros(batch_size, max_cus, dtype=torch.long)
        valid_ru = torch.zeros(batch_size, max_rus, dtype=torch.bool)
        valid_du = torch.zeros(batch_size, max_dus, dtype=torch.bool)
        valid_cu = torch.zeros(batch_size, max_cus, dtype=torch.bool)

        ru_remaining = torch.zeros(batch_size, max_rus)
        du_remaining = torch.zeros(batch_size, max_dus)
        cu_remaining = torch.zeros(batch_size, max_cus)
        ru_du = torch.zeros(batch_size, max_rus, max_dus, dtype=torch.bool)
        du_cu = torch.zeros(batch_size, max_dus, max_cus, dtype=torch.bool)

        rb_remaining = torch.zeros(batch_size, dtype=torch.long)
        rb_ratio = torch.zeros(batch_size)
        rb_domain_mask = torch.zeros(batch_size, max_rbs, dtype=torch.bool)
        rb_mask = torch.zeros(batch_size, max_rbs, dtype=torch.bool)
        bandwidth_per_rb = torch.zeros(batch_size)
        k_du = torch.zeros(batch_size)
        k_cu = torch.zeros(batch_size)
        speed_of_light = torch.zeros(batch_size)

        for b, state in enumerate(states):
            num_ues, num_rus, num_dus, num_cus = state.sizes
            num_nodes = num_rus + num_dus + num_cus
            du_start = num_rus
            cu_start = num_rus + num_dus

            node_features[b, :num_nodes] = state.node_features
            adjacency[b, :num_nodes, :num_nodes] = state.adjacency_matrix
            valid_node[b, :num_nodes] = True

            ue_features[b, :num_ues] = state.ue_features
            ue_ru_features[b, :num_ues, :num_rus] = state.ue_ru_features
            ue_ru_gain[b, :num_ues, :num_rus] = state.ue_ru_gain
            ue_ru_distance[b, :num_ues, :num_rus] = state.ue_ru_distance
            ue_r_min[b, :num_ues] = state.ue_r_min
            ue_l_max[b, :num_ues] = state.ue_l_max
            ue_packet_bits[b, :num_ues] = state.ue_packet_bits
            ue_lambda_pps[b, :num_ues] = state.ue_lambda_pps
            ue_eta[b, :num_ues] = state.ue_eta
            ue_ids[b, :num_ues] = state.ue_ids
            valid_ue[b, :num_ues] = True
            active_ue[b, :num_ues] = state.active_ue_mask

            ru_indices[b, :num_rus] = torch.arange(num_rus)
            du_indices[b, :num_dus] = torch.arange(du_start, cu_start)
            cu_indices[b, :num_cus] = torch.arange(cu_start, num_nodes)
            valid_ru[b, :num_rus] = True
            valid_du[b, :num_dus] = True
            valid_cu[b, :num_cus] = True

            ru_remaining[b, :num_rus] = state.ru_remaining
            du_remaining[b, :num_dus] = state.du_remaining
            cu_remaining[b, :num_cus] = state.cu_remaining
            ru_du[b, :num_rus, :num_dus] = state.ru_du
            du_cu[b, :num_dus, :num_cus] = state.du_cu

            rb_remaining[b] = state.rb_remaining
            rb_ratio[b] = state.rb_ratio
            rb_domain = min(max_rbs, state.max_rbs_allowed)
            rb_available = min(rb_domain, state.rb_remaining)
            rb_domain_mask[b, :rb_domain] = True
            rb_mask[b, :rb_available] = True
            bandwidth_per_rb[b] = state.bandwidth_per_rb
            k_du[b] = state.k_du
            k_cu[b] = state.k_cu
            speed_of_light[b] = state.speed_of_light

        batch = StateBatch(
            node_features=node_features,
            adjacency_matrix=adjacency,
            valid_node_mask=valid_node,
            ue_features=ue_features,
            ue_ru_features=ue_ru_features,
            ue_ru_gain=ue_ru_gain,
            ue_ru_distance=ue_ru_distance,
            ue_r_min=ue_r_min,
            ue_l_max=ue_l_max,
            ue_packet_bits=ue_packet_bits,
            ue_lambda_pps=ue_lambda_pps,
            ue_eta=ue_eta,
            ue_ids=ue_ids,
            valid_ue_mask=valid_ue,
            active_ue_mask=active_ue,
            ru_indices=ru_indices,
            du_indices=du_indices,
            cu_indices=cu_indices,
            valid_ru_mask=valid_ru,
            valid_du_mask=valid_du,
            valid_cu_mask=valid_cu,
            ru_remaining=ru_remaining,
            du_remaining=du_remaining,
            cu_remaining=cu_remaining,
            ru_du=ru_du,
            du_cu=du_cu,
            rb_remaining=rb_remaining,
            rb_ratio=rb_ratio,
            rb_domain_mask=rb_domain_mask,
            rb_mask=rb_mask,
            bandwidth_per_rb=bandwidth_per_rb,
            k_du=k_du,
            k_cu=k_cu,
            speed_of_light=speed_of_light,
        )
        if pin_memory and torch.cuda.is_available():
            return StateBatch(**{name: tensor.pin_memory() for name, tensor in vars(batch).items()})
        return batch

    @staticmethod
    def _build_adjacency(ru_du, du_cu, num_rus, num_dus, num_cus):
        num_nodes = num_rus + num_dus + num_cus
        du_start = num_rus
        cu_start = num_rus + num_dus
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adjacency[:num_rus, du_start:cu_start] = ru_du
        adjacency[du_start:cu_start, :num_rus] = ru_du.T
        adjacency[du_start:cu_start, cu_start:] = du_cu
        adjacency[cu_start:, du_start:cu_start] = du_cu.T
        return adjacency

    @staticmethod
    def _ue_features(ue):
        return [
            math.log1p(max(float(ue["R_min_k"]), 0.0)) / RATE_LOG_SCALE,
            float(ue["SINR_min_k"]) / SINR_SCALE,
            math.log1p(max(float(ue["L_max_k"]), 0.0) * 1e6) / LATENCY_US_LOG_SCALE,
            math.log1p(max(float(ue["packet_size_bits"]), 0.0)) / PACKET_LOG_SCALE,
            math.log1p(max(float(ue["lambda_pps"]), 0.0)) / LAMBDA_LOG_SCALE,
            float(ue["eta_k"]),
            float(ue["w_acc_s"]),
            float(ue["w_thr_s"]),
            float(ue["w_lat_s"]),
        ]

    @staticmethod
    def _read_ue_ru_arrays(ue, num_rus):
        gain = np.asarray(ue["g_i_k"], dtype=np.float32)
        distance = np.asarray(ue["d_i_k"], dtype=np.float32)
        if gain.shape != (num_rus,) or distance.shape != (num_rus,):
            raise ValueError(f"UE {ue.get('id')}: g_i_k hoặc d_i_k sai shape.")
        if not np.all(np.isfinite(gain)) or np.any(gain < 0.0):
            raise ValueError(f"UE {ue.get('id')}: g_i_k không hợp lệ.")
        if not np.all(np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError(f"UE {ue.get('id')}: d_i_k không hợp lệ.")
        return gain, distance

    @staticmethod
    def _ue_ru_features(gain, distance):
        log_gain = np.log10(np.maximum(gain, 1e-30))
        log_distance = np.log1p(distance)
        return np.column_stack([
            np.tanh(log_gain / 10.0),
            np.tanh((log_gain - log_gain.max()) / 5.0),
            log_distance / max(float(log_distance.max()), 1.0),
            distance / max(float(distance.max()), 1.0),
        ]).astype(np.float32)

    @staticmethod
    def _read_vector(data, key, size):
        value = np.asarray(data[key], dtype=np.float32)
        if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError(f"{key} phải có shape ({size},) và chứa giá trị không âm hữu hạn.")
        return value

    @staticmethod
    def _read_positive(data, key):
        value = float(data[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} phải là số dương hữu hạn.")
        return value

    @staticmethod
    def _read_binary(data, key, shape):
        value = np.asarray(data[key])
        if value.shape != shape or not np.all(np.isin(value, [0, 1])):
            raise ValueError(f"{key} phải có shape {shape} và chỉ chứa 0/1.")
        return value.astype(bool)


# ======================================================================
# 4. RAN ENCODER
# ======================================================================

class GraphSAGELayer(nn.Module):
    """GraphSAGE mean/max aggregation với gated residual."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.update = nn.Linear(3 * hidden_dim, hidden_dim)
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_embeddings, adjacency, valid_node_mask, node_weights):
        # Encoder tự loại padding edge và self-loop.
        valid_pair = valid_node_mask.unsqueeze(1) & valid_node_mask.unsqueeze(2)
        adjacency = adjacency.to(node_embeddings.dtype) * valid_pair.to(node_embeddings.dtype)
        eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=torch.bool).unsqueeze(0)
        adjacency = adjacency.masked_fill(eye, 0.0)

        weighted_adjacency = adjacency * node_weights.unsqueeze(1)
        degree = weighted_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        neighbor_mean = torch.bmm(weighted_adjacency, node_embeddings) / degree
        neighbors = node_embeddings.unsqueeze(1).expand(
            -1, adjacency.size(1), -1, -1
        )
        neighbor_max = neighbors.masked_fill(
            adjacency.unsqueeze(-1) <= 0.0, -torch.inf
        ).amax(dim=2)
        neighbor_max = torch.where(
            torch.isfinite(neighbor_max), neighbor_max, torch.zeros_like(neighbor_max)
        )
        context = torch.cat([node_embeddings, neighbor_mean, neighbor_max], dim=-1)
        update = F.silu(self.update(context))
        node_embeddings = self.norm1(node_embeddings + torch.sigmoid(self.gate(context)) * update)
        node_embeddings = self.norm2(node_embeddings + self.ffn(node_embeddings))
        return node_embeddings * valid_node_mask.unsqueeze(-1)


class RANEncoder(nn.Module):
    """GraphSAGE hai nhánh: local MLP + message passing có cổng.

    Nhánh local giúp encoder ít nhất biểu diễn được baseline MLP; nhánh graph
    chỉ bổ sung thông tin topology khi hữu ích, hạn chế oversmoothing trên
    graph RU-DU-CU dày đặc.
    """

    def __init__(self, node_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden_dim)
        weight_hidden = max(hidden_dim // 2, 32)
        self.weight_encoder = nn.Sequential(
            nn.Linear(node_dim, weight_hidden),
            nn.SiLU(),
            nn.Linear(weight_hidden, 1),
        )
        self.local_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.local_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.graph_layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim) for _ in range(num_layers)
        ])
        # Sau UE->RU message passing, layer này truyền nhu cầu mới từ
        # RU sang DU/CU. Không có bước này, embedding DU/CU không nhìn
        # thấy cạnh tranh UE hiện tại.
        self.demand_fusion = nn.ModuleList([
            GraphSAGELayer(hidden_dim),  # UE -> RU -> DU
            GraphSAGELayer(hidden_dim),  # DU -> CU
        ])
        self.graph_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pool = nn.Linear(6 * hidden_dim, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.apply(init_linear)

    def forward(self, node_features, adjacency, valid_node_mask):
        valid = valid_node_mask.unsqueeze(-1)
        base = F.silu(self.input(node_features)) * valid
        node_weights = self.compute_node_weights(node_features, valid_node_mask)

        local = base
        for layer, norm in zip(self.local_layers, self.local_norms):
            local = norm(local + F.silu(layer(local))) * valid

        graph = base
        for layer in self.graph_layers:
            graph = layer(graph, adjacency, valid_node_mask, node_weights)

        gate = torch.sigmoid(self.graph_gate(torch.cat([local, graph], dim=-1)))
        node_embeddings = (local + gate * (graph - local)) * valid

        graph_embedding = self.pool_nodes(
            node_embeddings, node_features, valid_node_mask, node_weights
        )
        return node_embeddings, graph_embedding

    def propagate_demand(
        self, node_embeddings, node_features, adjacency, valid_node_mask,
    ):
        node_weights = self.compute_node_weights(node_features, valid_node_mask)
        for layer in self.demand_fusion:
            node_embeddings = layer(
                node_embeddings, adjacency, valid_node_mask, node_weights
            )
        return node_embeddings

    def compute_node_weights(self, node_features, valid_node_mask):
        """Học importance riêng cho từng node từ resource/capacity/type/degree."""
        weights = F.softplus(self.weight_encoder(node_features).squeeze(-1)) + 1e-4
        weights = weights * valid_node_mask.to(weights.dtype)
        count = valid_node_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = weights.sum(dim=-1, keepdim=True) / count.to(weights.dtype)
        return weights / mean.clamp_min(1e-6)

    def pool_nodes(
        self, node_embeddings, node_features, valid_node_mask, node_weights=None,
    ):
        """Pool theo loại node; dùng lại sau message passing UE-RU."""
        if node_weights is None:
            node_weights = self.compute_node_weights(node_features, valid_node_mask)
        ru_mask = valid_node_mask & (node_features[..., 3] > 0.5)
        du_mask = valid_node_mask & (node_features[..., 4] > 0.5)
        cu_mask = valid_node_mask & (node_features[..., 5] > 0.5)
        pooled = torch.cat([
            masked_pool(node_embeddings, ru_mask, node_weights),
            masked_pool(node_embeddings, du_mask, node_weights),
            masked_pool(node_embeddings, cu_mask, node_weights),
        ], dim=-1)
        return self.pool_norm(F.silu(self.pool(pooled)))


# ======================================================================
# 5. UE + UE-RU ENCODER
# ======================================================================

class UEEncoder(nn.Module):
    """Shared MLP cho UE với residual + LayerNorm."""

    def __init__(self, ue_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.input = nn.Linear(ue_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.pool = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.apply(init_linear)

    def forward(self, features, valid_mask, active_mask):
        embeddings = F.silu(self.input(features)) * valid_mask.unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            embeddings = norm(embeddings + F.silu(layer(embeddings)))
            embeddings = embeddings * valid_mask.unsqueeze(-1)

        set_embedding = self.pool_embeddings(embeddings, valid_mask, active_mask)
        return embeddings, set_embedding

    def pool_embeddings(self, embeddings, valid_mask, active_mask):
        active_mask = valid_mask & active_mask
        set_embedding = self.pool_norm(F.silu(self.pool(masked_pool(embeddings, active_mask))))
        set_embedding = set_embedding * active_mask.any(dim=-1, keepdim=True).to(set_embedding.dtype)
        return set_embedding


class UERURelationEncoder(nn.Module):
    """Shared MLP cho từng cặp UE-RU với residual + LayerNorm."""

    def __init__(self, relation_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.input = nn.Linear(relation_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.apply(init_linear)

    def forward(self, features, valid_ue_mask, valid_ru_mask):
        pair_mask = valid_ue_mask.unsqueeze(-1) & valid_ru_mask.unsqueeze(1)
        embeddings = F.silu(self.input(features)) * pair_mask.unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            embeddings = norm(embeddings + F.silu(layer(embeddings)))
            embeddings = embeddings * pair_mask.unsqueeze(-1)
        return embeddings


class UERUGraphSAGE(nn.Module):
    """Message passing hai chiều trên graph UE-RU có edge feature channel.

    Đây là phần graph mà MLP baseline không có. Nó cho phép mỗi RU tổng hợp
    nhu cầu/cạnh tranh của mọi UE active và mỗi UE nhận lại trạng thái của các
    RU mà nó có thể kết nối, thay vì chỉ encode từng cặp UE-RU độc lập.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # Phép chiếu cộng tránh tạo tensor concat [..., 3H], tiết kiệm VRAM.
        self.ue_messages = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ru_messages = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.edge_messages = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ue_updates = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ru_updates = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ue_gates = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ru_gates = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ue_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ru_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    @staticmethod
    def _masked_mean(values, mask, dim):
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)

    @staticmethod
    def _masked_max(values, mask, dim):
        maximum = values.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=dim)
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    def forward(self, ue, ru, relation, active_ue_mask, valid_ru_mask):
        valid_pair = active_ue_mask.unsqueeze(-1) & valid_ru_mask.unsqueeze(1)
        ue_valid = active_ue_mask.unsqueeze(-1)
        ru_valid = valid_ru_mask.unsqueeze(-1)

        for ue_message, ru_message, edge_message, ue_update, ru_update, ue_gate, ru_gate, ue_norm, ru_norm in zip(
            self.ue_messages,
            self.ru_messages,
            self.edge_messages,
            self.ue_updates,
            self.ru_updates,
            self.ue_gates,
            self.ru_gates,
            self.ue_norms,
            self.ru_norms,
        ):
            messages = F.silu(
                ue_message(ue).unsqueeze(2)
                + ru_message(ru).unsqueeze(1)
                + edge_message(relation)
            )
            messages = messages * valid_pair.unsqueeze(-1)

            transposed_messages = messages.transpose(1, 2)
            transposed_mask = valid_pair.transpose(1, 2)
            ue_mean = self._masked_mean(messages, valid_pair, dim=2)
            ue_max = self._masked_max(messages, valid_pair, dim=2)
            ru_mean = self._masked_mean(transposed_messages, transposed_mask, dim=2)
            ru_max = self._masked_max(transposed_messages, transposed_mask, dim=2)

            ue_context = torch.cat([ue, ue_mean, ue_max], dim=-1)
            ru_context = torch.cat([ru, ru_mean, ru_max], dim=-1)
            next_ue = ue_norm(
                ue + torch.sigmoid(ue_gate(ue_context)) * F.silu(ue_update(ue_context))
            )
            next_ru = ru_norm(
                ru + torch.sigmoid(ru_gate(ru_context)) * F.silu(ru_update(ru_context))
            )
            ue = torch.where(ue_valid, next_ue, ue)
            ru = torch.where(ru_valid, next_ru, ru)

        return ue, ru


# ======================================================================
# 6. ACTOR-CRITIC
# ======================================================================

class ActorCritic(nn.Module):
    """
    Thứ tự sinh action:
        UE -> RU -> RB -> Power -> DU -> CU

    Schema lưu trong rollout vẫn là:
        [UE, RU, DU, CU, RB, Power]

    Action gửi RLEnv:
        (UE_id, RU, DU, CU, Power, RB)
    """

    UE, RU, DU, CU, RB, POWER = range(6)
    NUM_ACTIONS = 6

    def __init__(self, node_dim, ue_dim, relation_dim, power_levels, max_rbs,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_rbs = int(max_rbs)

        power_levels = torch.as_tensor(power_levels, dtype=torch.float32)
        if power_levels.ndim != 1 or power_levels.numel() == 0:
            raise ValueError("power_levels phải là vector 1D không rỗng.")
        if not torch.isfinite(power_levels).all() or torch.any(power_levels <= 0.0):
            raise ValueError("power_levels phải chứa các giá trị dương hữu hạn.")
        if torch.unique(power_levels).numel() != power_levels.numel() or torch.any(power_levels[1:] <= power_levels[:-1]):
            raise ValueError("power_levels phải không trùng và tăng dần.")
        if self.max_rbs <= 0:
            raise ValueError("max_rbs phải > 0.")
        self.register_buffer("power_levels", power_levels)

        h = self.hidden_dim
        self.ran_encoder = RANEncoder(node_dim, h, num_layers)
        self.ue_encoder = UEEncoder(ue_dim, h, num_layers)
        self.relation_encoder = UERURelationEncoder(relation_dim, h, num_layers)
        self.ue_ru_graph = UERUGraphSAGE(h, num_layers)

        self.ue_relation_pool = nn.Linear(2 * h, h)
        self.ue_relation_norm = nn.LayerNorm(h)
        self.relation_set_pool = nn.Linear(2 * h, h)
        self.relation_set_norm = nn.LayerNorm(h)

        # RAN + UE set + relation set + RB ratio + active ratio
        # + log(total UE) + log(active UE).
        self.global_dim = 3 * h + 4

        self.ue_head = nn.Sequential(
            nn.Linear(2 * h + self.global_dim, h), nn.SiLU(), nn.Linear(h, 1)
        )
        self.ru_head = nn.Sequential(
            nn.Linear(3 * h + self.global_dim, h), nn.SiLU(), nn.Linear(h, 1)
        )

        allocation_dim = 3 * h + self.global_dim
        self.rb_head = nn.Sequential(
            nn.Linear(allocation_dim, h), nn.SiLU(), nn.Linear(h, self.max_rbs)
        )
        self.power_head = nn.Sequential(
            nn.Linear(allocation_dim + 1, h), nn.SiLU(), nn.Linear(h, power_levels.numel())
        )

        # decision_features = RB + Power + Rate + CPU_DU_need + CPU_CU_need.
        decision_dim = 5
        self.du_head = nn.Sequential(
            nn.Linear(4 * h + self.global_dim + decision_dim, h), nn.SiLU(), nn.Linear(h, 1)
        )
        self.cu_head = nn.Sequential(
            nn.Linear(5 * h + self.global_dim + decision_dim, h), nn.SiLU(), nn.Linear(h, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.global_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
        # Raw scales đi qua softplus trong forward. Khởi tạo vừa phải để
        # heuristic vật lý hỗ trợ thay vì lấn át embedding học từ topology:
        # softplus([0, -1.5, -1.5]) ~= [0.693, 0.201, 0.201].
        self.prior_scales = nn.Parameter(torch.tensor([0.0, -1.5, -1.5]))

        self.apply(init_linear)
        # Bắt đầu gần uniform weight, sau đó tự học tầm quan trọng node.
        init_linear(self.ran_encoder.weight_encoder[-1], gain=0.01)
        # ActorCritic.apply() also reinitializes every child module.  Set this
        # bias afterwards so the graph branch really starts as a small
        # correction to the local MLP branch: sigmoid(-2) ~= 0.119.
        nn.init.constant_(self.ran_encoder.graph_gate.bias, -2.0)
        # Mọi GraphSAGE residual bắt đầu nhỏ. Bản cũ để bias=0, tương ứng
        # gate=0.5 ngay từ đầu và dễ oversmooth trên topology dày.
        for layer in (
            *self.ran_encoder.graph_layers,
            *self.ran_encoder.demand_fusion,
        ):
            nn.init.constant_(layer.gate.bias, -2.0)
        for gate in (*self.ue_ru_graph.ue_gates, *self.ue_ru_graph.ru_gates):
            nn.init.constant_(gate.bias, -2.0)
        for head in (self.ue_head, self.ru_head, self.rb_head, self.power_head, self.du_head, self.cu_head):
            init_linear(head[-1], gain=0.01)
        init_linear(self.value_head[-1], gain=1.0)

    @staticmethod
    def _categorical(logits, mask, action=None, deterministic=False, fallback_mask=None):
        """Masked Categorical; fallback dùng khi không còn action resource-feasible."""
        mask = mask.bool()
        if fallback_mask is not None:
            fallback_mask = fallback_mask.bool()
            mask = torch.where(mask.any(dim=-1, keepdim=True), mask, fallback_mask)

        # Safety cuối cùng để distribution luôn hữu hạn; chỉ xảy ra nếu topology/domain hỏng.
        safe_first = F.one_hot(
            torch.zeros(mask.size(0), dtype=torch.long, device=mask.device),
            num_classes=mask.size(-1),
        ).bool()
        mask = torch.where(mask.any(dim=-1, keepdim=True), mask, safe_first)

        masked_logits = logits.float().masked_fill(~mask, -1e9)
        distribution = Categorical(logits=masked_logits)

        if action is None:
            action = masked_logits.argmax(dim=-1) if deterministic else distribution.sample()
        else:
            action = action.long()

        if DEBUG_MASKS and not bool(mask.gather(1, action.unsqueeze(1)).all()):
            raise RuntimeError("Action không hợp lệ với action mask của state.")
        # Chuẩn hóa entropy về [0, 1] để cùng một entropy_coef có ý nghĩa
        # nhất quán khi số UE/node/action hợp lệ thay đổi theo topology.
        entropy = distribution.entropy()
        num_valid = mask.sum(dim=-1).to(entropy.dtype)
        max_entropy = torch.log(num_valid.clamp_min(2.0))
        normalized_entropy = torch.where(
            num_valid > 1,
            entropy / max_entropy,
            torch.zeros_like(entropy),
        )
        return action, distribution.log_prob(action), normalized_entropy

    def _encode_state(self, batch):
        node_embeddings, _ = self.ran_encoder(
            batch.node_features, batch.adjacency_matrix, batch.valid_node_mask
        )
        ue_embeddings, _ = self.ue_encoder(
            batch.ue_features, batch.valid_ue_mask, batch.active_ue_mask
        )
        relation_embeddings = self.relation_encoder(
            batch.ue_ru_features, batch.valid_ue_mask, batch.valid_ru_mask
        )

        batch_size, num_ues, num_rus, hidden_dim = relation_embeddings.shape
        batch_index = torch.arange(batch_size, device=node_embeddings.device)
        ru_embeddings = node_embeddings[batch_index[:, None], batch.ru_indices]
        active_mask = batch.valid_ue_mask & batch.active_ue_mask
        ue_embeddings, ru_embeddings = self.ue_ru_graph(
            ue_embeddings,
            ru_embeddings,
            relation_embeddings,
            active_mask,
            batch.valid_ru_mask,
        )

        # Ghi embedding RU đã nhận message từ UE trở lại heterogeneous graph.
        ru_selector = F.one_hot(
            batch.ru_indices, num_classes=node_embeddings.size(1)
        ).to(node_embeddings.dtype)
        ru_selector = ru_selector * batch.valid_ru_mask.unsqueeze(-1)
        ru_updates = torch.bmm(ru_selector.transpose(1, 2), ru_embeddings)
        has_ru_update = ru_selector.sum(dim=1).bool().unsqueeze(-1)
        node_embeddings = torch.where(has_ru_update, ru_updates, node_embeddings)
        node_embeddings = self.ran_encoder.propagate_demand(
            node_embeddings,
            batch.node_features,
            batch.adjacency_matrix,
            batch.valid_node_mask,
        )
        ran_embedding = self.ran_encoder.pool_nodes(
            node_embeddings, batch.node_features, batch.valid_node_mask
        )
        ue_set_embedding = self.ue_encoder.pool_embeddings(
            ue_embeddings, batch.valid_ue_mask, batch.active_ue_mask
        )

        pair_mask = (
            batch.valid_ue_mask.unsqueeze(-1) & batch.valid_ru_mask.unsqueeze(1)
        ).reshape(batch_size * num_ues, num_rus)

        ue_relation_embedding = masked_pool(
            relation_embeddings.reshape(batch_size * num_ues, num_rus, hidden_dim),
            pair_mask,
        )
        ue_relation_embedding = self.ue_relation_norm(
            F.silu(self.ue_relation_pool(ue_relation_embedding))
        ).reshape(batch_size, num_ues, hidden_dim)
        ue_relation_embedding = ue_relation_embedding * batch.valid_ue_mask.unsqueeze(-1)

        relation_set_embedding = self.relation_set_norm(
            F.silu(self.relation_set_pool(masked_pool(ue_relation_embedding, active_mask)))
        )
        relation_set_embedding = relation_set_embedding * active_mask.any(
            dim=-1, keepdim=True
        ).to(relation_set_embedding.dtype)

        total_count = batch.valid_ue_mask.sum(dim=-1).clamp_min(1)
        active_count = active_mask.sum(dim=-1)
        active_ratio = active_count.to(ran_embedding.dtype) / total_count.to(ran_embedding.dtype)
        total_count_log = torch.log1p(total_count.to(ran_embedding.dtype)) / UE_COUNT_LOG_SCALE
        active_count_log = torch.log1p(active_count.to(ran_embedding.dtype)) / UE_COUNT_LOG_SCALE

        global_context = torch.cat([
            ran_embedding,
            ue_set_embedding,
            relation_set_embedding,
            batch.rb_ratio.unsqueeze(-1),
            active_ratio.unsqueeze(-1),
            total_count_log.unsqueeze(-1),
            active_count_log.unsqueeze(-1),
        ], dim=-1)

        return (
            node_embeddings,
            ue_embeddings,
            relation_embeddings,
            ue_relation_embedding,
            global_context,
        )

    def value(self, batch):
        *_, global_context = self._encode_state(batch)
        return self.value_head(global_context).squeeze(-1)

    def forward(self, batch, actions=None, deterministic=False):
        if actions is not None and (actions.ndim != 2 or actions.size(1) != self.NUM_ACTIONS):
            raise ValueError("actions phải có shape [B, 6].")

        node_embeddings, ue_embeddings, relations, ue_relations, global_context = self._encode_state(batch)
        value = self.value_head(global_context).squeeze(-1)

        b = torch.arange(node_embeddings.size(0), device=node_embeddings.device)
        ru_embeddings = node_embeddings[b[:, None], batch.ru_indices]
        du_embeddings = node_embeddings[b[:, None], batch.du_indices]
        cu_embeddings = node_embeddings[b[:, None], batch.cu_indices]
        active_ue_mask = batch.valid_ue_mask & batch.active_ue_mask
        log_probs, entropies = [], []

        # Chỉ cho policy chọn UE/RU có ít nhất một cặp RB-power đủ R_min.
        structural_cu = batch.valid_cu_mask
        structural_du = batch.valid_du_mask & (batch.du_cu & structural_cu.unsqueeze(1)).any(dim=-1)
        structural_ru = batch.valid_ru_mask & (batch.ru_du & structural_du.unsqueeze(1)).any(dim=-1)
        rb_values = torch.arange(1, self.max_rbs + 1, device=value.device, dtype=value.dtype)
        candidate_power = self.power_levels.view(1, 1, 1, 1, -1)
        candidate_rb = rb_values.view(1, 1, 1, -1, 1)
        # P_ib_k là tổng công suất UE trên n_RB:
        # P_per_RB = P_ib_k / n_RB; SNR = P_per_RB * g_i_k.
        candidate_snr = (
            candidate_power
            * batch.ue_ru_gain.unsqueeze(-1).unsqueeze(-1)
            / candidate_rb
        )
        candidate_rate = (
            candidate_rb
            * batch.bandwidth_per_rb.view(-1, 1, 1, 1, 1)
            * torch.log2(1.0 + candidate_snr)
        ).float()
        candidate_valid = batch.rb_mask[:, None, None, :, None]
        candidate_valid = candidate_valid & (
            candidate_power
            <= batch.ru_remaining[:, None, :, None, None] + 1e-9
        )
        candidate_valid = candidate_valid & (
            candidate_rate >= batch.ue_r_min[:, :, None, None, None]
        )

        # Khớp chính xác RLEnv.calculate_latency():
        # L = d/c + Q/R + 1/(R/Q-lambda) + 1/(R/Q-lambda).
        packet_bits = batch.ue_packet_bits[:, :, None, None, None].clamp_min(1.0)
        arrival = batch.ue_lambda_pps[:, :, None, None, None].clamp_min(0.0)
        service = candidate_rate / packet_bits
        stable_queue = service > arrival
        queue_gap = (service - arrival).clamp_min(1e-12)
        latency = (
            batch.ue_ru_distance[:, :, :, None, None]
            / batch.speed_of_light[:, None, None, None, None]
            + packet_bits / candidate_rate.clamp_min(1e-12)
            + 2.0 / queue_gap
        )
        candidate_valid = candidate_valid & stable_queue
        candidate_valid = candidate_valid & (
            latency <= batch.ue_l_max[:, :, None, None, None]
        )

        # Một candidate chỉ hợp lệ khi tồn tại cùng một DU có đủ CPU và nối
        # tới ít nhất một CU đủ CPU. OR theo DU tiết kiệm bộ nhớ hơn tensor 6D.
        cpu_factor = (1.0 + batch.ue_eta).clamp_min(0.0)
        du_denominator = batch.k_du[:, None] * cpu_factor
        cu_denominator = batch.k_cu[:, None] * cpu_factor
        downstream_valid = torch.zeros_like(candidate_valid)
        for du_index in range(batch.du_remaining.size(1)):
            cu_links = batch.du_cu[:, du_index] & batch.valid_cu_mask
            cu_capacity = batch.cu_remaining.masked_fill(~cu_links, -1.0).amax(dim=-1)
            du_rate_limit = torch.where(
                du_denominator > 0,
                batch.du_remaining[:, du_index, None] / du_denominator.clamp_min(1e-12),
                torch.full_like(du_denominator, torch.inf),
            )
            cu_rate_limit = torch.where(
                cu_denominator > 0,
                cu_capacity[:, None] / cu_denominator.clamp_min(1e-12),
                torch.full_like(cu_denominator, torch.inf),
            )
            downstream_valid |= (
                batch.ru_du[:, None, :, du_index, None, None]
                & batch.valid_du_mask[:, None, None, du_index, None, None]
                & (candidate_rate <= du_rate_limit[:, :, None, None, None])
                & (candidate_rate <= cu_rate_limit[:, :, None, None, None])
            )
        candidate_valid = candidate_valid & downstream_valid
        feasible_ue_ru = candidate_valid.any(dim=(-1, -2)) & structural_ru.unsqueeze(1)
        feasible_ue_mask = active_ue_mask & feasible_ue_ru.any(dim=-1)

        # ------------------------------------------------------------------
        # 1) UE
        # ------------------------------------------------------------------
        num_ues = ue_embeddings.size(1)
        ue_context = torch.cat([
            ue_embeddings,
            ue_relations,
            global_context.unsqueeze(1).expand(-1, num_ues, -1),
        ], dim=-1)
        ue_logits = self.ue_head(ue_context).squeeze(-1)
        ue_action, log_prob, entropy = self._categorical(
            ue_logits,
            feasible_ue_mask,
            None if actions is None else actions[:, self.UE],
            deterministic,
            active_ue_mask,
        )
        log_probs.append(log_prob); entropies.append(entropy)
        selected_ue = ue_embeddings[b, ue_action]
        selected_ue_relations = relations[b, ue_action]

        # ------------------------------------------------------------------
        # 2) RU | UE
        # ------------------------------------------------------------------
        num_rus = ru_embeddings.size(1)
        ru_context = torch.cat([
            ru_embeddings,
            selected_ue.unsqueeze(1).expand(-1, num_rus, -1),
            selected_ue_relations,
            global_context.unsqueeze(1).expand(-1, num_rus, -1),
        ], dim=-1)
        prior_scales = F.softplus(self.prior_scales)
        channel_prior = batch.ue_ru_features[b, ue_action, :, 1]
        ru_headroom = batch.node_features[b[:, None], batch.ru_indices, 0]
        ru_logits = self.ru_head(ru_context).squeeze(-1)
        ru_logits = ru_logits + prior_scales[0] * channel_prior + prior_scales[1] * ru_headroom
        ru_mask = feasible_ue_ru[b, ue_action]
        ru_action, log_prob, entropy = self._categorical(
            ru_logits,
            ru_mask,
            None if actions is None else actions[:, self.RU],
            deterministic,
            structural_ru,
        )
        log_probs.append(log_prob); entropies.append(entropy)
        selected_ru = ru_embeddings[b, ru_action]
        selected_relation = selected_ue_relations[b, ru_action]

        # ------------------------------------------------------------------
        # 3) RB | UE, RU
        # ------------------------------------------------------------------
        allocation_context = torch.cat([
            selected_ue, selected_ru, selected_relation, global_context
        ], dim=-1)
        rb_logits = self.rb_head(allocation_context)
        selected_rate_grid = candidate_rate[b, ue_action, ru_action]
        selected_candidate_mask = candidate_valid[b, ue_action, ru_action]
        rb_feasible_mask = selected_candidate_mask.any(dim=-1)
        rb_action, log_prob, entropy = self._categorical(
            rb_logits,
            rb_feasible_mask,
            None if actions is None else actions[:, self.RB],
            deterministic,
            batch.rb_domain_mask,
        )
        log_probs.append(log_prob); entropies.append(entropy)
        num_rbs = rb_action + 1
        rb_float = num_rbs.float()
        rb_feature = (rb_float / self.max_rbs).unsqueeze(-1)

        # ------------------------------------------------------------------
        # 4) Power | UE, RU, RB
        # ------------------------------------------------------------------
        power_logits = self.power_head(torch.cat([allocation_context, rb_feature], dim=-1))
        power_rate = selected_rate_grid[b, rb_action]
        power_mask = selected_candidate_mask[b, rb_action]
        power_action, log_prob, entropy = self._categorical(
            power_logits,
            power_mask,
            None if actions is None else actions[:, self.POWER],
            deterministic,
            torch.ones_like(power_mask),
        )
        log_probs.append(log_prob); entropies.append(entropy)
        power_value = self.power_levels[power_action]
        power_feature = (power_value / self.power_levels[-1]).unsqueeze(-1)
        # Lấy trực tiếp từ grid vật lý đã dùng để tạo mask. Nhờ đó công thức
        # P_per_RB=P_total/n_RB không thể lệch giữa masking và action thực tế.
        rate = power_rate.gather(1, power_action.unsqueeze(1)).squeeze(1)

        eta = batch.ue_eta[b, ue_action]
        cpu_du_need = batch.k_du * rate * (1.0 + eta)
        cpu_cu_need = batch.k_cu * rate * (1.0 + eta)

        rate_feature = (torch.log1p(rate) / RATE_LOG_SCALE).unsqueeze(-1)
        cpu_du_feature = (torch.log1p(cpu_du_need) / CPU_RESOURCE_LOG_SCALE).unsqueeze(-1)
        cpu_cu_feature = (torch.log1p(cpu_cu_need) / CPU_RESOURCE_LOG_SCALE).unsqueeze(-1)
        decision_features = torch.cat([
            rb_feature,
            power_feature,
            rate_feature,
            cpu_du_feature,
            cpu_cu_feature,
        ], dim=-1)

        # ------------------------------------------------------------------
        # 5) DU | UE, RU, RB, Power
        # ------------------------------------------------------------------
        num_dus = du_embeddings.size(1)
        ru_du_links = batch.ru_du[b, ru_action]
        du_domain_mask = ru_du_links & structural_du

        cu_resource_mask = structural_cu & (batch.cu_remaining >= cpu_cu_need.unsqueeze(-1))
        du_has_feasible_cu = (batch.du_cu & cu_resource_mask.unsqueeze(1)).any(dim=-1)
        du_mask = (
            du_domain_mask
            & (batch.du_remaining >= cpu_du_need.unsqueeze(-1))
            & du_has_feasible_cu
        )

        du_context = torch.cat([
            du_embeddings,
            selected_ue.unsqueeze(1).expand(-1, num_dus, -1),
            selected_ru.unsqueeze(1).expand(-1, num_dus, -1),
            selected_relation.unsqueeze(1).expand(-1, num_dus, -1),
            global_context.unsqueeze(1).expand(-1, num_dus, -1),
            decision_features.unsqueeze(1).expand(-1, num_dus, -1),
        ], dim=-1)
        du_headroom = batch.node_features[b[:, None], batch.du_indices, 0]
        du_degree = batch.node_features[b[:, None], batch.du_indices, 6]
        du_logits = self.du_head(du_context).squeeze(-1)
        du_logits = du_logits + prior_scales[1] * du_headroom + prior_scales[2] * du_degree
        du_action, log_prob, entropy = self._categorical(
            du_logits,
            du_mask,
            None if actions is None else actions[:, self.DU],
            deterministic,
            du_domain_mask,
        )
        log_probs.append(log_prob); entropies.append(entropy)
        selected_du = du_embeddings[b, du_action]

        # ------------------------------------------------------------------
        # 6) CU | UE, RU, RB, Power, DU
        # ------------------------------------------------------------------
        num_cus = cu_embeddings.size(1)
        du_cu_links = batch.du_cu[b, du_action]
        cu_domain_mask = du_cu_links & structural_cu
        cu_mask = cu_domain_mask & (batch.cu_remaining >= cpu_cu_need.unsqueeze(-1))

        cu_context = torch.cat([
            cu_embeddings,
            selected_ue.unsqueeze(1).expand(-1, num_cus, -1),
            selected_ru.unsqueeze(1).expand(-1, num_cus, -1),
            selected_du.unsqueeze(1).expand(-1, num_cus, -1),
            selected_relation.unsqueeze(1).expand(-1, num_cus, -1),
            global_context.unsqueeze(1).expand(-1, num_cus, -1),
            decision_features.unsqueeze(1).expand(-1, num_cus, -1),
        ], dim=-1)
        cu_headroom = batch.node_features[b[:, None], batch.cu_indices, 0]
        cu_degree = batch.node_features[b[:, None], batch.cu_indices, 6]
        cu_logits = self.cu_head(cu_context).squeeze(-1)
        cu_logits = cu_logits + prior_scales[1] * cu_headroom + prior_scales[2] * cu_degree
        cu_action, log_prob, entropy = self._categorical(
            cu_logits,
            cu_mask,
            None if actions is None else actions[:, self.CU],
            deterministic,
            cu_domain_mask,
        )
        log_probs.append(log_prob); entropies.append(entropy)

        # Schema buffer giữ nguyên dù thứ tự sinh action đã thay đổi.
        action_indices = torch.stack([
            ue_action, ru_action, du_action, cu_action, rb_action, power_action
        ], dim=-1)

        return {
            "action_indices": action_indices,
            "log_prob": torch.stack(log_probs).sum(dim=0),
            "entropy": torch.stack(entropies, dim=0).mean(dim=0),
            "value": value,
            "num_rbs": num_rbs,
            "power": power_value,
            "rate": rate,
        }


# ======================================================================
# 7. ROLLOUT BUFFER + PPO CONFIG
# ======================================================================

class RolloutBuffer:
    """Rollout buffer; action schema = [UE, RU, DU, CU, RB, Power]."""

    def __init__(self):
        self.clear()

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.next_values = [], [], []
        self.dones, self.env_ids = [], []

    def add(self, state, action, log_prob, reward, value, next_value, done, env_id=0):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (ActorCritic.NUM_ACTIONS,):
            raise ValueError("action phải có shape (6,) = [UE, RU, DU, CU, RB, Power].")

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.dones.append(bool(done))
        self.env_ids.append(int(env_id))

    def compute_gae(self, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        """GAE độc lập theo env_id, hỗ trợ rollout xen kẽ nhiều environment."""
        if not self.rewards:
            raise ValueError("RolloutBuffer đang rỗng.")

        advantages = np.zeros(len(self), dtype=np.float32)
        next_advantage = {}

        for t in reversed(range(len(self))):
            env_id = self.env_ids[t]
            non_terminal = 1.0 - float(self.dones[t])
            delta = (
                self.rewards[t]
                + gamma * non_terminal * self.next_values[t]
                - self.values[t]
            )
            advantages[t] = (
                delta
                + gamma * gae_lambda * non_terminal * next_advantage.get(env_id, 0.0)
            )
            next_advantage[env_id] = advantages[t]

        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return advantages, returns


@dataclass(frozen=True)
class PPOConfig:
    """Config nội bộ; người dùng không cần truyền khi tạo agent."""

    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    learning_rate: float = LEARNING_RATE
    min_learning_rate: float = MIN_LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    adam_eps: float = ADAM_EPS
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_ratio: float = CLIP_RATIO
    value_clip: float = VALUE_CLIP
    value_coef: float = VALUE_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    target_kl: float = TARGET_KL
    entropy_start: float = ENTROPY_COEF_START
    entropy_end: float = ENTROPY_COEF_END
    entropy_anneal_fraction: float = ENTROPY_ANNEAL_FRACTION
    rollout_steps: int = STEPS_PER_UPDATE
    minibatch_size: int = MINIBATCH_SIZE
    update_epochs: int = UPDATE_EPOCHS
    num_envs: int = NUM_ENVS
    device: str = DEVICE
    use_amp: bool = USE_AMP
    pin_memory: bool = PIN_MEMORY
    seed: int = TRAIN_SEED

DEFAULT_CONFIG = PPOConfig()


# ======================================================================
# 8. PPO AGENT
# ======================================================================

class PPOAgent:
    """Inference, PPO update, schedule và checkpoint."""

    ALGORITHM_NAME = "PPO_GSA_MLP"
    ARCHITECTURE_VERSION = 4

    def __init__(self, power_levels, max_rbs, config: PPOConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(self.config.device)
        self.max_rbs = int(max_rbs)
        self.action_power_levels = tuple(float(value) for value in power_levels)
        self.update_count = 0
        self.current_episode = 0
        self.best_validation_reward = -math.inf
        self.best_validation_episode = 0

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.state_builder = StateBuilder()
        self.model = ActorCritic(
            node_dim=StateBuilder.NODE_DIM,
            ue_dim=StateBuilder.UE_DIM,
            relation_dim=StateBuilder.UE_RU_DIM,
            power_levels=self.action_power_levels,
            max_rbs=self.max_rbs,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
            fused=self.device.type == "cuda",
        )

        self.use_amp = self.device.type == "cuda" and self.config.use_amp
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )
        self.entropy_coef = self.config.entropy_start

    @classmethod
    def from_state(cls, state, config: PPOConfig | None = None):
        """Khởi tạo trực tiếp từ public state của RLEnv; không cần truyền config."""
        spec = EnvironmentSpec.from_state(state)
        return cls(spec.power_levels, spec.max_rbs_per_ue, config=config)

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _validate_env(self, state):
        """Đảm bảo action domain của state khớp với model/checkpoint."""
        spec = EnvironmentSpec.from_state(state)
        env_power = np.asarray(spec.power_levels, dtype=np.float64)
        action_power = np.asarray(self.action_power_levels, dtype=np.float64)

        if spec.max_rbs_per_ue != self.max_rbs:
            raise ValueError("max_RBs_per_UE của environment không khớp model.")
        if env_power.shape != action_power.shape or not np.array_equal(env_power, action_power):
            raise ValueError("Power codebook của environment không khớp model.")

    @staticmethod
    def _has_active_ue(state):
        return any(bool(ue["status"].get("active", 0)) for ue in state["state_UEs_req"].values())

    def _batch_from_raw(self, states):
        for state in states:
            self._validate_env(state)
        tensor_states = [self.state_builder.build(state) for state in states]
        batch = self.state_builder.collate(
            tensor_states,
            self.max_rbs,
            pin_memory=self.device.type == "cuda" and self.config.pin_memory,
        ).to(self.device)
        return tensor_states, batch

    def act(self, states, deterministic=False):
        """Sinh action cho một batch raw state."""
        if not states:
            raise ValueError("states không được rỗng.")
        if not all(self._has_active_ue(state) for state in states):
            raise RuntimeError("agent.act() được gọi trên state không còn UE active.")

        tensor_states, batch = self._batch_from_raw(states)
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            output = self.model(batch, deterministic=deterministic)

        action_indices = output["action_indices"]
        ue_ids = batch.ue_ids.gather(
            1, action_indices[:, self.model.UE:self.model.UE + 1]
        ).squeeze(1)

        action_cpu = action_indices.cpu().numpy()
        ue_ids = ue_ids.cpu().numpy()
        num_rbs = output["num_rbs"].cpu().numpy()

        env_actions = [
            (
                int(ue_ids[i]),
                int(action[self.model.RU]),
                int(action[self.model.DU]),
                int(action[self.model.CU]),
                self.action_power_levels[int(action[self.model.POWER])],
                int(num_rbs[i]),
            )
            for i, action in enumerate(action_cpu)
        ]

        return {
            "tensor_states": tensor_states,
            "env_actions": env_actions,
            "action_indices": action_cpu,
            "log_probs": output["log_prob"].float().cpu().numpy(),
            "values": output["value"].float().cpu().numpy(),
        }

    def values(self, states):
        """Tính V(s) cho một batch raw state."""
        if not states:
            return np.empty(0, dtype=np.float32)

        _, batch = self._batch_from_raw(states)
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            values = self.model.value(batch)
        return values.float().cpu().numpy()

    def _set_schedules(self, episode):
        """Cosine LR decay + linear entropy annealing."""
        self.current_episode = int(episode)
        progress = min(max(self.current_episode / max(MAX_EPISODES, 1), 0.0), 1.0)

        entropy_progress = min(
            progress / max(self.config.entropy_anneal_fraction, 1e-8), 1.0
        )
        self.entropy_coef = (
            self.config.entropy_start
            + (self.config.entropy_end - self.config.entropy_start) * entropy_progress
        )

        lr = self.config.min_learning_rate + 0.5 * (
            self.config.learning_rate - self.config.min_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def update(self, buffer, episode):
        """PPO clipped update với value clipping và target-KL early stopping."""
        if len(buffer) == 0:
            raise ValueError("RolloutBuffer đang rỗng.")

        cfg = self.config
        lr = self._set_schedules(episode)
        advantages, returns = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions = np.asarray(buffer.actions, dtype=np.int64)
        old_log_probs = np.asarray(buffer.log_probs, dtype=np.float32)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_minibatches = 0
        epochs_done = 0
        self.model.train()

        for epoch in range(cfg.update_epochs):
            epoch_kls = []

            for indices in bucketed_minibatches(
                buffer.states, cfg.minibatch_size
            ):
                batch = self.state_builder.collate(
                    [buffer.states[i] for i in indices],
                    self.max_rbs,
                    pin_memory=self.device.type == "cuda" and cfg.pin_memory,
                ).to(self.device)

                action = torch.as_tensor(actions[indices], dtype=torch.long, device=self.device)
                old_log_prob = torch.as_tensor(old_log_probs[indices], dtype=torch.float32, device=self.device)
                advantage = torch.as_tensor(advantages[indices], dtype=torch.float32, device=self.device)
                return_value = torch.as_tensor(returns[indices], dtype=torch.float32, device=self.device)

                with self._autocast():
                    output = self.model(batch, actions=action)

                # Các phép PPO nhạy số luôn tính float32.
                new_log_prob = output["log_prob"].float()
                new_value = output["value"].float()
                entropy = output["entropy"].float().mean()
                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                policy_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage,
                ).mean()

                # Return của môi trường có thể ở thang hàng chục/hàng trăm.
                # Value clip tuyệt đối 0.2 làm critic gần như không thể bắt kịp
                # target. Smooth-L1 học nhanh hơn MSE nhưng vẫn chống outlier.
                value_loss = F.smooth_l1_loss(
                    new_value,
                    return_value,
                    beta=1.0,
                )

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_fraction"] += clip_fraction.item()
                epoch_kls.append(approx_kl.item())
                num_minibatches += 1

            epochs_done = epoch + 1
            if epoch_kls and np.mean(epoch_kls) > 1.5 * cfg.target_kl:
                break

        self.update_count += 1
        for key in metrics:
            metrics[key] /= max(num_minibatches, 1)
        metrics.update({
            "lr": lr,
            "entropy_coef": self.entropy_coef,
            "epochs": epochs_done,
        })
        return metrics

    def save(self, path, **metadata):
        """Lưu model + optimizer + AMP + RNG để có thể resume training."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "algorithm": self.ALGORITHM_NAME,
            "architecture_version": self.ARCHITECTURE_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.config),
            "power_levels": self.action_power_levels,
            "max_rbs": self.max_rbs,
            "update_count": self.update_count,
            "current_episode": self.current_episode,
            "best_validation_reward": self.best_validation_reward,
            "best_validation_episode": self.best_validation_episode,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metadata": metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device=None, restore_rng=False):
        """Load checkpoint; restore_rng=True nếu muốn tiếp tục đúng RNG state."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("algorithm") not in (None, cls.ALGORITHM_NAME):
            raise ValueError(
                f"Checkpoint {checkpoint.get('algorithm')} không phải {cls.ALGORITHM_NAME}."
            )
        version = int(checkpoint.get("architecture_version", 0))
        if version != cls.ARCHITECTURE_VERSION:
            raise ValueError(
                f"Checkpoint architecture v{version}, code yêu cầu "
                f"v{cls.ARCHITECTURE_VERSION}. Hãy train lại vào thư mục mới."
            )

        config_data = dict(checkpoint["config"])
        if device is not None:
            config_data["device"] = str(device)
        config = PPOConfig(**config_data)

        power_levels = checkpoint["power_levels"]
        if torch.is_tensor(power_levels):
            power_levels = power_levels.numpy()

        agent = cls(power_levels, checkpoint["max_rbs"], config)
        agent.model.load_state_dict(checkpoint["model"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            agent.scaler.load_state_dict(checkpoint["scaler"])
        agent.update_count = int(checkpoint.get("update_count", 0))
        agent.current_episode = int(checkpoint.get("current_episode", 0))
        agent.best_validation_reward = float(
            checkpoint.get("best_validation_reward", -math.inf)
        )
        agent.best_validation_episode = int(
            checkpoint.get("best_validation_episode", 0)
        )

        for optimizer_state in agent.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(agent.device)

        if restore_rng:
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        return agent, checkpoint.get("metadata", {})


# ======================================================================
# 9. TRAINING
# ======================================================================
@torch.inference_mode()
def validate_agent(
    agent,
    env,
    *,
    env_factory=None,
    ue_range=None,
    episodes=VALIDATION_EPISODES,
    seed_start=VALIDATION_SEED_START,
):
    """Đánh giá deterministic trên tập seed cố định, tách khỏi training.

    Validation chỉ dùng để chọn checkpoint. Evaluation cuối cùng phải dùng
    một tập seed khác để tránh selection bias.
    """
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("VALIDATION_EPISODES phải > 0.")

    if ue_range is not None:
        ue_min, ue_max = map(int, ue_range)
        if env_factory is None or ue_min <= 0 or ue_max < ue_min:
            raise ValueError("ue_range validation cần env_factory và 0 < min <= max.")
        # Bảo đảm mỗi kích thước UE xuất hiện ít nhất một lần.
        episodes = max(episodes, ue_max - ue_min + 1)

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rewards = []

    try:
        for index in range(episodes):
            seed = int(seed_start) + index

            if ue_range is None:
                worker = copy.deepcopy(env)
                state = reset_fixed_env(worker, seed)
            else:
                num_ues = ue_min + index % (ue_max - ue_min + 1)
                # Dùng index nằm ngoài training range để tạo snapshot held-out.
                worker = env_factory(num_ues, seed)
                state = worker.get_state()
            num_ues = len(state["state_UEs_req"])
            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                policy = agent.act([state], deterministic=True)
                state, reward, done, _ = worker.step(policy["env_actions"][0])
                total_reward += float(reward)
                steps += 1

                if steps > num_ues:
                    raise RuntimeError(
                        "Validation: số step vượt số UE; kiểm tra RLEnv."
                    )

            rewards.append(total_reward)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    return float(np.mean(rewards)), float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0


def train_agent(agent, env, results_root, *, env_factory=None, ue_range=None):
    """Huấn luyện song song và cân bằng trên nhiều kích thước UE."""
    os.makedirs(results_root, exist_ok=True)
    prefix = CHECKPOINT_PREFIX
    paths = {
        "reward": os.path.join(results_root, f"{prefix}_train_reward_hist.txt"),
        "accept": os.path.join(results_root, f"{prefix}_train_accept_hist.txt"),
        "throughput": os.path.join(results_root, f"{prefix}_train_throughput_hist.txt"),
        "latency": os.path.join(results_root, f"{prefix}_train_latency_hist.txt"),
        "total_latency": os.path.join(results_root, f"{prefix}_train_total_latency_hist.txt"),
    }

    start_episode = agent.current_episode + 1
    episodes_left = MAX_EPISODES - agent.current_episode
    if episodes_left <= 0:
        return {"episodes": [], "updates": []}
    if agent.config.num_envs <= 0:
        raise ValueError("num_envs phải lớn hơn 0.")

    if ue_range is not None:
        ue_min, ue_max = map(int, ue_range)
        if env_factory is None or ue_min <= 0 or ue_max < ue_min:
            raise ValueError("ue_range cần 0 < min <= max và env_factory.")
        ue_range = (ue_min, ue_max)

    cycle_orders = {}

    def sample_num_ues(episode_index):
        ue_min, ue_max = ue_range
        if UE_SIZE_SAMPLING == "transition_balanced":
            sizes = np.arange(ue_min, ue_max + 1, dtype=np.int64)
            # Một episode K UE sinh K transition. P(K) ∝ 1/K làm mỗi kích
            # thước đóng góp xấp xỉ ngang nhau vào PPO update.
            probabilities = 1.0 / sizes.astype(np.float64)
            probabilities /= probabilities.sum()
            return int(np.random.default_rng(
                TRAIN_SEED + int(episode_index)
            ).choice(sizes, p=probabilities))
        if UE_SIZE_SAMPLING != "episode_balanced":
            raise ValueError(
                "PPO_UE_SIZE_SAMPLING phải là transition_balanced "
                "hoặc episode_balanced."
            )
        span = ue_max - ue_min + 1
        cycle, position = divmod(episode_index, span)
        if cycle not in cycle_orders:
            cycle_orders[cycle] = np.random.default_rng(
                TRAIN_SEED + cycle
            ).permutation(span)
        order = cycle_orders[cycle]
        return ue_min + int(order[position])

    def new_episode(episode_index):
        if ue_range is not None:
            # Constructor RLEnv đã sinh đúng một snapshot mới cho episode.
            worker = env_factory(sample_num_ues(episode_index), episode_index)
            state = worker.get_state()
        else:
            worker = copy.deepcopy(env)
            state = reset_fixed_env(worker, TRAIN_SEED + episode_index)
        return worker, state

    num_workers = min(agent.config.num_envs, episodes_left)
    episode_offset = agent.current_episode
    initial = [new_episode(episode_offset + index) for index in range(num_workers)]
    environments = [item[0] for item in initial]
    worker_ids = list(range(num_workers))
    states = [item[1] for item in initial]
    starts = [time.perf_counter() for _ in environments]
    totals = [
        {
            "reward": 0.0,
            "accepted": 0,
            "throughput": 0.0,
            "latency": 0.0,
            "steps": 0,
            "num_ues": len(state["state_UEs_req"]),
        }
        for state in states
    ]

    mode = "a" if agent.current_episode > 0 else "w"
    files = {name: open(path, mode) for name, path in paths.items()}
    buffer = RolloutBuffer()
    history = {"episodes": [], "updates": [], "validations": []}
    total_steps = 0
    launched = num_workers
    completed = agent.current_episode
    start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=min(CPU_ENV_WORKERS, num_workers)) if CPU_ENV_WORKERS > 1 else None
    next_validation_episode = (
        ((agent.current_episode // VALIDATE_EVERY) + 1) * VALIDATE_EVERY
        if VALIDATE_EVERY > 0 else None
    )

    try:
        while completed < MAX_EPISODES:
            policy = agent.act(states, deterministic=False)
            tensor_states = policy["tensor_states"]
            if executor is None:
                transitions = [worker.step(policy["env_actions"][i]) for i, worker in enumerate(environments)]
            else:
                futures = [executor.submit(worker.step, policy["env_actions"][i]) for i, worker in enumerate(environments)]
                transitions = [future.result() for future in futures]

            bootstrap_states = []
            bootstrap_positions = []

            for position, (next_state, _, done, _) in enumerate(transitions):
                if not done:
                    bootstrap_positions.append(position)
                    bootstrap_states.append(next_state)

            next_values = np.zeros(len(environments), dtype=np.float32)
            if bootstrap_states:
                next_values[bootstrap_positions] = agent.values(bootstrap_states)

            finished_workers = []
            for position, (next_state, reward, done, info) in enumerate(transitions):
                buffer.add(
                    tensor_states[position],
                    policy["action_indices"][position],
                    policy["log_probs"][position],
                    reward,
                    policy["values"][position],
                    next_values[position],
                    done,
                    worker_ids[position],
                )

                total = totals[position]
                total["reward"] += float(reward)
                accepted = int(info.get("pi_k", info.get("accepted", info.get("success", 0))))
                total["accepted"] += accepted
                if accepted:
                    total["throughput"] += float(info["R_k"] if "R_k" in info else info["throughput_bps"])
                    total["latency"] += float(info["L_k"] if "L_k" in info else info["latency_s"])
                total["steps"] += 1
                total_steps += 1
                states[position] = next_state

                if total["steps"] > total["num_ues"]:
                    raise RuntimeError(
                        "Số step vượt số UE; kiểm tra logic deactivate UE trong RLEnv."
                    )
                if done:
                    finished_workers.append(position)

            retired_workers = []
            for position in finished_workers:
                completed += 1
                total = totals[position]
                accepted = total["accepted"]
                num_ues = total["num_ues"]
                row = {
                    "episode": completed,
                    "num_ues": num_ues,
                    "reward": total["reward"],
                    "accepted": accepted,
                    "acceptance_rate": accepted / max(num_ues, 1),
                    "throughput": total["throughput"],
                    "avg_latency": total["latency"] / max(accepted, 1),
                    "total_latency": total["latency"],
                    "steps": total["steps"],
                    "elapsed": time.perf_counter() - starts[position],
                }
                history["episodes"].append(row)

                for key, value in (
                    ("reward", row["reward"]),
                    ("accept", row["accepted"]),
                    ("throughput", row["throughput"]),
                    ("latency", row["avg_latency"]),
                    ("total_latency", row["total_latency"]),
                ):
                    files[key].write(f"{value}\n")
                    files[key].flush()

                if completed == start_episode or (
                    PRINT_EVERY > 0 and completed % PRINT_EVERY == 0
                ):
                    recent = history["episodes"][-PRINT_EVERY:]
                    mean_reward = float(np.mean([x["reward"] for x in recent]))
                    print(
                        f"[{DISPLAY_NAME} Ep {completed:05d}/{MAX_EPISODES}] "
                        f"Reward={row['reward']:8.3f} | AvgReward={mean_reward:8.3f} | "
                        f"Accept={accepted:3d}/{num_ues} "
                        f"({100.0 * row['acceptance_rate']:5.1f}%) | "
                        f"Thr={row['throughput']/1e6:8.2f} Mbps | "
                        f"Lat/UE={row['avg_latency']*1e3:7.3f} ms | "
                        f"Steps={total_steps:8d} | Updates={agent.update_count:4d}"
                    )

                agent.current_episode = completed
                if CHECKPOINT_EVERY > 0 and completed % CHECKPOINT_EVERY == 0:
                    agent.save(
                        os.path.join(results_root, f"{prefix}_latest.pt"),
                        episodes=completed,
                        seed=TRAIN_SEED,
                        total_steps=total_steps,
                        ue_range=ue_range,
                        ue_size_sampling=UE_SIZE_SAMPLING,
                    )

                if launched < episodes_left:
                    episode_index = episode_offset + launched
                    environments[position], states[position] = new_episode(episode_index)
                    starts[position] = time.perf_counter()
                    totals[position] = {
                        "reward": 0.0,
                        "accepted": 0,
                        "throughput": 0.0,
                        "latency": 0.0,
                        "steps": 0,
                        "num_ues": len(states[position]["state_UEs_req"]),
                    }
                    launched += 1
                else:
                    retired_workers.append(position)

            if len(buffer) >= agent.config.rollout_steps:
                metrics = agent.update(buffer, completed)
                buffer.clear()
                metrics.update({
                    "update": agent.update_count,
                    "episode": completed,
                    "steps": total_steps,
                })
                history["updates"].append(metrics)

            if (
                next_validation_episode is not None
                and completed >= next_validation_episode
            ):
                validation_mean, validation_std = validate_agent(
                    agent,
                    env,
                    env_factory=env_factory,
                    ue_range=ue_range,
                )
                validation = {
                    "episode": completed,
                    "mean_reward": validation_mean,
                    "std_reward": validation_std,
                }
                history["validations"].append(validation)

                improved = validation_mean > agent.best_validation_reward
                if improved:
                    agent.best_validation_reward = validation_mean
                    agent.best_validation_episode = completed
                    agent.save(
                        os.path.join(results_root, f"{prefix}_best.pt"),
                        episodes=completed,
                        seed=TRAIN_SEED,
                        total_steps=total_steps,
                        validation=validation,
                        ue_range=ue_range,
                        ue_size_sampling=UE_SIZE_SAMPLING,
                    )

                print(
                    f"[{DISPLAY_NAME} Validation @ {completed:05d}] "
                    f"Reward={validation_mean:.3f} +/- {validation_std:.3f} | "
                    f"Best={agent.best_validation_reward:.3f} "
                    f"@ {agent.best_validation_episode:05d}" +
                    (" | SAVED" if improved else "")
                )

                while next_validation_episode <= completed:
                    next_validation_episode += VALIDATE_EVERY

            if retired_workers:
                keep = [
                    index for index in range(len(environments))
                    if index not in retired_workers
                ]
                environments = [environments[index] for index in keep]
                states = [states[index] for index in keep]
                starts = [starts[index] for index in keep]
                totals = [totals[index] for index in keep]
                worker_ids = [worker_ids[index] for index in keep]

        if len(buffer):
            metrics = agent.update(buffer, completed)
            metrics.update({
                "update": agent.update_count,
                "episode": completed,
                "steps": total_steps,
            })
            history["updates"].append(metrics)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for stream in files.values():
            stream.close()

    agent.current_episode = completed
    agent.save(
        os.path.join(results_root, f"{prefix}_model.pt"),
        episodes=completed,
        seed=TRAIN_SEED,
        total_steps=total_steps,
        ue_range=ue_range,
        ue_size_sampling=UE_SIZE_SAMPLING,
    )
    print(
        f"Finished | Episodes={completed} | Steps={total_steps} | "
        f"Updates={agent.update_count} | Time={time.time() - start_time:.2f}s"
    )
    return history
