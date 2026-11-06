from __future__ import annotations

import copy
import numpy as np
from numpy.linalg import norm

from env.RANManager import RANManager
from env.UEManager import UEManager
from config import (P_ib_k_val, W_b, carrier_frequency_GHz, k_CU, k_DU, max_RBs_per_UE, noise_figure_dB, num_ru_antennas, reference_temperature_K, w_acc, w_thr, w_lat)

class Env:
    def __init__(self, num_UEs, num_RBs, num_RUs, num_DUs, num_CUs):
        self.num_UEs = int(num_UEs)
        self.num_RBs = int(num_RBs)
        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        
        self.RAN_manager = None
        self.UE_manager = None
        
        self.coordinate_UEs = None
        self.coordinate_RUs = None
        self.d_i_k = None
        self.g_i_k = None

        self.env_config = {}
        self.env_snapshot = {}
        
        self.reset()
        
    # ==================================================================
    # RESET
    # ==================================================================
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.RAN_manager = RANManager(self.num_RBs, self.num_RUs, self.num_DUs, self.num_CUs)
        self.UE_manager = UEManager(self.num_UEs)
        
        self.coordinate_UEs = self.UE_manager.coordinate_UEs
        self.coordinate_RUs = self.RAN_manager.coordinate_RUs
        
        self.d_i_k = self.calculate_distances(self.coordinate_UEs, self.coordinate_RUs)
        self.g_i_k = self.calculate_gain(self.d_i_k)
        
        self.RAN_manager.get_state_RAN()
        self.UE_manager.get_state_UEs_req(self.coordinate_UEs, self.d_i_k, self.g_i_k)
        
        self.get_env_config()
        
        return self.get_state()
    
    def get_env_config(self):
        self.env_config = {
            "W_b": float(W_b),
            "k_DU": float(k_DU),
            "k_CU": float(k_CU),
            "w_acc": float(w_acc),
            "w_thr": float(w_thr),
            "w_lat": float(w_lat),
            "P_ib_k_val": copy.deepcopy(P_ib_k_val),
            "max_RBs_per_UE": int(max_RBs_per_UE),
        }

        return self.env_config
    
    def get_state(self):
        """Trả snapshot độc lập của trạng thái hiện tại."""
        self.env_snapshot = {
            "state_RAN": self.RAN_manager.state_RAN,
            "state_UEs_req": self.UE_manager.state_UEs_req,
            "env_config": self.env_config,
        }

        return copy.deepcopy(self.env_snapshot)
    
    # ==================================================================
    # DISTANCES
    # ==================================================================
    def calculate_distances(self, coordinate_UEs, coordinate_RUs):
        """Tính khoảng cách 2D từ tất cả RU tới tất cả UE."""
        d_i_k = []
        for ru in coordinate_RUs:
            ue_distances = []
            for ue in coordinate_UEs:
                x_ue, y_ue = ue
                x_ru, y_ru = ru
                distance = np.hypot(float(x_ru) - float(x_ue), float(y_ru) - float(y_ue))
                
                ue_distances.append(max(float(distance), 1.0))
            
            d_i_k.append(ue_distances)

        return np.asarray(d_i_k, dtype=float)

    # ==================================================================
    # GAIN
    # ==================================================================
    def calculate_gain(self, d_i_k):
        """Tính channel-to-noise gain cho từng liên kết RU-UE."""

        # 1. THERMAL NOISE

        k_B = 1.38064852e-23
        N0_W_per_Hz = k_B * reference_temperature_K
        noise_figure_linear = 10 ** (noise_figure_dB / 10.0)
        thermal_noise_power_RB = N0_W_per_Hz * W_b * noise_figure_linear

        shadowing_std_dB = 8.0
        fast_fading_std_dB = 5.0
        interference_std_dB = 6.0

        blockage_probability = 0.12
        blockage_loss_dB = (8.0, 20.0)

        g_i_k = []

        for d_i in d_i_k:
            g_i = []

            for d in d_i:
                d = max(float(d), 1.0)

                # 2. LARGE-SCALE FADING

                path_loss_db = 28.0 + 20.0 * np.log10(carrier_frequency_GHz) + 22.0 * np.log10(d)
                path_loss_db += np.random.normal(0.0, shadowing_std_dB)

                if np.random.random() < blockage_probability:
                    path_loss_db += np.random.uniform(*blockage_loss_dB)

                path_loss_linear = 10 ** (-path_loss_db / 10.0)

                # 3. SMALL-SCALE FADING

                h_real = np.random.randn(num_ru_antennas)
                h_imag = np.random.randn(num_ru_antennas)
                h = np.sqrt(path_loss_linear) * (h_real + 1j * h_imag) / np.sqrt(2.0)

                channel_power_gain = norm(h, 2) ** 2
                fast_fading_dB = np.random.normal(0.0, fast_fading_std_dB)
                channel_power_gain *= 10 ** (fast_fading_dB / 10.0)

                # 4. INTERFERENCE

                interference_dB = np.clip(np.random.normal(0.0, interference_std_dB), -10.0, 20.0)
                interference_factor = 10 ** (interference_dB / 10.0)
                effective_noise_power_RB = thermal_noise_power_RB * (1.0 + interference_factor)

                normalized_gain = channel_power_gain / effective_noise_power_RB
                g_i.append(float(np.clip(normalized_gain, 1e-15, 1e15)))

            g_i_k.append(g_i)

        return np.asarray(g_i_k, dtype=float)
        
        
        
        
