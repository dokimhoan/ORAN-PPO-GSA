"""Môi trường RL tuần tự sử dụng cùng snapshot và logic đánh giá của Env."""

from __future__ import annotations

import copy

import numpy as np

from env.Env import Env

from config import speed_of_light_mps

class RLEnv(Env):
    """Một episode là một snapshot; mỗi step xử lý đúng một UE."""
    def reset(self, seed=None):
        state = super().reset(seed=seed)
        self.episode_done = self.UE_manager.check_UE_all_inactive()
        return copy.deepcopy(state)
    
    def calculate_throughput(self, UE_infor, phi_i_k, num_z_i_k, num_P_i_k):
            """Tính SNR và thông lượng theo mô hình vô tuyến hiện tại.
    
            p_RB = p_UE / n_RB
            SNR  = p_RB * gain
            R    = n_RB * W_RB * log2(1 + SNR)
            """
            power_per_RB = float(num_P_i_k) / int(num_z_i_k)
            g_i_k = float(UE_infor["g_i_k"][phi_i_k])
            SNR_linear = power_per_RB * g_i_k
    
            R_k = num_z_i_k * self.env_config["W_b"] * np.log2(1.0 + SNR_linear)
    
            SNR_db = 10.0 * np.log10(max(SNR_linear, 1e-30))
            
            return float(R_k), float(SNR_linear), float(SNR_db)
        
    def compute_cpu_request(self, UE_infor, R_k):
            
        eta_k = float(UE_infor.get("eta_k", 0.0))
        k_DU = self.env_config["k_DU"]
        k_CU = self.env_config["k_CU"]
        cpu_DU_req =  k_DU * R_k * (1 + eta_k)
        cpu_CU_req =  k_CU * R_k * (1 + eta_k)
            
        return cpu_DU_req, cpu_CU_req
    
    def calculate_latency(self, UE_infor, phi_i_k, R_k):
        """
        E2E latency:
    
            L_k = L_prop,k + L_tran,k + L_DU,k + L_CU,k
            L_prop,k = d_i,k / c
            L_tran,k = Q_k / R_k
            mu_k     = R_k / Q_k
            L_DU,k   = 1 / (mu_k - lambda_k)
            L_CU,k   = 1 / (mu_k - lambda_k)
    
        Nếu mu_k <= lambda_k thì hàng đợi không ổn định và L_k = inf.
        """
            
        packet_size_bits_k = float(UE_infor["packet_size_bits"])
        lambda_k = float(UE_infor["lambda_pps"])
        distance_m = float(UE_infor["d_i_k"][phi_i_k])
            
        # -------------------- Propagation delay --------------------
        L_prop_k = distance_m / float(speed_of_light_mps)
            
        # -------------------- Transmission delay --------------------
        L_tran_k = packet_size_bits_k / R_k
    
        mu_k = float(R_k) / packet_size_bits_k
        if mu_k <= lambda_k:
            return np.inf, L_prop_k, L_tran_k, np.inf, np.inf
        # -------------------- DU delay --------------------
        L_du_k = 1.0 / (mu_k - lambda_k)
            
        # -------------------- CU delay --------------------
        L_cu_k = 1.0 / (mu_k - lambda_k)
    
        L_k = L_prop_k + L_tran_k + L_du_k + L_cu_k
            
        return L_k, L_prop_k, L_tran_k, L_du_k, L_cu_k
    
    def check_action(self, UE_k, phi_i_k, phi_j_k, phi_m_k, num_P_i_k, num_z_i_k):
        """Kiểm tra các giá trị của action có hợp lệ hay không."""
        # -------------------- UE --------------------
        if UE_k not in self.UE_manager.state_UEs_req:
            return False, "invalid_UE"

        if not bool(self.UE_manager.state_UEs_req[UE_k]["status"]["active"]):
            return False, "UE_already_processed"

        # -------------------- RU / DU / CU --------------------
        if not 0 <= phi_i_k < self.num_RUs:
            return False, "invalid_RU"

        if not 0 <= phi_j_k < self.num_DUs:
            return False, "invalid_DU"

        if not 0 <= phi_m_k < self.num_CUs:
            return False, "invalid_CU"

        # -------------------- Power --------------------
        if not np.isfinite(num_P_i_k) or num_P_i_k <= 0.0:
            return False, "invalid_power"

        power_levels = np.asarray(self.env_config["P_ib_k_val"], dtype=float)

        if not np.any(np.isclose(power_levels, num_P_i_k, atol=1e-8, rtol=0.0)):
            return False, "invalid_power_level"

        # -------------------- RB --------------------
        if not 0 <= num_z_i_k <= int(self.env_config["max_RBs_per_UE"]):
            return False, "invalid_num_RB"

        return True, "valid_action"
    
    def check_feasible(self, UE_infor, phi_i_k, phi_j_k, phi_m_k, num_P_i_k, num_z_i_k, R_k, SNR_db, L_k, cpu_DU_req, cpu_CU_req):

        state_RAN = self.RAN_manager.state_RAN

        # -------------------- Link RU-DU --------------------
        if state_RAN["l_ru_du"][phi_i_k, phi_j_k] != 1:
            return False, "RU_DU_not_connected"

        # -------------------- Link DU-CU --------------------
        if state_RAN["l_du_cu"][phi_j_k, phi_m_k] != 1:
            return False, "DU_CU_not_connected"

        # -------------------- RB Resource --------------------
        if num_z_i_k > state_RAN["RB_rem"]:
            return False, "insufficient_RB"

        # -------------------- RU Power --------------------
        if num_P_i_k > state_RAN["P_i_rem"][phi_i_k]:
            return False, "insufficient_RU_power"

        # -------------------- DU CPU --------------------
        if cpu_DU_req > state_RAN["A_j_rem"][phi_j_k]:
            return False, "insufficient_DU_CPU"

        # -------------------- CU CPU --------------------
        if cpu_CU_req > state_RAN["A_m_rem"][phi_m_k]:
            return False, "insufficient_CU_CPU"

        # -------------------- Minimum throughput --------------------
        if not np.isfinite(R_k) or R_k < float(UE_infor["R_min_k"]):
            return False, "R_min_not_satisfied"

        # -------------------- Minimum SNR --------------------
        # if not np.isfinite(SNR_db) or SNR_db < float(UE_infor["SINR_min_k"]):
        #     return False, "SINR_min_not_satisfied"

        # -------------------- Queue stability --------------------
        if not np.isfinite(L_k):
            return False, "unstable_queue"

        # -------------------- Maximum latency --------------------
        if L_k > float(UE_infor["L_max_k"]):
            return False, "L_max_not_satisfied"

        return True, "accepted"
    
    def compute_reward(self, UE_infor, R_k, L_k):
        """Tính reward của UE được accept."""

        R_min_k = float(UE_infor["R_min_k"])
        L_max_k = float(UE_infor["L_max_k"])

        w_acc_k = float(UE_infor.get("w_acc_s", 1.0))
        w_thr_k = float(UE_infor.get("w_thr_s", 1.0))
        w_lat_k = float(UE_infor.get("w_lat_s", 1.0))

        acc_term = self.env_config["w_acc"] * w_acc_k
        thr_term = self.env_config["w_thr"] * w_thr_k * R_k / R_min_k
        lat_term = self.env_config["w_lat"] * w_lat_k * L_k / L_max_k
        
        reward = acc_term + thr_term - lat_term

        return float(reward)
    
    def accept_UE(self, UE_k, allocation, reward):
        """Accept UE và cập nhật tài nguyên."""
        self.RAN_manager.update_resources(
            allocation["phi_i_k"], allocation["phi_j_k"], allocation["phi_m_k"],
            allocation["num_z_i_k"], allocation["num_P_i_k"],
            allocation["cpu_DU_req"], allocation["cpu_CU_req"],
        )

        self.UE_manager.update_UE_request(UE_k, {
            "allocation": allocation,
            "status": {"active": 0, "served": True, "reason": "accepted"},
            "reward": float(reward),
        })

        self.RAN_manager.get_state_RAN()
        
    def reject_UE(self, UE_k, reason):
        """Reject UE nhưng không thay đổi tài nguyên."""
        self.UE_manager.update_UE_request(UE_k, {
            "allocation": self.UE_manager.empty_alloc(),
            "status": {"active": 0, "served": False, "reason": reason},
            "reward": 0.0,
        })
    
    def step(self, action):
        """
        Xử lý một UE.

        action = (UE_k, phi_i_k, phi_j_k, phi_m_k, num_P_i_k, num_z_i_k)
        return = (state, reward, done, info)
        """
        if self.episode_done:
            return self.get_state(), 0.0, True, {
                "pi_k": 0,
                "reason": "episode_already_done",
            }
            
        if len(action) != 6:
            return self.get_state(), 0.0, self.episode_done, {
                "pi_k": 0,
                "reason": "invalid_action_size",
            }
        
        try:
            UE_k = int(action[0])
            phi_i_k = int(action[1])
            phi_j_k = int(action[2])
            phi_m_k = int(action[3])
            num_P_i_k = float(action[4])
            num_z_i_k = int(action[5])
        except (TypeError, ValueError, OverflowError) as error:
            return self.get_state(), 0.0, self.episode_done, {
                "pi_k": 0,
                "reason": f"invalid_action: {error}",
            }
        
        valid_action, reason_action = self.check_action(UE_k, phi_i_k, phi_j_k, phi_m_k, num_P_i_k, num_z_i_k)
        
        if not valid_action:
            return self.get_state(), 0.0, self.episode_done, {
                "pi_k": 0,
                "reason": reason_action,
                "UE_k": UE_k,
                "phi_i_k": phi_i_k,
                "phi_j_k": phi_j_k,
                "phi_m_k": phi_m_k,
                "num_P_i_k": num_P_i_k,
                "num_z_i_k": num_z_i_k,
            }
 
        UE_infor = self.UE_manager.state_UEs_req[UE_k]
        
        R_k, SNR_linear, SNR_db = self.calculate_throughput(UE_infor, phi_i_k, num_z_i_k, num_P_i_k)
        
        L_k, L_prop_k, L_tran_k, L_du_k, L_cu_k = self.calculate_latency(UE_infor, phi_i_k, R_k)
        
        cpu_DU_req, cpu_CU_req = self.compute_cpu_request(UE_infor, R_k)
        
        feasible, reason_feasible = self.check_feasible(UE_infor, phi_i_k, phi_j_k, phi_m_k, num_P_i_k, num_z_i_k, R_k, SNR_db, L_k, cpu_DU_req, cpu_CU_req)
        
        if not feasible:
            self.reject_UE(UE_k, reason_feasible)
            self.episode_done = self.UE_manager.check_UE_all_inactive()

            return self.get_state(), 0.0, self.episode_done, {
                "pi_k": 0,
                "reason": reason_feasible,
                "UE_k": UE_k,
            }

        allocation = {
            "phi_i_k": phi_i_k,
            "phi_j_k": phi_j_k,
            "phi_m_k": phi_m_k,
            "num_z_i_k": num_z_i_k,
            "num_P_i_k": num_P_i_k,
            "R_k": float(R_k),
            "SNR_linear": float(SNR_linear),
            "SNR_db": float(SNR_db),
            "L_k": float(L_k),
            "latency_components": {
                "L_prop_k": float(L_prop_k),
                "L_tran_k": float(L_tran_k),
                "L_du_k": float(L_du_k),
                "L_cu_k": float(L_cu_k),
            },
            "cpu_DU_req": float(cpu_DU_req),
            "cpu_CU_req": float(cpu_CU_req),
        }
        
        reward = self.compute_reward(UE_infor, R_k, L_k)
        self.accept_UE(UE_k, allocation, reward)
        self.episode_done = self.UE_manager.check_UE_all_inactive()
        
        info = {
            "pi_k": 1,
            "reason": "accepted",
            "UE_k": UE_k,
            **copy.deepcopy(allocation),
        }

        return self.get_state(), float(reward), self.episode_done, info
