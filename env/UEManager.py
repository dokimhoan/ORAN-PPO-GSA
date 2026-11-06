"""Sinh tập yêu cầu UE cố định cho một bài toán tối ưu tại một thời điểm."""

from __future__ import annotations

import copy
import numpy as np

from config import SLICE_PRESET, network_size_m


class UEManager:
    def __init__(self, num_UEs):
        self.network_size_m = float(network_size_m)
        
        self.num_UEs = int(num_UEs)
        
        self.SLICE_PRESET = copy.deepcopy(SLICE_PRESET)
        self.slice_names = list(self.SLICE_PRESET)
        
        self.coordinate_UEs = self.generate_coordinate_UEs()
        
        self.state_UEs_req = {}

    @staticmethod
    def empty_status():
        return {"active": 1, "served": False, "reason": None}

    @staticmethod
    def empty_alloc():
        return {
            "phi_i_k": None, "phi_j_k": None, "phi_m_k": None,
            "num_z_i_k": 0, "num_P_i_k": 0.0,
            "R_k": 0.0, "SNR_linear": 0.0,
            "SNR_db": -np.inf, "L_k": np.inf,
            "latency_components": {}, "cpu_DU_req": 0.0,
            "cpu_CU_req": 0.0,
        }

    def generate_coordinate_UEs(self):
        """Sinh vị trí UE."""
        half_size = self.network_size_m / 2.0
    
        x = np.random.uniform(-half_size, half_size, size=self.num_UEs)
        y = np.random.uniform(-half_size, half_size, size=self.num_UEs)
        
        coordinates = np.column_stack((x, y))
        
        return coordinates

    def get_state_UEs_req(self, coordinate_UEs, d_i_k, g_i_k):

        for ue_id in range(self.num_UEs):

            slice_name = np.random.choice(self.slice_names,p=[0.7, 0.3])

            self.state_UEs_req[ue_id] = {
                "id": ue_id,

                **copy.deepcopy(self.SLICE_PRESET[slice_name]),

                # Tọa độ của UE hiện tại
                "coordinate_UE": copy.deepcopy(coordinate_UEs[ue_id]),

                # Khoảng cách từ UE này tới tất cả RU
                "d_i_k": copy.deepcopy(d_i_k[:, ue_id]),

                # Gain từ UE này tới tất cả RU
                "g_i_k": copy.deepcopy(g_i_k[:, ue_id]),

                "status": self.empty_status(),
                "allocation": self.empty_alloc(),
                "reward": 0.0,
            }


    def clear_current_allocations(self):
        for ue in self.state_UEs_req.values():
            ue["status"] = self.empty_status()
            ue["allocation"] = self.empty_alloc()
            ue["reward"] = 0.0

    def update_UE_request(self, ue_id, update_dict):
        ue = self.state_UEs_req[int(ue_id)]
        for key, value in update_dict.items():
            if isinstance(value, dict) and isinstance(ue.get(key), dict):
                ue[key].update(copy.deepcopy(value))
            else:
                ue[key] = copy.deepcopy(value)

    def check_UE_all_inactive(self):
        if not self.state_UEs_req:
            return True
        return all(int(ue["status"].get("active", 0)) == 0 for ue in self.state_UEs_req.values())
