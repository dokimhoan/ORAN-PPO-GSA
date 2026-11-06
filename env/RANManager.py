"""Quản lý topology RAN tĩnh và tài nguyên tại các nút vô tuyến.

Các link truyền dẫn chỉ là ràng buộc kết nối dạng nhị phân. Link không có
dung lượng băng thông và cũng không có trạng thái băng thông còn lại.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from config import (A_j_random_list, A_m_random_list, P_i_random_list, network_size_m)
class RANManager:
    """Tạo một snapshot RAN và quản lý tài nguyên hữu hạn tại các nút."""

    def __init__(self, num_RBs, num_RUs, num_DUs, num_CUs):
        
        # Kích thước mạng
        self.num_RBs = int(num_RBs)
        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)

        # Danh sách capacity dùng để sinh topology
        self.P_i_random_list = P_i_random_list
        self.A_j_random_list = A_j_random_list
        self.A_m_random_list = A_m_random_list

        # -------------------- CPU model --------------------
        G = self.create_topo()

        self.P_i, self.A_j, self.A_m = self.get_node_cap(G)
        self.l_ru_du, self.l_du_cu = self.get_links(G)

        self.coordinate_RUs = self.set_coordinate_RUs()
        
        self.state_RAN = {}
        
        self.reset()
        
    def reset(self):
        """Khôi phục trạng thái chưa phân bổ mà không thay đổi topology."""
        self.RB_rem = self.num_RBs
        self.P_i_rem = self.P_i.copy()
        self.A_j_rem = self.A_j.copy()
        self.A_m_rem = self.A_m.copy()
            
    def get_state_RAN(self):
        self.state_RAN = {
            "num_RBs": self.num_RBs, 
            "num_RUs": self.num_RUs,
            "num_DUs": self.num_DUs, 
            "num_CUs": self.num_CUs,
                    
            "RB_rem": int(self.RB_rem),
            
            "coordinate_RUs": np.copy(self.coordinate_RUs),
                        
            "P_i_rem": np.copy(self.P_i_rem),
            "A_j_rem": np.copy(self.A_j_rem),
            "A_m_rem": np.copy(self.A_m_rem),
                        
            "l_ru_du":  np.copy(self.l_ru_du),
            "l_du_cu":  np.copy(self.l_du_cu),
            
            "max_P_i": float(max(P_i_random_list)),
            "max_A_j": float(max(A_j_random_list)),
            "max_A_m": float(max(A_m_random_list)),   
        }
            
    def set_coordinate_RUs(self):
        if self.num_RUs == 1:
            return [(0.0, 0.0)]
        circle_RU_out = float(network_size_m/2) * 0.5
        angles = np.linspace(0.0, 2 * np.pi, self.num_RUs - 1, endpoint=False) 
        x = np.concatenate(([0.0], circle_RU_out * np.cos(angles)))  
        y = np.concatenate(([0.0], circle_RU_out * np.sin(angles)))  
        coordinates_RUs = np.column_stack((x, y))
        return coordinates_RUs

    def create_topo(self):
        link_prob = 0.65
        while True:
            G = nx.Graph()

            # Tạo danh sách node
            RUs = [f'RU{i+1}' for i in range(self.num_RUs)]
            DUs = [f'DU{i+1}' for i in range(self.num_DUs)]
            CUs = [f'CU{i+1}' for i in range(self.num_CUs)]

            # Thêm node RU, DU, CU
            for ru in RUs:
                G.add_node(ru, type='RU', power=np.random.choice(P_i_random_list))
            for du in DUs:
                G.add_node(du, type='DU', capacity=np.random.choice(A_j_random_list))
            for cu in CUs:
                G.add_node(cu, type='CU', capacity=np.random.choice(A_m_random_list))

            # Random RU-DU links
            for ru in RUs:
                for du in DUs:
                    if np.random.rand() < link_prob: G.add_edge( ru, du, link_type="RU-DU")

            # Random DU-CU links
            for du in DUs:
                for cu in CUs:
                    if np.random.rand() < link_prob:
                        G.add_edge(du, cu, link_type="DU-CU")

            # Giữ topology nếu mạng liên thông
            if nx.is_connected(G):
                return G
    
    def get_node_cap(self, G):
        """Đọc power/CPU capacities từ graph."""
        ru_cap = []
        du_cap = []
        cu_cap = []
    
        for _, data in G.nodes(data=True):
            if data["type"] == "RU":
                ru_cap.append(float(data["power"]))
            elif data["type"] == "DU":
                du_cap.append(float(data["capacity"]))
            elif data["type"] == "CU":
                cu_cap.append(float(data["capacity"]))
    
        return np.asarray(ru_cap, dtype=float), np.asarray(du_cap, dtype=float), np.asarray(cu_cap, dtype=float)
    
    def get_links(self, G):
        """Chuyển bandwidth trên graph thành hai ma trận capacity."""
        RUs = [n for n, d in G.nodes(data=True) if d["type"] == "RU"]
        DUs = [n for n, d in G.nodes(data=True) if d["type"] == "DU"]
        CUs = [n for n, d in G.nodes(data=True) if d["type"] == "CU"]
    
        ru_index = {name: idx for idx, name in enumerate(RUs)}
        du_index = {name: idx for idx, name in enumerate(DUs)}
        cu_index = {name: idx for idx, name in enumerate(CUs)}
    
        l_ru_du = np.zeros((len(RUs), len(DUs)), dtype=float)
        l_du_cu = np.zeros((len(DUs), len(CUs)), dtype=float)
    
        for u, v, data in G.edges(data=True):
            type_u = G.nodes[u]["type"]
            type_v = G.nodes[v]["type"]
    
            if type_u == "RU" and type_v == "DU":
                l_ru_du[ru_index[u], du_index[v]] = 1
            elif type_u == "DU" and type_v == "RU":
                l_ru_du[ru_index[v], du_index[u]] = 1
            elif type_u == "DU" and type_v == "CU":
                l_du_cu[du_index[u], cu_index[v]] = 1
            elif type_u == "CU" and type_v == "DU":
                l_du_cu[du_index[v], cu_index[u]] = 1
    
        return l_ru_du, l_du_cu
        
    def update_resources(self, phi_i_k, phi_j_k, phi_m_k, num_z_i_k, num_P_i_k, cpu_DU_req, cpu_CU_req):
        ru, du, cu = int(phi_i_k), int(phi_j_k), int(phi_m_k)
        self.RB_rem -= int(num_z_i_k)
        self.P_i_rem[ru] -= float(num_P_i_k)
        self.A_j_rem[du] -= float(cpu_DU_req)
        self.A_m_rem[cu] -= float(cpu_CU_req)
        
        self.P_i_rem[np.abs(self.P_i_rem) < 1e-9] = 0.0
        self.A_j_rem[np.abs(self.A_j_rem) < 1e-6] = 0.0
        self.A_m_rem[np.abs(self.A_m_rem) < 1e-6] = 0.0

    def release_resources(self, phi_i_k, phi_j_k, phi_m_k, num_z_i_k, num_P_i_k, cpu_DU_req, cpu_CU_req):
        ru, du, cu = int(phi_i_k), int(phi_j_k), int(phi_m_k)
        self.RB_rem = min(self.num_RBs, self.RB_rem + int(num_z_i_k))
        self.P_i_rem[ru] = min(self.P_i[ru], self.P_i_rem[ru] + float(num_P_i_k))
        self.A_j_rem[du] = min(self.A_j[du], self.A_j_rem[du] + float(cpu_DU_req))
        self.A_m_rem[cu] = min(self.A_m[cu], self.A_m_rem[cu] + float(cpu_CU_req))
