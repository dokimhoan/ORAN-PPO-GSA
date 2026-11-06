from __future__ import annotations

import numpy as np
import cvxpy as cp


SOLVER = cp.MOSEK

# ============================================================
# MOSEK parameters
# ============================================================
ROUNDROBIN_MOSEK_PARAMS = {
    "MSK_DPAR_MIO_MAX_TIME": 120.0,
    "MSK_DPAR_MIO_TOL_REL_GAP": 5e-2,
    "MSK_IPAR_NUM_THREADS": 4,
    "MSK_IPAR_LOG": 0,
}

def RoundRobin_RU(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, R_min_k, L_max_k, packet_size_bits_k, lambda_k, P_i, A_j, A_m, k_DU, k_CU, eta_k, l_ru_du, l_du_cu, W_b, g_ib_k, d_i_k, w_acc, w_thr, w_lat, w_acc_k, w_thr_k, w_lat_k, arr_roundrobin_phi_i_k):

    try:
        epsilon = 1e-6
        speed_of_light_mps = 3e8

        # -------------------- RU gần nhất cố định cho mỗi UE --------------------
        arr_roundrobin_phi_i_k = np.asarray(arr_roundrobin_phi_i_k, dtype=int)

        # -------------------- Công suất cố định mỗi RB --------------------
        roundrobin_P_ib_k = np.max(P_i) / num_RBs

        # -------------------- Tính trước data rate --------------------
        rate_coeff_ib_k = W_b * np.log2(1.0 + g_ib_k * roundrobin_P_ib_k)
        R_max_k = np.max(np.sum(rate_coeff_ib_k, axis=1), axis=0)

        # -------------------- Khởi tạo biến nhị phân --------------------
        roundrobin_z_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(num_UEs):
                    roundrobin_z_ib_k[i, b, k] = cp.Variable(boolean=True, name=f"roundrobin_z_ib_k({i},{b},{k})")

        roundrobin_phi_i_k = np.empty((num_RUs, num_UEs), dtype=object)
        roundrobin_phi_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        roundrobin_phi_m_k = np.empty((num_CUs, num_UEs), dtype=object)

        # T_j_k = phi_j_k * R_k và T_m_k = phi_m_k * R_k
        roundrobin_T_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        roundrobin_T_m_k = np.empty((num_CUs, num_UEs), dtype=object)

        for k in range(num_UEs):
            for i in range(num_RUs):
                roundrobin_phi_i_k[i, k] = cp.Variable(boolean=True, name=f"roundrobin_phi_i_k({i},{k})")
            for j in range(num_DUs):
                roundrobin_phi_j_k[j, k] = cp.Variable(boolean=True, name=f"roundrobin_phi_j_k({j},{k})")
                roundrobin_T_j_k[j, k] = cp.Variable(nonneg=True, name=f"roundrobin_T_j_k({j},{k})")
            for m in range(num_CUs):
                roundrobin_phi_m_k[m, k] = cp.Variable(boolean=True, name=f"roundrobin_phi_m_k({m},{k})")
                roundrobin_T_m_k[m, k] = cp.Variable(nonneg=True, name=f"roundrobin_T_m_k({m},{k})")

        roundrobin_R_k = cp.Variable(num_UEs, nonneg=True, name="roundrobin_R_k")
        roundrobin_L_k = cp.Variable(num_UEs, nonneg=True, name="roundrobin_L_k")
        roundrobin_pi_k = cp.Variable(num_UEs, boolean=True, name="roundrobin_pi_k")

        constraints = []

        # -------------------- Cố định RU gần nhất --------------------
        for i in range(num_RUs):
            for k in range(num_UEs):
                constraints.append(roundrobin_phi_i_k[i, k] <= arr_roundrobin_phi_i_k[i, k])

        # -------------------- Mỗi RB chỉ gán cho tối đa một UE --------------------
        for b in range(num_RBs):
            constraints.append(cp.sum([roundrobin_z_ib_k[i, b, k] for i in range(num_RUs) for k in range(num_UEs)]) <= 1)

        # -------------------- Ràng buộc QoS (Data rate) --------------------
        for k in range(num_UEs):
            R_k = cp.sum([rate_coeff_ib_k[i, b, k] * roundrobin_z_ib_k[i, b, k] for i in range(num_RUs) for b in range(num_RBs)])

            constraints.append(roundrobin_R_k[k] == R_k)
            constraints.append(roundrobin_R_k[k] >= R_min_k[k] * roundrobin_pi_k[k])

        roundrobin_total_R_k = cp.sum(roundrobin_R_k)

        # -------------------- Tổng công suất của mỗi RU --------------------
        for i in range(num_RUs):
            total_P_i = cp.sum([roundrobin_z_ib_k[i, b, k] * roundrobin_P_ib_k for b in range(num_RBs) for k in range(num_UEs)])
            
            constraints.append(total_P_i <= P_i[i])

        # -------------------- Big-M cho T_j = phi_j * R và T_m = phi_m * R --------------------
        for k in range(num_UEs):
            for j in range(num_DUs):
                constraints.append(roundrobin_T_j_k[j, k] <= roundrobin_R_k[k])
                constraints.append(roundrobin_T_j_k[j, k] <= R_max_k[k] * roundrobin_phi_j_k[j, k])
                constraints.append(roundrobin_T_j_k[j, k] >= roundrobin_R_k[k] - R_max_k[k] * (1 - roundrobin_phi_j_k[j, k]))

            for m in range(num_CUs):
                constraints.append(roundrobin_T_m_k[m, k] <= roundrobin_R_k[k])
                constraints.append(roundrobin_T_m_k[m, k] <= R_max_k[k] * roundrobin_phi_m_k[m, k])
                constraints.append(roundrobin_T_m_k[m, k] >= roundrobin_R_k[k] - R_max_k[k] * (1 - roundrobin_phi_m_k[m, k]))

        # -------------------- Tài nguyên DU / CU --------------------
        for j in range(num_DUs):
            total_du_j = cp.sum([k_DU * roundrobin_T_j_k[j, k] * (1 + eta_k[k]) for k in range(num_UEs)])
            constraints.append(total_du_j <= A_j[j])

        for m in range(num_CUs):
            total_cu_m = cp.sum([k_CU * roundrobin_T_m_k[m, k] * (1 + eta_k[k]) for k in range(num_UEs)])
            constraints.append(total_cu_m <= A_m[m])

        # -------------------- Ánh xạ RU–DU–CU --------------------
        # UE được accept phải chọn đúng 1 RU, 1 DU và 1 CU.
        for k in range(num_UEs):
            constraints.append(cp.sum([roundrobin_phi_i_k[i, k] for i in range(num_RUs)]) == roundrobin_pi_k[k])
            constraints.append(cp.sum([roundrobin_phi_j_k[j, k] for j in range(num_DUs)]) == roundrobin_pi_k[k])
            constraints.append(cp.sum([roundrobin_phi_m_k[m, k] for m in range(num_CUs)]) == roundrobin_pi_k[k])

        # -------------------- Chuyển đổi z_ib_k ↔ phi_i_k --------------------
        # phi_i = 1 khi và chỉ khi UE nhận ít nhất một RB từ RU i.
        for i in range(num_RUs):
            for k in range(num_UEs):
                avg_z = (1 / num_RBs) * cp.sum([roundrobin_z_ib_k[i, b, k] for b in range(num_RBs)])
                constraints.append(avg_z <= roundrobin_phi_i_k[i, k])
                constraints.append(roundrobin_phi_i_k[i, k] <= avg_z + (1 - epsilon))

        # -------------------- Liên kết topo RU–DU–CU --------------------
        for k in range(num_UEs):
            for i in range(num_RUs):
                for j in range(num_DUs):
                    constraints.append(roundrobin_phi_j_k[j, k] <= l_ru_du[i, j] - roundrobin_phi_i_k[i, k] + 1)

            for j in range(num_DUs):
                for m in range(num_CUs):
                    constraints.append(roundrobin_phi_m_k[m, k] <= l_du_cu[j, m] - roundrobin_phi_j_k[j, k] + 1)

        # ==================== TÍNH DELAY ====================
        for k in range(num_UEs):

            # -------------------- Propagation delay --------------------
            L_prop_k = cp.sum([(d_i_k[i, k] / speed_of_light_mps) * roundrobin_phi_i_k[i, k] for i in range(num_RUs)])

            # -------------------- Transmission delay --------------------
            L_tran_k = (packet_size_bits_k[k] / R_min_k[k]) * roundrobin_pi_k[k]

            # -------------------- Service rate tối thiểu --------------------
            mu_min_k = R_min_k[k] / packet_size_bits_k[k]

            if mu_min_k <= lambda_k[k]:
                raise ValueError(f"UE {k}: M/M/1 không ổn định tại R_min.")

            # -------------------- DU delay --------------------
            L_du_k = cp.sum([roundrobin_phi_j_k[j, k] / (mu_min_k - lambda_k[k]) for j in range(num_DUs)])

            # -------------------- CU delay --------------------
            L_cu_k = cp.sum([roundrobin_phi_m_k[m, k] / (mu_min_k - lambda_k[k]) for m in range(num_CUs)])

            L_k = L_prop_k + L_tran_k + L_du_k + L_cu_k
            constraints.append(roundrobin_L_k[k] == L_k)
            constraints.append(roundrobin_L_k[k] <= L_max_k[k] * roundrobin_pi_k[k])

        # -------------------- Tính toán các term --------------------
        acc_term = 0
        thr_term = 0
        lat_pen = 0

        for k in range(num_UEs):
            acc_term += w_acc_k[k] * roundrobin_pi_k[k]
            thr_term += w_thr_k[k] * (roundrobin_R_k[k] / max(R_min_k[k], epsilon))
            lat_pen += w_lat_k[k] * (roundrobin_L_k[k] / max(L_max_k[k], epsilon))

        # -------------------- Mục tiêu tối ưu --------------------
        objective = cp.Maximize(w_acc * acc_term + w_thr * thr_term - w_lat * lat_pen)

        # -------------------- Giải bài toán --------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=SOLVER,
                      warm_start=True,
                      verbose=False,  
                      mosek_params=ROUNDROBIN_MOSEK_PARAMS)

        print("\n================ SOLVER INFO ================")
        print("Status       =", problem.status)
        print("Objective    =", problem.value)
        print("Solve time   =", problem.solver_stats.solve_time)
        print("Num iters    =", problem.solver_stats.num_iters)
        print("=============================================\n")

        # -------------------- Kiểm tra nghiệm --------------------
        if problem.status != cp.OPTIMAL:
            print(f"RoundRobin-RU solver status không hợp lệ: {problem.status}")
            return (None,) * 13

        if problem.value is None or not np.isfinite(problem.value):
            print(f"RoundRobin-RU objective không hợp lệ: {problem.value}")
            return (None,) * 13

        # -------------------- Kết quả --------------------
        roundrobin_objective = float(problem.value)
        roundrobin_total_pi_k = float(cp.sum(roundrobin_pi_k).value)
        roundrobin_total_z_ib_k = float(cp.sum([roundrobin_z_ib_k[i, b, k] for i in range(num_RUs) for b in range(num_RBs) for k in range(num_UEs)]).value)
        roundrobin_total_P_ib_k = roundrobin_total_z_ib_k * roundrobin_P_ib_k
        roundrobin_total_R_k = float(roundrobin_total_R_k.value)
        roundrobin_total_delay = 0.0

        for k in range(num_UEs):
            if hasattr(roundrobin_L_k[k], "value"):
                roundrobin_total_delay += float(roundrobin_L_k[k].value)
            else:
                roundrobin_total_delay += float(roundrobin_L_k[k])

        print("✅ RoundRobin-RU solved successfully.")

        return roundrobin_pi_k, roundrobin_z_ib_k, roundrobin_P_ib_k, roundrobin_phi_i_k, roundrobin_phi_j_k, roundrobin_phi_m_k, roundrobin_objective, roundrobin_total_pi_k, roundrobin_total_R_k, roundrobin_L_k, roundrobin_total_delay, roundrobin_total_z_ib_k, roundrobin_total_P_ib_k

    except cp.SolverError as e:
        print(f"RoundRobin-RU solver error: {e}")
        return (None,) * 13

    except Exception as e:
        print(f"Error in RoundRobin-RU solver: {e}")
        return (None,) * 13
