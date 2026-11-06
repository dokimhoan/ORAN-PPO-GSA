import numpy as np
import cvxpy as cp
import time
import traceback as tb
import random

SOLVER = cp.MOSEK

MOSEK_PARAMS = {
    "MSK_DPAR_MIO_MAX_TIME": 900.0,
    "MSK_DPAR_MIO_TOL_REL_GAP": 5e-2,
    "MSK_IPAR_NUM_THREADS": 0,
    "MSK_IPAR_LOG": 0,
}

def global_solving(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, R_min_k, L_max_k, packet_size_bits_k, lambda_k, P_i, A_j, A_m, k_DU, k_CU, eta_k, l_ru_du, l_du_cu, W_b, g_ib_k, d_i_k, w_acc, w_thr, w_lat, w_acc_k, w_thr_k, w_lat_k, P_ib_k_val):

    try:
        epsilon = 1e-4
        inv_log2 = 1.0 / np.log(2.0)
        speed_of_light_mps = 3e8
        
        # ============================================================
        # Tính trước rate cho mỗi power level
        # r_i,b,k,l = W_b log2(1 + g_i,b,k P_l)
        # ============================================================
        rate_coeff_ibkl = (W_b * np.log2(1.0 + g_ib_k[:, :, :, None] * power_levels[None, None, None, :]))

        # --- create variables (kept as object arrays for minimal change) ---
        z_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)
        p_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)
        mu_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(num_UEs):
                    z_ib_k[i, b, k] = cp.Variable(boolean=True, name=f"z_ib_k({i}, {b}, {k})")
                    p_ib_k[i, b, k] = cp.Variable(nonneg=True, name=f"p_ib_k({i}, {b}, {k})")
                    mu_ib_k[i, b, k] = cp.Variable(nonneg=True, name=f"mu_ib_k({i}, {b}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_k, phi_j_k, phi_m_k
        phi_i_k = np.empty((num_RUs, num_UEs), dtype=object)
        phi_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        phi_m_k = np.empty((num_CUs, num_UEs), dtype=object)
        
        # T_j_k = phi_j_k * R_k và T_m_k = phi_m_k * R_k.
        # T là biến liên tục không âm vì R_k là biến liên tục.
        T_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        T_m_k = np.empty((num_CUs, num_UEs), dtype=object)

        for k in range(num_UEs):
            for i in range(num_RUs):
                phi_i_k[i, k] = cp.Variable(boolean=True, name=f"phi_i_k({i},{k})")
            for j in range(num_DUs):
                phi_j_k[j, k] = cp.Variable(boolean=True, name=f"phi_j_k({j},{k})")
                T_j_k[j, k] = cp.Variable(nonneg=True, name=f"T_j_k({j},{k})")
            for m in range(num_CUs):
                phi_m_k[m, k] = cp.Variable(boolean=True, name=f"phi_m_k({m},{k})")
                T_m_k[m, k] = cp.Variable(nonneg=True, name=f"T_m_k({m},{k})")
                
        R_k = cp.Variable(num_UEs, nonneg=True, name="R_k")
        L_k = cp.Variable(num_UEs, nonneg=True, name="L_k")
        
        # Biến tối ưu cho việc phân bổ
        pi_k = cp.Variable(num_UEs, boolean=True, name="pi_k") 
        
        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_k[i, b, k] for k in range(num_UEs) for i in range(num_RUs)]) <= 1) 

        # Ràng buộc: Đảm bảo QoS (Data rate)
        for k in range(num_UEs):
            R_term = cp.sum([W_b * cp.log1p(g_ib_k[i, b, k] * mu_ib_k[i, b, k]) * inv_log2 for i in range(num_RUs) for b in range(num_RBs)])
            
            constraints.append(R_k[k] == R_term)
            constraints.append(R_k[k] >= R_min_k[k] * pi_k[k])
        total_R_k = cp.sum(R_k)

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_P_i = cp.sum([mu_ib_k[i, b, k] 
                                  for b in range(num_RBs) 
                                  for k in range(num_UEs) ])
            constraints.append(total_P_i <= P_i[i])
            
        # -------------------- Big-M cho T_j = phi_j * R và T_m = phi_m * R --------------------
        for k in range(num_UEs):
            for j in range(num_DUs):
                constraints.append(T_j_k[j, k] <= R_k[k])
                constraints.append(T_j_k[j, k] <= R_max_k[k] * phi_j_k[j, k])
                constraints.append(T_j_k[j, k] >= R_k[k] - R_max_k[k] * (1 - phi_j_k[j, k]))
            for m in range(num_CUs):
                constraints.append(T_m_k[m, k] <= R_k[k])
                constraints.append(T_m_k[m, k] <= R_max_k[k] * phi_m_k[m, k])
                constraints.append(T_m_k[m, k] >= R_k[k] - R_max_k[k] * (1 - phi_m_k[m, k]))

        # -------------------- Tài nguyên DU / CU --------------------
        for j in range(num_DUs):
            total_du_j = cp.sum([k_DU * T_j_k[j, k] * (1 + eta_k[k]) for k in range(num_UEs)])
            constraints.append(total_du_j <= A_j[j])

        for m in range(num_CUs):
            total_cu_m = cp.sum([k_CU * T_m_k[m, k] * (1 + eta_k[k])
                               for k in range(num_UEs)])
            constraints.append(total_cu_m <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for k in range(num_UEs):
            constraints.append(cp.sum([phi_i_k[i, k] for i in range(num_RUs)]) == pi_k[k])
            constraints.append(cp.sum([phi_j_k[j, k] for j in range(num_DUs)]) == pi_k[k])
            constraints.append(cp.sum([phi_m_k[m, k] for m in range(num_CUs)]) == pi_k[k])

        # Ràng buộc: Chuyển đổi z_ib_k sang phi_i_k
        for i in range(num_RUs):
            for k in range(num_UEs):
                avg_z = (1 / num_RBs) * cp.sum([z_ib_k[i, b, k] for b in range(num_RBs)])
                constraints.append(avg_z <= phi_i_k[i, k])
                constraints.append(phi_i_k[i, k] <= avg_z + (1 - epsilon))

        # -------------------- Liên kết topo RU–DU–CU --------------------
        for k in range(num_UEs):
            for i in range(num_RUs):
                for j in range(num_DUs):
                    constraints.append(phi_j_k[j, k] <= l_ru_du[i, j] - phi_i_k[i, k] + 1)
                        
            for j in range(num_DUs):
                for m in range(num_CUs):
                    constraints.append(phi_m_k[m, k] <= l_du_cu[j, m] - phi_j_k[j, k] + 1)


        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_k = z_ib_k * p_ib_k)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(num_UEs):
                    constraints.append(mu_ib_k[i, b, k] <= W_b[i] * z_ib_k[i, b, k])  
                    constraints.append(mu_ib_k[i, b, k] >= p_ib_k[i, b, k] - W_b[i] * (1 - z_ib_k[i, b, k]))
                    constraints.append(mu_ib_k[i, b, k] <= p_ib_k[i, b, k])  

        
        # ======= TÍNH DELAY (đúng shapes mô hình latency) =======
        for k in range(num_UEs):

            # -------------------- Propagation delay --------------------
            L_prop_k = cp.sum([(d_i_k[i, k] / speed_of_light_mps) * phi_i_k[i, k] for i in range(num_RUs)])

            # -------------------- Transmission delay --------------------
            L_tran_k = (packet_size_bits_k[k]/ R_k[k]) * pi_k[k]
            
            mu_k = (R_k[k] / packet_size_bits_k[k])
            
            if mu_k <= lambda_k[k]:
                raise ValueError(f"UE {k}: M/M/1 không ổn định tại R_min.")
                        
            # -------------------- DU delay --------------------
            L_du_k = cp.sum([phi_j_k[j, k] / (mu_k - lambda_k[k]) for j in range(num_DUs)])

            # -------------------- CU delay --------------------
            L_cu_k = cp.sum([phi_m_k[m, k] / (mu_k - lambda_k[k]) for m in range(num_CUs)])
            
            L_k = L_prop_k + L_tran_k + L_du_k + L_cu_k
            
            constraints.append(L_k[k] == L_k)
            constraints.append(L_k[k] <= L_max_k[k] * pi_k[k])


        # -------------------- Tính toán các term --------------------
        acc_term = 0
        thr_term = 0
        lat_pen  = 0
        for k in range(num_UEs):
            acc_term += w_acc_k[k] * pi_k[k]
            thr_term += w_thr_k[k] * (R_k[k] / max(R_min_k[k], epsilon))
            lat_pen  += w_lat_k[k] * (L_k[k] / max(L_max_k[k], epsilon))
        
        # -------------------- Mục tiêu tối ưu --------------------
        objective = cp.Maximize(w_acc * acc_term  + w_thr * thr_term - w_lat * lat_pen)

        # -------------------- Giải bài toán --------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver = SOLVER, warm_start = True, mosek_params = MOSEK_PARAMS)
        print("\n================ SOLVER INFO ================")
        print("Status       =", problem.status)
        print("Objective    =", problem.value)
        print("Solve time   =", problem.solver_stats.solve_time)
        print("Num iters    =", problem.solver_stats.num_iters)
        print("=============================================\n")

        # -------------------- Kiểm tra nghiệm --------------------
        if problem.status != cp.OPTIMAL:
            print(f"Term solver status không hợp lệ: {problem.status}")
            return (None,) * 13

        if problem.value is None or not np.isfinite(problem.value):
            print(f"Term objective không hợp lệ: {problem.value}")
            return (None,) * 13

        # -------------------- Kết quả --------------------
        objective = float(problem.value)
        total_pi_k = float(cp.sum(pi_k).value)
        total_z_ib_k = float(cp.sum([z_ib_k[i, b, k] for i in range(num_RUs) for b in range(num_RBs) for k in range(num_UEs)]).value)

        total_R_k = float(total_R_k.value)
        total_delay = 0.0

        for k in range(num_UEs):
            if hasattr(L_k[k], "value"):
                total_delay += float(L_k[k].value)
            else:
                total_delay += float(L_k[k])

        print("✅ Term solved successfully.")

        return pi_k, z_ib_k, P_ib_k, phi_i_k, phi_j_k, phi_m_k, objective, total_pi_k, total_R_k, L_k, total_delay, total_z_ib_k, total_P_ib_k

    except cp.SolverError as e:
        print(f"Term solver error: {e}")
        return (None,) * 13

    except Exception as e:
        print(f"Error in Term solver: {e}")
        return (None,) * 13
    