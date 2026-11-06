from __future__ import annotations

import numpy as np
import cvxpy as cp
import warnings


SOLVER = cp.MOSEK
# ============================================================
# MOSEK parameters
# ============================================================
LONG_MOSEK_PARAMS = {
    "MSK_DPAR_MIO_MAX_TIME": 120.0,
    "MSK_DPAR_MIO_TOL_REL_GAP": 5e-2,
    "MSK_IPAR_NUM_THREADS": 4,
    "MSK_IPAR_LOG": 0,
}

SHORT_MOSEK_PARAMS = {
    "MSK_DPAR_OPTIMIZER_MAX_TIME": 50.0,

    "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",

    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,

    "MSK_IPAR_NUM_THREADS": 1,
    "MSK_IPAR_LOG": 0,
}


def Long_Term(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, R_min_k, L_max_k, packet_size_bits_k, lambda_k, P_i, A_j, A_m, k_DU, k_CU, eta_k, l_ru_du, l_du_cu, W_b, g_ib_k, d_i_k, w_acc, w_thr, w_lat, w_acc_k, w_thr_k, w_lat_k):
    
    try:
        epsilon = 1e-6
        light_speed = 3e8

        # -------------------- Scale --------------------
        R_scale = 1e6   # bps -> Mbps
        L_scale = 1e3   # s -> ms

        R_min_mbps_k = np.asarray(R_min_k, dtype=float) / R_scale
        L_max_ms_k = np.asarray(L_max_k, dtype=float) * L_scale
        A_j_scaled = np.asarray(A_j, dtype=float) / R_scale
        A_m_scaled = np.asarray(A_m, dtype=float) / R_scale

        # -------------------- Công suất cố định mỗi RB --------------------
        long_P_ib_k = float(np.max(P_i)) / num_RBs

        # -------------------- Data rate [Mbps] --------------------
        rate_coeff_ib_k = (W_b / R_scale) * np.log2(1.0 + g_ib_k * long_P_ib_k)
        R_max_k = np.max(np.sum(rate_coeff_ib_k, axis=1), axis=0)

        # -------------------- Biến quyết định --------------------
        long_z_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(num_UEs):
                    long_z_ib_k[i, b, k] = cp.Variable(boolean=True, name=f"long_z({i},{b},{k})")

        long_phi_i_k = np.empty((num_RUs, num_UEs), dtype=object)
        long_phi_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        long_phi_m_k = np.empty((num_CUs, num_UEs), dtype=object)
        long_T_j_k = np.empty((num_DUs, num_UEs), dtype=object)
        long_T_m_k = np.empty((num_CUs, num_UEs), dtype=object)

        for k in range(num_UEs):
            for i in range(num_RUs):
                long_phi_i_k[i, k] = cp.Variable(boolean=True, name=f"long_phi_i({i},{k})")
            for j in range(num_DUs):
                long_phi_j_k[j, k] = cp.Variable(boolean=True, name=f"long_phi_j({j},{k})")
                long_T_j_k[j, k] = cp.Variable(nonneg=True, name=f"long_T_j({j},{k})")
            for m in range(num_CUs):
                long_phi_m_k[m, k] = cp.Variable(boolean=True, name=f"long_phi_m({m},{k})")
                long_T_m_k[m, k] = cp.Variable(nonneg=True, name=f"long_T_m({m},{k})")

        long_R_k = cp.Variable(num_UEs, nonneg=True, name="long_R_k")   # Mbps
        long_L_k = cp.Variable(num_UEs, nonneg=True, name="long_L_k")   # ms
        long_pi_k = cp.Variable(num_UEs, boolean=True, name="long_pi_k")

        constraints = []

        # -------------------- Mỗi RB chỉ gán tối đa một UE --------------------
        for b in range(num_RBs):
            constraints.append(cp.sum([long_z_ib_k[i, b, k] for i in range(num_RUs) for k in range(num_UEs)]) <= 1)

        # -------------------- QoS rate --------------------
        for k in range(num_UEs):
            R_k = cp.sum([
                rate_coeff_ib_k[i, b, k] * long_z_ib_k[i, b, k]
                for i in range(num_RUs)
                for b in range(num_RBs)
            ])

            constraints.append(long_R_k[k] == R_k)
            constraints.append(long_R_k[k] >= R_min_mbps_k[k] * long_pi_k[k])

        long_total_R_k = cp.sum(long_R_k)

        # -------------------- Công suất mỗi RU --------------------
        for i in range(num_RUs):
            total_P_i = cp.sum([
                long_P_ib_k * long_z_ib_k[i, b, k]
                for b in range(num_RBs)
                for k in range(num_UEs)
            ])
            constraints.append(total_P_i <= P_i[i])

        # -------------------- Big-M: T = phi * R --------------------
        for k in range(num_UEs):
            for j in range(num_DUs):
                constraints.append(long_T_j_k[j, k] <= long_R_k[k])
                constraints.append(long_T_j_k[j, k] <= R_max_k[k] * long_phi_j_k[j, k])
                constraints.append(long_T_j_k[j, k] >= long_R_k[k] - R_max_k[k] * (1 - long_phi_j_k[j, k]))

            for m in range(num_CUs):
                constraints.append(long_T_m_k[m, k] <= long_R_k[k])
                constraints.append(long_T_m_k[m, k] <= R_max_k[k] * long_phi_m_k[m, k])
                constraints.append(long_T_m_k[m, k] >= long_R_k[k] - R_max_k[k] * (1 - long_phi_m_k[m, k]))

        # -------------------- Tài nguyên DU / CU --------------------
        for j in range(num_DUs):
            total_du_j = cp.sum([
                k_DU * long_T_j_k[j, k] * (1 + eta_k[k])
                for k in range(num_UEs)
            ])
            constraints.append(total_du_j <= A_j_scaled[j])

        for m in range(num_CUs):
            total_cu_m = cp.sum([
                k_CU * long_T_m_k[m, k] * (1 + eta_k[k])
                for k in range(num_UEs)
            ])
            constraints.append(total_cu_m <= A_m_scaled[m])

        # -------------------- Ánh xạ RU-DU-CU --------------------
        for k in range(num_UEs):
            constraints.append(cp.sum([long_phi_i_k[i, k] for i in range(num_RUs)]) == long_pi_k[k])
            constraints.append(cp.sum([long_phi_j_k[j, k] for j in range(num_DUs)]) == long_pi_k[k])
            constraints.append(cp.sum([long_phi_m_k[m, k] for m in range(num_CUs)]) == long_pi_k[k])

        # -------------------- z_ib_k <-> phi_i_k --------------------
        for i in range(num_RUs):
            for k in range(num_UEs):
                sum_z_i_k = cp.sum([long_z_ib_k[i, b, k] for b in range(num_RBs)])
                constraints.append(sum_z_i_k <= num_RBs * long_phi_i_k[i, k])
                constraints.append(sum_z_i_k >= long_phi_i_k[i, k])

        # -------------------- Liên kết topology --------------------
        for k in range(num_UEs):
            for i in range(num_RUs):
                for j in range(num_DUs):
                    constraints.append(long_phi_j_k[j, k] <= l_ru_du[i, j] - long_phi_i_k[i, k] + 1)

            for j in range(num_DUs):
                for m in range(num_CUs):
                    constraints.append(long_phi_m_k[m, k] <= l_du_cu[j, m] - long_phi_j_k[j, k] + 1)

        # -------------------- Latency [ms] --------------------
        for k in range(num_UEs):

            # Propagation delay [ms]
            L_prop_k = cp.sum([
                (d_i_k[i, k] / light_speed) * L_scale * long_phi_i_k[i, k]
                for i in range(num_RUs)
            ])

            # Transmission delay [ms]
            L_tran_k = (packet_size_bits_k[k] / (R_min_mbps_k[k] * 1000.0)) * long_pi_k[k]

            # Service rate [packet/s]
            mu_min_k = R_min_mbps_k[k] * R_scale / packet_size_bits_k[k]

            if mu_min_k <= lambda_k[k]:
                raise ValueError(f"UE {k}: M/M/1 không ổn định tại R_min.")

            # DU / CU delay [ms]
            L_du_k = cp.sum([
                (L_scale / (mu_min_k - lambda_k[k])) * long_phi_j_k[j, k]
                for j in range(num_DUs)
            ])

            L_cu_k = cp.sum([
                (L_scale / (mu_min_k - lambda_k[k])) * long_phi_m_k[m, k]
                for m in range(num_CUs)
            ])

            L_k = L_prop_k + L_tran_k + L_du_k + L_cu_k

            constraints.append(long_L_k[k] == L_k)
            constraints.append(long_L_k[k] <= L_max_ms_k[k] * long_pi_k[k])

        # -------------------- Hàm mục tiêu --------------------
        acc_term = 0
        thr_term = 0
        lat_pen = 0

        for k in range(num_UEs):
            acc_term += w_acc_k[k] * long_pi_k[k]
            thr_term += w_thr_k[k] * long_R_k[k] / max(R_min_mbps_k[k], epsilon)
            lat_pen += w_lat_k[k] * long_L_k[k] / max(L_max_ms_k[k], epsilon)

        objective = cp.Maximize(w_acc * acc_term + w_thr * thr_term - w_lat * lat_pen)

        # -------------------- Giải bài toán --------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=SOLVER, 
                      warm_start=True,
                      verbose=False, 
                      mosek_params=LONG_MOSEK_PARAMS)

        print("\n================ SOLVER INFO ================")
        print("Status       =", problem.status)
        print("Objective    =", problem.value)
        print("Solve time   =", problem.solver_stats.solve_time)
        print("Num iters    =", problem.solver_stats.num_iters)
        print("=============================================\n")

        # -------------------- Kiểm tra nghiệm --------------------
        if problem.status != cp.OPTIMAL:
            print(f"Long_Term solver status không hợp lệ: {problem.status}")
            return (None,) * 13

        if problem.value is None or not np.isfinite(problem.value):
            print(f"Long_Term objective không hợp lệ: {problem.value}")
            return (None,) * 13

        # -------------------- Kết quả --------------------
        long_objective = float(problem.value)
        long_total_pi_k = float(np.sum(long_pi_k.value))
        long_total_z_ib_k = float(np.sum([
            long_z_ib_k[i, b, k].value
            for i in range(num_RUs)
            for b in range(num_RBs)
            for k in range(num_UEs)
        ]))

        long_total_P_ib_k = long_total_z_ib_k * long_P_ib_k

        # Mbps -> bps
        long_total_R_value = float(long_total_R_k.value) * R_scale

        # ms -> s
        long_L_value_k = np.asarray(long_L_k.value, dtype=float) / L_scale
        long_total_delay = float(np.sum(long_L_value_k))

        print("✅ Long_Term solved successfully.")


        return long_pi_k,long_z_ib_k, long_P_ib_k, long_phi_i_k, long_phi_j_k, long_phi_m_k, long_objective, long_total_pi_k, long_total_R_value, long_L_value_k, long_total_delay, long_total_z_ib_k, long_total_P_ib_k

    except cp.SolverError as e:
        print(f"Long_Term solver error: {e}")
        return (None,) * 13

    except Exception as e:
        print(f"Error in Long_Term solver: {e}")
        return (None,) * 13

    
def Short_Term(num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, R_min_k, L_max_k, packet_size_bits_k, lambda_k, P_i, A_j, A_m, k_DU, k_CU, eta_k, l_ru_du, l_du_cu, W_b, g_ib_k, d_i_k, w_acc, w_thr, w_lat, w_acc_k, w_thr_k, w_lat_k, arr_long_pi_k, arr_long_z_ib_k, arr_long_phi_i_k, arr_long_phi_j_k, arr_long_phi_m_k):
    try:
        epsilon = 1e-6
        inv_log2 = 1.0 / np.log(2.0)
        light_speed = 3e8

        # -------------------- Scale --------------------
        R_scale = 1e6
        L_scale = 1e3

        R_min_mbps_k = np.asarray(R_min_k, dtype=float) / R_scale
        L_max_ms_k = np.asarray(L_max_k, dtype=float) * L_scale
        W_b_mhz = float(W_b) / R_scale

        # -------------------- Cố định quyết định Long-Term --------------------
        short_pi_k = np.rint(np.asarray(arr_long_pi_k, dtype=float)).astype(int)
        short_z_ib_k = np.rint(np.asarray(arr_long_z_ib_k, dtype=float)).astype(int)
        short_phi_i_k = np.rint(np.asarray(arr_long_phi_i_k, dtype=float)).astype(int)
        short_phi_j_k = np.rint(np.asarray(arr_long_phi_j_k, dtype=float)).astype(int)
        short_phi_m_k = np.rint(np.asarray(arr_long_phi_m_k, dtype=float)).astype(int)

        # -------------------- Biến công suất tại z = 1 --------------------
        short_p_ib_k = np.empty((num_RUs, num_RBs, num_UEs), dtype=object)

        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(num_UEs):
                    short_p_ib_k[i, b, k] = (
                        cp.Variable(nonneg=True, name=f"short_p({i},{b},{k})")
                        if short_z_ib_k[i, b, k] == 1 else 0.0
                    )

        constraints = []

        # -------------------- Shannon capacity và QoS rate --------------------
        short_total_R_k = 0
        short_R_k = np.empty(num_UEs, dtype=object)

        for k in range(num_UEs):
            rate_terms = [
                W_b_mhz * cp.log1p(g_ib_k[i, b, k] * short_p_ib_k[i, b, k]) * inv_log2
                for i in range(num_RUs)
                for b in range(num_RBs)
                if short_z_ib_k[i, b, k] == 1
            ]

            if short_pi_k[k] == 1:
                if not rate_terms:
                    raise ValueError(f"UE {k} được accept nhưng không có RB.")

                R_k = cp.sum(rate_terms)
                constraints.append(R_k >= R_min_mbps_k[k])

            else:
                if rate_terms:
                    raise ValueError(f"UE {k} bị reject nhưng vẫn có RB.")
                R_k = 0.0

            short_R_k[k] = R_k
            short_total_R_k += R_k

        # -------------------- Tổng công suất của từng RU --------------------
        for i in range(num_RUs):
            power_terms = [
                short_p_ib_k[i, b, k]
                for b in range(num_RBs)
                for k in range(num_UEs)
                if short_z_ib_k[i, b, k] == 1
            ]

            if power_terms:
                constraints.append(cp.sum(power_terms) <= P_i[i])

        # -------------------- Latency theo rate Short-Term --------------------
        short_total_L_k = 0
        short_L_k = np.empty(num_UEs, dtype=object)

        for k in range(num_UEs):
            if short_pi_k[k] == 0:
                short_L_k[k] = 0.0
                continue

            # Propagation delay [ms]
            L_prop_k = float(np.sum((d_i_k[:, k] / light_speed) * short_phi_i_k[:, k])) * L_scale

            # Transmission delay [ms]
            L_tran_k = (packet_size_bits_k[k] / 1000.0) * cp.inv_pos(short_R_k[k])

            # Arrival traffic [Mbps]
            arrival_rate_mbps_k = lambda_k[k] * packet_size_bits_k[k] / R_scale
            rate_margin_k = short_R_k[k] - arrival_rate_mbps_k

            constraints.append(rate_margin_k >= epsilon)

            # DU delay [ms]
            L_du_k = cp.sum([
                (packet_size_bits_k[k] / 1000.0) * cp.inv_pos(rate_margin_k)
                for j in range(num_DUs)
                if short_phi_j_k[j, k] == 1
            ])

            # CU delay [ms]
            L_cu_k = cp.sum([
                (packet_size_bits_k[k] / 1000.0) * cp.inv_pos(rate_margin_k)
                for m in range(num_CUs)
                if short_phi_m_k[m, k] == 1
            ])

            L_k = L_prop_k + L_tran_k + L_du_k + L_cu_k

            short_L_k[k] = L_k
            short_total_L_k += L_k

            constraints.append(L_k <= L_max_ms_k[k])

        # -------------------- Hàm mục tiêu --------------------
        acc_term = 0
        thr_term = 0
        lat_pen = 0

        for k in range(num_UEs):
            acc_term += w_acc_k[k] * short_pi_k[k]
            thr_term += w_thr_k[k] * short_R_k[k] / max(R_min_mbps_k[k], epsilon)
            lat_pen += w_lat_k[k] * short_L_k[k] / max(L_max_ms_k[k], epsilon)

        objective = cp.Maximize(w_acc * acc_term + w_thr * thr_term - w_lat * lat_pen)

        # -------------------- Giải bài toán --------------------
        problem = cp.Problem(objective, constraints)

        if not problem.is_dcp():
            print("❌ Short-Term không tuân theo DCP.")
            return (None,) * 13

        problem.solve(solver=SOLVER,
                      warm_start=True,
                      verbose=False, 
                      mosek_params=SHORT_MOSEK_PARAMS)

        print("\n================ SOLVER INFO ================")
        print("Status       =", problem.status)
        print("Objective    =", problem.value)
        print("Solve time   =", problem.solver_stats.solve_time)
        print("Num iters    =", problem.solver_stats.num_iters)
        print("=============================================\n")

        # -------------------- Kiểm tra nghiệm --------------------
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            print(f"Short-Term solver status không hợp lệ: {problem.status}")
            return (None,) * 13

        if problem.value is None or not np.isfinite(problem.value):
            print(f"Short-Term objective không hợp lệ: {problem.value}")
            return (None,) * 13

        # -------------------- Kết quả --------------------
        short_objective = float(problem.value)
        short_total_pi_k = float(np.sum(short_pi_k))
        short_total_z_ib_k = float(np.sum(short_z_ib_k))

        short_total_P_ib_k = float(np.sum([
            float(short_p_ib_k[i, b, k].value)
            for i in range(num_RUs)
            for b in range(num_RBs)
            for k in range(num_UEs)
            if short_z_ib_k[i, b, k] == 1
        ]))

        # Rate từng UE [Mbps]
        short_R_mbps_value_k = np.asarray([
            float(short_R_k[k].value) if hasattr(short_R_k[k], "value") else float(short_R_k[k])
            for k in range(num_UEs)
        ])

        # Tổng rate: Mbps -> bps
        short_total_R_value = float(np.sum(short_R_mbps_value_k)) * R_scale

        # Latency từng UE [ms]
        short_L_ms_value_k = np.asarray([
            float(short_L_k[k].value) if hasattr(short_L_k[k], "value") else float(short_L_k[k])
            for k in range(num_UEs)
        ])

        # ms -> second
        short_L_value_k = short_L_ms_value_k / L_scale
        short_total_delay = float(np.sum(short_L_value_k))

        print("✅ Short-Term solved successfully.")

        return (
            short_pi_k,
            short_z_ib_k,
            short_p_ib_k,
            short_phi_i_k,
            short_phi_j_k,
            short_phi_m_k,
            short_objective,
            short_total_pi_k,
            short_total_R_value,
            short_L_value_k,
            short_total_delay,
            short_total_z_ib_k,
            short_total_P_ib_k
        )

    except cp.SolverError as e:
        print(f"Short-Term solver error: {e}")
        return (None,) * 13

    except Exception as e:
        print(f"Error in Short-Term solver: {e}")
        return (None,) * 13