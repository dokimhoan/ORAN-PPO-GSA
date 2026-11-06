import numpy as np


SLICE_NAMES = ["eMBB", "uRLLC"]


def state_to_MILP(state):

    RAN = state["RAN"]
    UEs = state["UE_requests"]

    # ============================================================
    # 1. Kích thước hệ thống
    # ============================================================
    num_slices = len(SLICE_NAMES)
    num_UEs = len(UEs)
    num_RUs = RAN["num_RUs"]
    num_DUs = RAN["num_DUs"]
    num_CUs = RAN["num_CUs"]
    num_RBs = RAN["num_RBs"]

    # ============================================================
    # 2. Tài nguyên hệ thống
    # ============================================================
    P_i = np.array(RAN["RU_remaining"], dtype=float)
    A_j = np.array(RAN["DU_remaining"], dtype=float)
    A_m = np.array(RAN["CU_remaining"], dtype=float)

    k_DU = float(RAN["k_DU"])
    k_CU = float(RAN["k_CU"])

    # ============================================================
    # 3. Topology
    # ============================================================
    l_ru_du = np.array(RAN["l_ru_du"], dtype=float)
    l_du_cu = np.array(RAN["l_du_cu"], dtype=float)

    # ============================================================
    # 4. Bandwidth mỗi RB
    # ============================================================
    W_b = float(RAN["bandwidth_per_RB"])

    # ============================================================
    # 5. Global weights
    # ============================================================
    w_acc = float(RAN["w_acc"])
    w_thr = float(RAN["w_thr"])
    w_lat = float(RAN["w_lat"])

    # ============================================================
    # 6. QoS / Traffic theo UE
    #
    # Không còn phụ thuộc trực tiếp vào chỉ số slice.
    #
    # Shape:
    #   R_min_k              : (K,)
    #   L_max_k              : (K,)
    #   packet_size_bits_k   : (K,)
    #   lambda_k             : (K,)
    #   eta_k                : (K,)
    # ============================================================
    R_min_k = np.zeros(num_UEs, dtype=float)
    L_max_k = np.zeros(num_UEs, dtype=float)

    packet_size_bits_k = np.zeros(num_UEs, dtype=float)
    lambda_k = np.zeros(num_UEs, dtype=float)
    eta_k = np.zeros(num_UEs, dtype=float)

    # ============================================================
    # 7. Slice mapping
    #
    # slice_mapping[s,k] = 1 nếu UE k thuộc slice s.
    #
    # Shape: (S,K)
    # ============================================================
    slice_mapping = np.zeros(
        (num_slices, num_UEs),
        dtype=float
    )

    # ============================================================
    # 8. Slice-specific weights
    #
    # Shape: (S,)
    # ============================================================
    w_acc_s = np.zeros(num_slices, dtype=float)
    w_thr_s = np.zeros(num_slices, dtype=float)
    w_lat_s = np.zeros(num_slices, dtype=float)

    # ============================================================
    # 9. Khoảng cách RU - UE
    #
    # d_i_k[i,k]:
    #   khoảng cách từ RU i đến UE k.
    #
    # Không cần chỉ số slice vì khoảng cách vật lý RU-UE không
    # phụ thuộc vào slice mà UE đang sử dụng.
    #
    # Shape: (I,K)
    # ============================================================
    d_i_k = np.zeros(
        (num_RUs, num_UEs),
        dtype=float
    )

    # ============================================================
    # 10. Channel gain
    #
    # Hiện vẫn giữ:
    #
    #   g_ib_sk[i,b,s,k]
    #
    # Shape: (I,B,S,K)
    #
    # UE k chỉ có gain tại slice mà UE đó thuộc vào.
    # Các slice khác bằng 0.
    #
    # Gain hiện tại giống nhau trên tất cả RB của cùng RU-UE.
    # ============================================================
    g_ib_sk = np.zeros(
        (
            num_RUs,
            num_RBs,
            num_slices,
            num_UEs
        ),
        dtype=float
    )

    # ============================================================
    # 11. Chuyển dữ liệu từng UE
    # ============================================================
    for k in range(num_UEs):

        ue = UEs[k]

        # --------------------------------------------------------
        # Slice của UE k
        # --------------------------------------------------------
        s = SLICE_NAMES.index(ue["type"])

        # --------------------------------------------------------
        # Slice mapping
        # --------------------------------------------------------
        slice_mapping[s, k] = 1.0

        # --------------------------------------------------------
        # QoS của UE k
        # --------------------------------------------------------
        R_min_k[k] = float(ue["R_min"])
        L_max_k[k] = float(ue["delay"])

        # --------------------------------------------------------
        # Traffic của UE k
        # --------------------------------------------------------
        packet_size_bits_k[k] = float(
            ue["packet_size_bits"]
        )

        lambda_k[k] = float(
            ue["lambda_pps"]
        )

        eta_k[k] = float(
            ue["eta_slice"]
        )

        # --------------------------------------------------------
        # Slice-specific weights
        #
        # Giả sử mọi UE thuộc cùng một slice có cùng bộ weight.
        # --------------------------------------------------------
        w_acc_s[s] = float(
            ue["weight_accept"]
        )

        w_thr_s[s] = float(
            ue["weight_throughput"]
        )

        w_lat_s[s] = float(
            ue["weight_latency"]
        )

        # --------------------------------------------------------
        # Distance RU - UE
        #
        # Không còn chỉ số slice.
        # --------------------------------------------------------
        for i in range(num_RUs):
            d_i_k[i, k] = float(
                ue["distances_RU_UE"][i]
            )

        # --------------------------------------------------------
        # Channel gain
        #
        # Hiện gain không phụ thuộc RB nên lặp cho toàn bộ RB.
        # Chỉ đặt giá trị ở đúng slice của UE.
        # --------------------------------------------------------
        for i in range(num_RUs):

            gain_i_k = float(
                ue["gain"][i]
            )

            for b in range(num_RBs):
                g_ib_sk[i, b, s, k] = gain_i_k

    # ============================================================
    # 12. Trả dữ liệu cho Long_Term()
    # ============================================================
    return {
        # ---------------- System size ----------------
        "num_slices": num_slices,
        "num_UEs": num_UEs,
        "num_RUs": num_RUs,
        "num_DUs": num_DUs,
        "num_CUs": num_CUs,
        "num_RBs": num_RBs,

        # ---------------- UE QoS ----------------
        "R_min_k": R_min_k,
        "L_max_k": L_max_k,
        "packet_size_bits_k": packet_size_bits_k,
        "lambda_k": lambda_k,

        # ---------------- RAN resources ----------------
        "P_i": P_i,
        "A_j": A_j,
        "A_m": A_m,

        "k_DU": k_DU,
        "k_CU": k_CU,
        "eta_k": eta_k,

        # ---------------- Topology ----------------
        "l_ru_du": l_ru_du,
        "l_du_cu": l_du_cu,

        # ---------------- Wireless ----------------
        "W_b": W_b,
        "g_ib_sk": g_ib_sk,
        "d_i_k": d_i_k,

        # ---------------- Slice ----------------
        "slice_mapping": slice_mapping,

        # ---------------- Global weights ----------------
        "w_acc": w_acc,
        "w_thr": w_thr,
        "w_lat": w_lat,

        # ---------------- Slice weights ----------------
        "w_acc_s": w_acc_s,
        "w_thr_s": w_thr_s,
        "w_lat_s": w_lat_s,
    }


def print_data(data):
    """
    In dữ liệu sau khi chuyển state -> Long-Term
    để kiểm tra kích thước và giá trị.
    """

    print("=" * 80)
    print("LONG-TERM INPUT DATA")
    print("=" * 80)

    # ============================================================
    # SYSTEM SIZE
    # ============================================================
    print("\n--- SYSTEM SIZE ---")

    print("num_slices =", data["num_slices"])
    print("num_UEs    =", data["num_UEs"])
    print("num_RUs    =", data["num_RUs"])
    print("num_DUs    =", data["num_DUs"])
    print("num_CUs    =", data["num_CUs"])
    print("num_RBs    =", data["num_RBs"])

    # ============================================================
    # QoS / TRAFFIC
    # ============================================================
    print("\n--- UE QoS / TRAFFIC ---")

    print("\nR_min_k:")
    print(data["R_min_k"])
    print("shape =", data["R_min_k"].shape)

    print("\nL_max_k:")
    print(data["L_max_k"])
    print("shape =", data["L_max_k"].shape)

    print("\npacket_size_bits_k:")
    print(data["packet_size_bits_k"])
    print(
        "shape =",
        data["packet_size_bits_k"].shape
    )

    print("\nlambda_k:")
    print(data["lambda_k"])
    print("shape =", data["lambda_k"].shape)

    print("\neta_k:")
    print(data["eta_k"])
    print("shape =", data["eta_k"].shape)

    # ============================================================
    # RAN RESOURCES
    # ============================================================
    print("\n--- RAN RESOURCES ---")

    print("\nP_i:")
    print(data["P_i"])
    print("shape =", data["P_i"].shape)

    print("\nA_j:")
    print(data["A_j"])
    print("shape =", data["A_j"].shape)

    print("\nA_m:")
    print(data["A_m"])
    print("shape =", data["A_m"].shape)

    print("\nk_DU =", data["k_DU"])
    print("k_CU =", data["k_CU"])

    # ============================================================
    # TOPOLOGY
    # ============================================================
    print("\n--- TOPOLOGY ---")

    print("\nl_ru_du:")
    print(data["l_ru_du"])
    print(
        "shape =",
        data["l_ru_du"].shape
    )

    print("\nl_du_cu:")
    print(data["l_du_cu"])
    print(
        "shape =",
        data["l_du_cu"].shape
    )

    # ============================================================
    # WIRELESS
    # ============================================================
    print("\n--- WIRELESS ---")

    print("\nW_b =", data["W_b"])

    print("\nd_i_k:")
    print(data["d_i_k"])
    print(
        "shape =",
        data["d_i_k"].shape
    )

    print(
        "\ng_ib_sk shape =",
        data["g_ib_sk"].shape
    )

    # Gain giống nhau trên các RB nên chỉ cần kiểm tra RB đầu tiên.
    print(
        "\ng_ib_sk[:, 0, :, :] "
        "(RB đầu tiên):"
    )

    print(
        data["g_ib_sk"][:, 0, :, :]
    )

    # ============================================================
    # SLICE
    # ============================================================
    print("\n--- SLICE ---")

    print("\nslice_mapping:")
    print(data["slice_mapping"])
    print(
        "shape =",
        data["slice_mapping"].shape
    )

    print("\nw_acc_s:")
    print(data["w_acc_s"])

    print("\nw_thr_s:")
    print(data["w_thr_s"])

    print("\nw_lat_s:")
    print(data["w_lat_s"])

    # ============================================================
    # GLOBAL WEIGHTS
    # ============================================================
    print("\n--- GLOBAL WEIGHTS ---")

    print("w_acc =", data["w_acc"])
    print("w_thr =", data["w_thr"])
    print("w_lat =", data["w_lat"])

    print("\n" + "=" * 80)

