"""Chuyển state của Env/RLEnv sang input cho các thuật toán MILP unslice."""

import numpy as np


# ======================================================================
# 1. Danh sách input chung của các MILP fixed-RU
# ======================================================================

COMMON_KEYS = [
    "num_UEs", "num_RUs", "num_DUs", "num_CUs", "num_RBs",
    "R_min_k", "L_max_k", "packet_size_bits_k", "lambda_k",
    "P_i", "A_j", "A_m", "k_DU", "k_CU", "eta_k",
    "l_ru_du", "l_du_cu", "W_b", "g_ib_k", "d_i_k",
    "w_acc", "w_thr", "w_lat", "w_acc_k", "w_thr_k", "w_lat_k",
]


# ======================================================================
# 2. Raw state -> dữ liệu MILP chuẩn chung
# ======================================================================

def state_to_MILP(state):
    """
    Chuyển state công khai của Env/RLEnv thành dữ liệu MILP chuẩn chung.

    Output:
        Dictionary dùng làm nguồn dữ liệu cho:
            - Long-Term
            - Short-Term
            - Nearest-RU
            - Random-RU
            - RoundRobin-RU
    """

    ran, ues, config = state["state_RAN"], state["state_UEs_req"], state["env_config"]

    # ------------------------------------------------------------------
    # Kích thước network và resource còn lại
    # ------------------------------------------------------------------
    P_i = np.asarray(ran["P_i_rem"], dtype=float)
    A_j = np.asarray(ran["A_j_rem"], dtype=float)
    A_m = np.asarray(ran["A_m_rem"], dtype=float)

    num_UEs = len(ues)
    num_RUs, num_DUs, num_CUs = len(P_i), len(A_j), len(A_m)
    num_RBs = int(ran["RB_rem"])

    # ------------------------------------------------------------------
    # Tham số theo UE
    # ------------------------------------------------------------------
    R_min_k = np.zeros(num_UEs)
    L_max_k = np.zeros(num_UEs)
    packet_size_bits_k = np.zeros(num_UEs)
    lambda_k = np.zeros(num_UEs)
    eta_k = np.zeros(num_UEs)

    w_acc_k = np.zeros(num_UEs)
    w_thr_k = np.zeros(num_UEs)
    w_lat_k = np.zeros(num_UEs)

    # ------------------------------------------------------------------
    # Channel / distance
    # ------------------------------------------------------------------
    d_i_k = np.zeros((num_RUs, num_UEs))
    g_ib_k = np.zeros((num_RUs, num_RBs, num_UEs))

    # Không giả định key UE luôn chính xác là 0,1,... theo thứ tự dictionary.
    ue_items = sorted(ues.items(), key=lambda item: int(item[0]))

    for k, (_, ue) in enumerate(ue_items):
        R_min_k[k] = float(ue["R_min_k"])
        L_max_k[k] = float(ue["L_max_k"])
        packet_size_bits_k[k] = float(ue["packet_size_bits"])
        lambda_k[k] = float(ue["lambda_pps"])
        eta_k[k] = float(ue["eta_k"])

        # Slice weight đã được copy trực tiếp vào từng UE.
        w_acc_k[k] = float(ue["w_acc_s"])
        w_thr_k[k] = float(ue["w_thr_s"])
        w_lat_k[k] = float(ue["w_lat_s"])

        d_i_k[:, k] = np.asarray(ue["d_i_k"], dtype=float)

        # Hiện tại gain của một UE-RU giống nhau trên tất cả RB.
        g_i_k = np.asarray(ue["g_i_k"], dtype=float)
        g_ib_k[:, :, k] = g_i_k[:, None]

    return {
        "num_UEs": num_UEs,
        "num_RUs": num_RUs,
        "num_DUs": num_DUs,
        "num_CUs": num_CUs,
        "num_RBs": num_RBs,

        "R_min_k": R_min_k,
        "L_max_k": L_max_k,
        "packet_size_bits_k": packet_size_bits_k,
        "lambda_k": lambda_k,

        "P_i": P_i,
        "A_j": A_j,
        "A_m": A_m,

        "k_DU": float(config["k_DU"]),
        "k_CU": float(config["k_CU"]),
        "eta_k": eta_k,

        "l_ru_du": np.asarray(ran["l_ru_du"], dtype=float),
        "l_du_cu": np.asarray(ran["l_du_cu"], dtype=float),

        "W_b": float(config["W_b"]),
        "g_ib_k": g_ib_k,
        "d_i_k": d_i_k,

        "w_acc": float(config["w_acc"]),
        "w_thr": float(config["w_thr"]),
        "w_lat": float(config["w_lat"]),

        "w_acc_k": w_acc_k,
        "w_thr_k": w_thr_k,
        "w_lat_k": w_lat_k,
    }


# ======================================================================
# 3. Converter cho từng thuật toán
# ======================================================================

def data_to_long_term(data):
    """
    Tạo input cho Long_Term().

    Dùng:
        result = Long_Term(**data_to_long_term(data))
    """
    return dict(data)


def data_to_short_term(data, long_result, binary_tol=1e-4):
    """
    Tạo input cho Short_Term() từ dữ liệu gốc và nghiệm Long-Term.

    Dùng:
        result = Short_Term(**data_to_short_term(data, long_result))
    """

    if long_result is None or len(long_result) != 13 or long_result[0] is None:
        raise ValueError("Long-Term không có nghiệm hợp lệ.")

    long_pi_k, long_z_ib_k, _, long_phi_i_k, long_phi_j_k, long_phi_m_k = long_result[:6]

    arrays = {
        "pi": get_value_array(long_pi_k),
        "z": get_value_array(long_z_ib_k),
        "phi_i": get_value_array(long_phi_i_k),
        "phi_j": get_value_array(long_phi_j_k),
        "phi_m": get_value_array(long_phi_m_k),
    }

    # Các biến Long-Term phải gần binary trước khi làm tròn.
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} chứa NaN/Inf.")

        if arr.size and np.max(np.abs(arr - np.rint(arr))) > binary_tol:
            raise ValueError(f"{name} không đủ gần binary với tolerance={binary_tol}.")

    short_data = dict(data)

    short_data.update({
        "arr_long_pi_k": np.rint(arrays["pi"]).astype(int),
        "arr_long_z_ib_k": np.rint(arrays["z"]).astype(int),
        "arr_long_phi_i_k": np.rint(arrays["phi_i"]).astype(int),
        "arr_long_phi_j_k": np.rint(arrays["phi_j"]).astype(int),
        "arr_long_phi_m_k": np.rint(arrays["phi_m"]).astype(int),
    })

    return short_data


def data_to_nearest_ru(data):
    """
    Tạo positional input cho Nearest_RU().

    Mỗi UE được cố định vào RU gần nhất.

    Dùng:
        result = Nearest_RU(*data_to_nearest_ru(data))
    """

    phi_i_k = np.zeros((data["num_RUs"], data["num_UEs"]), dtype=int)

    for k in range(data["num_UEs"]):
        phi_i_k[np.argmin(data["d_i_k"][:, k]), k] = 1

    return (*[data[key] for key in COMMON_KEYS], phi_i_k)


def data_to_random_ru(data, seed=None):
    """
    Tạo positional input cho Random_RU().

    Mỗi UE được cố định ngẫu nhiên vào một RU.

    Dùng:
        result = Random_RU(*data_to_random_ru(data, seed=42))
    """

    rng = np.random.default_rng(seed)
    phi_i_k = np.zeros((data["num_RUs"], data["num_UEs"]), dtype=int)

    for k in range(data["num_UEs"]):
        phi_i_k[rng.integers(data["num_RUs"]), k] = 1

    return (*[data[key] for key in COMMON_KEYS], phi_i_k)


def data_to_roundrobin_ru(data):
    """
    Tạo positional input cho RoundRobin_RU().

    UE k được gán:
        RU = k mod num_RUs

    Dùng:
        result = RoundRobin_RU(*data_to_roundrobin_ru(data))
    """

    phi_i_k = np.zeros((data["num_RUs"], data["num_UEs"]), dtype=int)

    for k in range(data["num_UEs"]):
        phi_i_k[k % data["num_RUs"], k] = 1

    return (*[data[key] for key in COMMON_KEYS], phi_i_k)


# ======================================================================
# 4. Helper lấy value từ CVXPY
# ======================================================================

def get_value_array(arr):
    """Chuyển CVXPY Variable/Expression hoặc object-array sang NumPy float."""

    if hasattr(arr, "value"):
        if arr.value is None:
            raise ValueError("CVXPY variable/expression chưa có value.")

        return np.asarray(arr.value, dtype=float)

    arr = np.asarray(arr)

    if arr.dtype != object:
        return arr.astype(float)

    values = np.empty(arr.shape, dtype=float)

    for index in np.ndindex(arr.shape):
        value = arr[index].value if hasattr(arr[index], "value") else arr[index]

        if value is None:
            raise ValueError(f"Phần tử {index} chưa có value.")

        values[index] = float(value)

    return values


# ======================================================================
# 5. Backward-compatible function
# ======================================================================

def long_result_to_short_data(data, long_result, binary_tol=1e-4):
    """
    Giữ tên hàm cũ để các file hiện tại không bị lỗi.

    Tương đương:
        data_to_short_term(data, long_result)
    """
    return data_to_short_term(data, long_result, binary_tol)


# ======================================================================
# 6. Backward-compatible Fixed-RU builders
# ======================================================================

def build_nearest_ru(data):
    """Trả về ma trận fixed phi_i_k của Nearest-RU."""
    return data_to_nearest_ru(data)[-1]


def build_random_ru(data, seed=None):
    """Trả về ma trận fixed phi_i_k của Random-RU."""
    return data_to_random_ru(data, seed)[-1]


def build_roundrobin_ru(data):
    """Trả về ma trận fixed phi_i_k của RoundRobin-RU."""
    return data_to_roundrobin_ru(data)[-1]


# ======================================================================
# 7. Print
# ======================================================================

def print_data(data):
    """In thông tin và giá trị dữ liệu MILP."""

    print("\n" + "=" * 100)
    print("MILP INPUT")
    print("=" * 100)

    print(
        f"UEs={data['num_UEs']}, "
        f"RBs={data['num_RBs']}, "
        f"RUs={data['num_RUs']}, "
        f"DUs={data['num_DUs']}, "
        f"CUs={data['num_CUs']}"
    )


    keys = [
        "R_min_k",
        "L_max_k",
        "packet_size_bits_k",
        "lambda_k",
        "eta_k",
        "w_acc_k",
        "w_thr_k",
        "w_lat_k",
        "d_i_k",
        "g_ib_k",
        "P_i",
        "A_j",
        "A_m",
        "l_ru_du",
        "l_du_cu",
    ]


    for key in keys:

        value = np.asarray(data[key])

        print("\n" + "-" * 80)

        print(f"{key}")
        print(f"shape = {value.shape}")

        # In toàn bộ với vector nhỏ
        if value.size <= 50:
            print(value)

        # Ma trận nhỏ
        elif value.ndim == 2 and value.shape[0] <= 10 and value.shape[1] <= 10:
            print(value)

        # Tensor lớn: chỉ in lát đầu
        elif value.ndim == 3:
            print("first slice:")
            print(value[:, :, 0])

        else:
            print("first values:")
            print(value.flatten()[:20])

    print("\n" + "=" * 100)