import numpy as np

# =========================================================
# 1. RB VÀ BĂNG THÔNG
# =========================================================
# NR FR1, SCS = 60 kHz.
channel_bandwidth_Hz = 100e6
subcarrier_bandwidth_Hz = 60e3
num_subcarriers_per_RB = 12

# Băng thông của một RB: 12 x 60 kHz = 720 kHz.
W_b = num_subcarriers_per_RB * subcarrier_bandwidth_Hz

# Với 100 MHz và SCS 30 kHz, grid mô phỏng sử dụng 273 RB.
# default_num_RBs = 273

max_RBs_per_UE = 10

# =========================================================
# 2. CẤU HÌNH SLICE / DỊCH VỤ
# =========================================================
SLICE_PRESET = {
    'eMBB': {
        'type'        : 'eMBB',
        
        # Minimum service rate
        'R_min_k'       : 50e6,        # [bps]
        'SINR_min_k'    : 10,          # [dB]
        'L_max_k'       : 5e-3,        # [s]
        
        # Traffic model.
        "packet_size_bits": 1500 * 8, # [bit/packet]
        "lambda_pps": 100.0,  # [packet/s]
        "eta_k": 0.05,
        
        # Reward weights.
        "w_acc_s": 0.3,
        "w_thr_s": 0.4,
        "w_lat_s": 0.3,
    },
    
    'uRLLC': {
        'type'        : 'uRLLC',
        
        # Minimum service rate
        "R_min_k": 5e6,                 # [bit/s]
        "SINR_min_k": 20.0,             # [dB]
        "L_max_k": 1e-3,                # [s]
        
        # Traffic model.
        "packet_size_bits": 128 * 8,  # [bit/packet]
        "lambda_pps": 500.0,  # [packet/s]
        "eta_k": 0.08,
        
        # Reward weights.
        "w_acc_s": 0.3,
        "w_thr_s": 0.3,
        "w_lat_s": 0.4,
    },
}

# =========================================================
# 3. CÔNG SUẤT RU VÀ POWER CODEBOOK CỦA ACTION
# =========================================================
# Tổng công suất tối đa của mỗi RU: 43 dBm ~ 19.95 W.
max_tx_power_dbm    = 43
max_tx_power_mw     = 10 ** (max_tx_power_dbm / 10)  # mW
max_tx_power_watts  = max_tx_power_mw / 1e3          # W

# Action power chia 20 mức.
min_ue_power_watts = 0.0
max_ue_power_watts = max_tx_power_watts
num_power_levels = 20

def generate_power_levels(
    p_min: float = min_ue_power_watts,
    p_max: float = max_ue_power_watts,
    num_levels: int = num_power_levels,
):
    """Sinh codebook công suất rời rạc cho action của agent."""
    if num_levels <= 1:
        return [float(p_max)]

    values = np.linspace(p_min, p_max, num_levels, dtype=float)
    return [float(round(x, 6)) for x in values]

P_ib_k_val = generate_power_levels()

# =========================================================
# 4. CPU CAPACITY TẠI DU/CU
# =========================================================
# Capacity của RU.
P_i_random_list = [10 ** ((42 - 30) / 10), 10 ** ((43 - 30) / 10), 10 ** ((44 - 30) / 10)]
# Capacity của DU, CU.
A_j_random_list = [6e9, 8e9, 10e9] # DU CPU [cycles/s]
A_m_random_list = [3e9, 5e9, 8e9]  # CU CPU [cycles/s]

# Số cycle cần thiết để xử lý một bit dữ liệu.
k_DU = 5.0  # [cycles/bit]: DU cần 5 CPU cycles để xử lý 1 bit.
k_CU = 3.0  # [cycles/bit]: CU cần 3 CPU cycles để xử lý 1 bit.

# =========================================================
# 5. VÙNG MÔ PHỎNG 
# =========================================================
# Vùng mạng là hình vuông 2 km x 2 km, tâm tại (0, 0).
# UE hợp lệ khi x,y thuộc [-1000, 1000] m.
network_size_m = 4000.0

# =========================================================
# 6. WIRELESS MODEL
# =========================================================
# Giữ mô hình 3GPP UMa đơn giản.
carrier_frequency_GHz = 6.0
num_ru_antennas = 32
noise_figure_dB = 5.0
reference_temperature_K = 290.0
speed_of_light_mps = 3e8
# =========================================================
# 7. Global objective weights
# =========================================================
# Old: 1/3 / 1/3 / 1/3
# w_acc = 10
# w_thr = 1
# w_lat = 12

w_acc = 100
w_thr = 1
w_lat = 20