# Joint Network Slicing and Resource Orchestration in Open RAN

**PPO-GSA: Graph-based Proximal Policy Optimization for joint UE admission and network resource orchestration**

This repository contains a Python simulation and evaluation framework for joint network slicing and resource orchestration in an Open Radio Access Network (O-RAN). It studies admission and assignment of user equipment (UEs) to radio units (RUs), distributed units (DUs), and centralized units (CUs), together with radio resource block (RB) and transmit-power allocation.

The code accompanies the research work **“Joint Network Slicing and Resource Orchestration in Open RAN via Proximal Policy Optimization.”** It compares a graph-based PPO policy with an MLP-based PPO policy and mathematical-optimization/heuristic baselines.

> **Scope:** The `RLEnv` implementation processes UEs sequentially within a network snapshot (one UE per environment step). This repository should not be interpreted as a mobility or handover simulator.

## Implemented methods

| Method | Description |
| --- | --- |
| **PPO-GSA** | PPO agent with graph-based network representation, implemented in `algorithms/PPO_GSA_MLP.py`. |
| **PPO-MLP** | PPO agent with an MLP policy, implemented in `algorithms/PPO_MLP.py`. |
| **BLO** | Optimization-based benchmark with long-term and short-term stages, implemented in `algorithms/MILP_BLO.py`. |
| **Nearest-RU** | Nearest-RU baseline, implemented in `algorithms/MILP_Nearest.py`. |
| **RoundRobin** | Round-robin RU assignment baseline, implemented in `algorithms/MILP_RoundRobin.py`. |
| **Random-RU** | Random-RU assignment baseline, implemented in `algorithms/MILP_Random.py`. |

The evaluation runner uses the labels `BLO`, `PPO-GSA`, `PPO-MLP`, `Nearest-RU`, `RoundRobin`, and `Random-RU`.

## Repository layout

```text
ORAN-PPO-GSA/
├── algorithms/             # PPO implementations and optimization baselines
├── env/                    # Network simulation and sequential RL environment
├── utils/                  # Data conversion and supporting utilities
├── results_eval_20/        # Committed evaluation outputs
├── results_plot_20/        # Committed plotting outputs
├── results_train_100/      # Committed training outputs
├── config.py               # Wireless, slice, resource, and objective settings
├── main_train.py           # Train PPO-GSA and PPO-MLP
├── main_eval.py            # Compare PPO, BLO, and RU baselines
├── main_evalution_2.py     # Additional evaluation script
├── main_evalution_3.py     # Additional evaluation script
└── README.md
```

> The result-directory names above reflect the repository root as inspected. The default scripts currently use `results_train_10` for training/checkpoints, while the committed training-results directory is named `results_train_100`. These paths need to be aligned if you intend to reuse existing checkpoints.

## Model and configuration

The environment includes two traffic/service classes, **eMBB** and **uRLLC**, and enforces service- and resource-related feasibility constraints.

Selected values in `config.py`:

| Parameter | eMBB | uRLLC |
| --- | ---: | ---: |
| Minimum requested rate | 50 Mbps | 5 Mbps |
| Minimum SINR | 10 dB | 20 dB |
| Maximum latency | 5 ms | 1 ms |
| Packet size | 12,000 bits | 1,024 bits |
| Packet arrival rate | 100 packets/s | 500 packets/s |
| Slice weights (admission / throughput / latency) | 0.3 / 0.4 / 0.3 | 0.3 / 0.3 / 0.4 |

Other relevant configuration values:

- RB bandwidth: **720 kHz** (`12 × 60 kHz`), as computed by `W_b`.
- Maximum RBs allocated to one UE: **10**.
- RU maximum transmit power: **43 dBm** (approximately 20 W).
- Discrete power codebook: **20 levels**.
- Carrier frequency: **6 GHz**; number of RU antennas: **32**; noise figure: **5 dB**.
- Global objective weights (admission / throughput / latency): **100 / 1 / 20**.

Refer to `config.py` for the actual values used by a run. In particular, update that file rather than relying on parameters quoted in older experimental descriptions.

## Software requirements

The previous README listed the following software versions as the author's environment:

- Python 3.12.6
- PyTorch 2.10.0 (CUDA 12.8)
- CVXPY 1.9.2
- MOSEK 11.2.3
- Gymnasium 1.4.0

These are **reported environment versions, not a verified reproducible lockfile**. The repository root did not expose a `requirements.txt` at the time this README was prepared. The code also imports NumPy and pandas. MOSEK-based experiments require an appropriately configured MOSEK installation and license.

### Set up a local environment

```bash
git clone https://github.com/dokimhoan/ORAN-PPO-GSA.git
cd ORAN-PPO-GSA
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the dependencies according to the Python imports and solver requirements of your setup. **Do not run `pip install -r requirements.txt` until that file has actually been added to the repository.** For exact reproduction, publish a validated dependency manifest and document the MOSEK license/setup separately.

## Training

Run from the repository root:

```bash
python main_train.py
```

The script sequentially trains the PPO-GSA-MLP and PPO-MLP agents. Its current defaults include:

| Setting | Value |
| --- | --- |
| Random seed | `10` |
| UEs | `20–50` |
| RBs | `50–135` |
| RUs | `5–7` |
| DUs | `3–5` |
| CUs | `3–5` |
| Parallel environments | `64` |
| CPU workers | `16` |
| Rollout steps | `8192` |
| Minibatch size | `1024` |
| Resume from checkpoint | `False` |
| Output root | `results_train_10/` |

The settings are defined near the beginning of `main_train.py` and may require adjustment for machines with less memory or CPU/GPU capacity. A CUDA GPU is detected automatically; the script also contains a CPU fallback.

## Evaluation

The main comparison script is `main_eval.py`:

```bash
python main_eval.py --runs 50 --device auto
```

**Prerequisite:** The script expects both trained checkpoints at the following exact paths:

```text
results_train_10/PPO_GSA_MLP/PPOGSA_best.pt
results_train_10/PPO_MLP/PPOMLP_best.pt
```

If your checkpoints are elsewhere, update `TRAIN_ROOT`, `PPO_GSA_CHECKPOINT`, and `PPO_MLP_CHECKPOINT` in `main_eval.py` before running. The script checks that both files exist and raises `FileNotFoundError` otherwise.

The built-in evaluation configurations are specified as `(UEs, RBs, RUs, DUs, CUs)`:

```text
(20,  30, 5, 3, 3)
(30, 135, 7, 5, 5)
(40, 135, 7, 5, 5)
(50, 135, 7, 5, 5)
```

The script evaluates methods on paired seeded RAN snapshots and only accepts a sample when all six methods yield valid results. The default settings request **50 accepted runs per configuration**, with up to **500 attempts**, an MILP time limit of **120 seconds**, and a relative MIP gap of **0.05**.

Useful options:

```bash
python main_eval.py --help
python main_eval.py --runs 5 --max-attempts 50 --device cpu --results results_eval_test
```

The second command is an example of a smaller evaluation, **not a guarantee that five valid paired samples will be obtained**. MOSEK must be correctly installed and licensed for the optimization baselines.

Evaluation artifacts are written to the selected results directory (default: `results_eval_20/`):

- `evaluation_runs.csv` — results for accepted method/configuration/seed combinations.
- `evaluation_summary.csv` — aggregated metrics.
- `evaluation_metadata.json` — run configuration and software metadata.

Existing files are not overwritten unless `--overwrite` is supplied; use this option only when replacement is intended.

## Evaluation metrics

The main evaluation script records the objective score, admitted UE count, admission rate, total throughput (Mbps), total latency (ms), allocated RBs, transmission power (W), and runtime measurements. For BLO, long-term and short-term runtime fields are also included.

## Reproducibility notes

1. Run scripts from the repository root so their relative imports and output paths resolve correctly.
2. Check and align the training/checkpoint paths before attempting evaluation.
3. Use the same `config.py` and random seeds when comparing methods.
4. Document the exact Python environment and solver versions/license if reproducing published numerical results.
5. The provided evaluation runner requires all six methods to return valid results on each accepted snapshot; therefore, its accepted sample count can differ from the number of attempted seeds.

## Research context

This is research code for simulated O-RAN network-slicing/resource-orchestration experiments. Results depend on the simulation assumptions, resource budgets, traffic presets, optimization tolerances, and policy checkpoints. It is not a deployment-ready O-RAN controller.

## Citation

If you use this repository in academic work, please cite the corresponding paper. A verified bibliographic entry (authors, venue, year, DOI) should be added here when available.
