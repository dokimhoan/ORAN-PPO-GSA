# Joint Network Slicing and Resource Orchestration in Open RAN

This repository contains the source code and experiment configurations
used in the paper "Joint Network Slicing and Resource Orchestration in
Open RAN via Proximal Policy Optimization."

## Software environment

- Python 3.12.6
- CVXPY 1.9.2
- MOSEK 11.2.3
- PyTorch 2.10.0
- CUDA 12.8
- Gymnasium 1.4.0

The complete dependency list is provided in `requirements.txt`.

## Implemented methods

- Two-stage optimization (TSO)
- PPO with graph sampling and aggregation (PPO-GSA)
- PPO with a multilayer perceptron (PPO-MLP)
- Nearest-RU
- Round-robin
- Random-RU

## Installation

Create a Python virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
