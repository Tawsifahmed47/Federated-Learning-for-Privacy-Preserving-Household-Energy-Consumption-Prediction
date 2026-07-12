# Federated Energy Project

Federated Learning Simulator for Privacy-Preserving Energy Consumption Prediction.

## Project Structure

- `data_loader/`: centralized and per-client loaders
- `models/`: PyTorch MLP regression model
- `clients/`: local client training logic (FedAvg/FedProx/SCAFFOLD support)
- `server/`: FL server orchestration and aggregation strategies
- `training/`: federated and centralized training pipelines
- `evaluation/`: evaluation utilities
- `utils/`: helpers, metrics, logging
- `configs/config.yaml`: all runtime configuration
- `results/`: logs, plots, trained model checkpoints
- `main.py`: entry point

## Dataset

Configured path: `data set/Output` (contains parquet/json/csv and `clients/` directory).

## Setup

1. Create environment (recommended):
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `pip install torch pandas pyarrow scikit-learn matplotlib pyyaml`

## Run

- Edit config in `configs/config.yaml`:
  - `training.mode`: `federated` / `centralized` / `both`
  - `training.algorithm`: `fedavg` / `fedprox` / `scaffold`
- Execute:
  - `python main.py`

## Outputs

Automatically saved under `results/`:
- `results/logs/`: per-run logs with round/epoch metrics
- `results/plots/`: loss/MAE curves and FL vs centralized comparison plot
- `results/models/`: trained model checkpoints
- `results/metrics/`: per-round (FL) and per-epoch (centralized) CSV metrics

## Research Notes

- Supports client sampling per round (`client_fraction`)
- FedProx proximal regularization via `training.fedprox_mu`
- Simplified SCAFFOLD control variates for variance reduction
- Optional malicious client simulation (`malicious_client_ids`, `malicious_probability`)
- Partition-aware experiments (`training.partitioning.mode`: `all` / `iid` / `noniid`)
- Optional differential privacy gradient noise (`bonus.enable_dp`, `bonus.dp_noise_std`)
- Optional communication cost estimate


