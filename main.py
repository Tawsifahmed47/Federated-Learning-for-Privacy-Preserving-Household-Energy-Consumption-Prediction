from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from data_loader.client_loader import ClientDataLoader
from data_loader.global_loader import GlobalDataLoader
from evaluation.evaluator import Evaluator
from models.energy_model import EnergyMLP
from training.centralized_trainer import CentralizedTrainer
from training.federated_trainer import FederatedTrainer
from utils.helpers import ensure_dir, load_yaml, set_seed
from utils.logger import build_logger

 
def model_factory(input_dim: int):
    return EnergyMLP(input_dim=input_dim)


def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_comparison_plot(results: Dict[str, Dict[str, float]], out_path: str) -> None:
    labels = list(results.keys())
    mae_vals = [results[k]["mae"] for k in labels]
    rmse_vals = [results[k]["rmse"] for k in labels]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar([i - width / 2 for i in x], mae_vals, width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], rmse_vals, width=width, label="RMSE")
    plt.xticks(list(x), labels, rotation=45)
    plt.ylabel("Metric Value")
    plt.title("Error Comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    energy_vals = [results[k].get("total_energy_kwh", 0) for k in labels]
    plt.bar(labels, energy_vals, color="green", alpha=0.7)
    plt.xticks(rotation=45)
    plt.ylabel("Total Energy (kWh)")
    plt.title("Total Energy Consumption")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_anomaly_plot(history: Dict[str, List[float]], out_path: str, title: str) -> None:
    if "anomaly_count" not in history or not history["anomaly_count"]:
        return

    x = range(1, len(history["anomaly_count"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, history["anomaly_count"], marker="o", linestyle="-", color="red")
    plt.xlabel("Round / Epoch")
    plt.ylabel("Anomaly Count")
    plt.title(f"Detected Anomalies: {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_comm_cost_bytes(model: torch.nn.Module, rounds: int, selected_clients_avg: float) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_model = n_params * 4
    per_round = bytes_per_model * 2 * selected_clients_avg
    return per_round * rounds


def resolve_partition_clients(config: Dict) -> List[str] | None:
    partition_cfg = config["training"].get("partitioning", {})
    mode = str(partition_cfg.get("mode", "all")).lower()
    if mode == "all":
        return None

    data_root = Path(config["paths"]["data_root"])
    if mode == "iid":
        partition_path = data_root / "iid_partition.json"
    elif mode == "noniid":
        partition_path = data_root / "noniid_partition.json"
    else:
        raise ValueError("partitioning.mode must be one of: all, iid, noniid")

    with open(partition_path, "r", encoding="utf-8") as f:
        partition_data = json.load(f)

    selected_groups = partition_cfg.get("selected_groups")
    max_clients = partition_cfg.get("max_clients")

    client_ids: List[str] = []
    for group, members in partition_data.items():
        if not isinstance(members, list):
            continue
        if selected_groups and group not in selected_groups:
            continue
        client_ids.extend([m for m in members if isinstance(m, str) and m.strip()])

    # De-duplicate while preserving order.
    seen = set()
    ordered_unique = []
    for cid in client_ids:
        if cid not in seen:
            ordered_unique.append(cid)
            seen.add(cid)

    if isinstance(max_clients, int) and max_clients > 0:
        ordered_unique = ordered_unique[:max_clients]
    return ordered_unique


def main() -> None:
    config = load_yaml("configs/config.yaml")
    set_seed(config["project"]["seed"])

    for p in [
        config["paths"]["results_root"],
        config["paths"]["logs_dir"],
        config["paths"]["plots_dir"],
        config["paths"]["models_dir"],
        config["paths"]["metrics_dir"],
    ]:
        ensure_dir(p)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = build_logger("federated_energy", Path(config["paths"]["logs_dir"]) / f"run_{run_id}.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_root = config["paths"]["data_root"]
    global_loader = GlobalDataLoader(
        data_root=data_root,
        target_column=config["data"]["target_column"],
        batch_size=config["data"]["batch_size"],
        drop_columns=config["data"].get("drop_columns", []),
        use_categorical=bool(config["data"].get("use_categorical", False)),
        max_categorical_levels=int(config["data"].get("max_categorical_levels", 50)),
        num_workers=config["data"]["num_workers"],
    )
    train_loader, test_loader, input_dim = global_loader.load_centralized()

    client_loader = ClientDataLoader(
        data_root=data_root,
        target_column=config["data"]["target_column"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    partition_client_ids = resolve_partition_clients(config)
    client_loaders, _, _ = client_loader.load_client_datasets(
        global_loader=global_loader,
        allowed_client_ids=partition_client_ids,
    )

    num_clients_cfg = config["training"]["num_clients"]
    if num_clients_cfg is not None:
        selected_keys = list(client_loaders.keys())[: int(num_clients_cfg)]
        client_loaders = {k: client_loaders[k] for k in selected_keys}

    mode = config["training"]["mode"].lower()
    comparison = {}

    if mode in ("federated", "both"):
        fl_trainer = FederatedTrainer(
            model_fn=model_factory,
            input_dim=input_dim,
            client_loaders=client_loaders,
            global_test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
        )
        fl_results = fl_trainer.train()
        fl_history = fl_results["global"]
        fl_client_history = fl_results["clients"]
        
        alg = config["training"]["algorithm"].lower()
        FederatedTrainer.save_plots(fl_history, config["paths"]["plots_dir"], f"fl_{alg}")
        FederatedTrainer.save_client_plots(fl_history, fl_client_history, config["paths"]["plots_dir"], f"fl_{alg}")
        fl_trainer.save_client_prediction_plots(config["paths"]["plots_dir"], f"fl_{alg}")
        
        # Add the new prediction plots for FL
        Evaluator.save_prediction_plots(
            fl_trainer.server.model,
            test_loader,
            device,
            config["paths"]["plots_dir"],
            f"fl_{alg}",
            config.get("anomaly")
        )

        fl_metrics_csv = fl_trainer.save_round_metrics_csv(fl_history, run_id, f"fl_{alg}")
        save_model(
            fl_trainer.server.model,
            str(Path(config["paths"]["models_dir"]) / f"fl_{alg}_{run_id}.pt"),
        )
        logger.info(f"Saved FL per-round metrics CSV: {fl_metrics_csv}")
        comparison[f"FL-{alg}"] = {
            k: fl_history[k][-1]
            for k in [
                "mae",
                "rmse",
                "mse",
                "r2",
                "total_energy_kwh",
                "avg_energy_kwh",
                "detection_recall",
                "detection_precision",
            ]
            if k in fl_history
        }
        save_anomaly_plot(
            fl_history,
            str(Path(config["paths"]["plots_dir"]) / f"fl_{alg}_anomalies_{run_id}.png"),
            f"FL-{alg}",
        )

        if config["bonus"]["estimate_communication_cost"]:
            avg_selected = max(1, int(config["training"]["client_fraction"] * len(client_loaders)))
            comm_bytes = compute_comm_cost_bytes(
                fl_trainer.server.model, config["training"]["num_rounds"], avg_selected
            )
            logger.info(f"Estimated communication cost (bytes): {comm_bytes:.2f}")

    if mode in ("centralized", "both"):
        central_model = model_factory(input_dim).to(device)
        ct = CentralizedTrainer(
            model=central_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=config["training"]["learning_rate"],
            epochs=config["training"]["centralized_epochs"],
            logger=logger,
            anomaly_cfg=config.get("anomaly", {"enabled": False}),
        )
        c_history = ct.train()
        
        # Add the new prediction plots for Centralized
        Evaluator.save_prediction_plots(
            central_model,
            test_loader,
            device,
            config["paths"]["plots_dir"],
            "centralized",
            config.get("anomaly")
        )

        c_metrics_path = str(Path(config["paths"]["metrics_dir"]) / f"centralized_epoch_metrics_{run_id}.csv")
        CentralizedTrainer.save_epoch_metrics_csv(c_history, c_metrics_path)
        save_model(central_model, str(Path(config["paths"]["models_dir"]) / f"centralized_{run_id}.pt"))
        logger.info(f"Saved centralized per-epoch metrics CSV: {c_metrics_path}")
        comparison["Centralized"] = {
            k: c_history[k][-1]
            for k in ["mae", "rmse", "mse", "r2", "total_energy_kwh", "avg_energy_kwh"]
        }
        save_anomaly_plot(
            c_history,
            str(Path(config["paths"]["plots_dir"]) / f"centralized_anomalies_{run_id}.png"),
            "Centralized",
        )

    if len(comparison) >= 2:
        save_comparison_plot(
            comparison, str(Path(config["paths"]["plots_dir"]) / f"comparison_{run_id}.png")
        )

    logger.info("Run completed.")
    for name, metrics in comparison.items():
        logger.info(f"{name}: {metrics}")


if __name__ == "__main__":
    main()

