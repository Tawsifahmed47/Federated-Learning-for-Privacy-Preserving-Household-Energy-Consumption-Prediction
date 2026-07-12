from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clients.client import FLClient
from evaluation.evaluator import Evaluator
from server.server import FLServer


class FederatedTrainer:
    def __init__(
        self,
        model_fn,
        input_dim: int,
        client_loaders: Dict[str, DataLoader],
        global_test_loader: DataLoader,
        config: Dict,
        device: torch.device,
        logger,
    ) -> None:
        self.model_fn = model_fn
        self.input_dim = input_dim
        self.client_loaders = client_loaders
        self.global_test_loader = global_test_loader
        self.config = config
        self.device = device
        self.logger = logger

        self.algorithm = config["training"]["algorithm"].lower()
        self.num_rounds = config["training"]["num_rounds"]
        self.client_fraction = config["training"]["client_fraction"]
        self.local_epochs = config["training"]["local_epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.mu = config["training"].get("fedprox_mu", 0.0)
        self.malicious_ids = set(config["training"].get("malicious_client_ids", []))
        self.malicious_probability = float(
            config["training"].get("malicious_probability", 0.0)
        )
        self.enable_dp = bool(config["bonus"].get("enable_dp", False))
        self.dp_noise_std = float(config["bonus"].get("dp_noise_std", 0.0))
        self.metrics_output_dir = config["paths"]["metrics_dir"]
        self.anomaly_cfg = config.get("anomaly", {"enabled": False})

        self.server = FLServer(self.model_fn(self.input_dim).to(self.device), self.algorithm)
        self.clients = {
            cid: FLClient(
                client_id=cid,
                data_loader=loader,
                model_fn=lambda: self.model_fn(self.input_dim),
                device=self.device,
                learning_rate=self.learning_rate,
                local_epochs=self.local_epochs,
                malicious=(cid in self.malicious_ids),
                malicious_probability=self.malicious_probability,
                enable_dp=self.enable_dp,
                dp_noise_std=self.dp_noise_std,
                anomaly_cfg=self.anomaly_cfg,
            )
            for cid, loader in self.client_loaders.items()
        }

    def train(self) -> Dict[str, Any]:
        history = {
            "loss": [],
            "mae": [],
            "rmse": [],
            "mse": [],
            "r2": [],
            "total_energy_kwh": [],
            "avg_energy_kwh": [],
            "anomaly_count": [],
            "detection_recall": [],
            "detection_precision": [],
        }
        client_history = {cid: {"loss": [], "round": [], "recall": [], "precision": []} for cid in self.clients.keys()}
        all_client_ids = list(self.clients.keys())

        for rnd in range(1, self.num_rounds + 1):
            selected_ids = self.server.sample_clients(all_client_ids, self.client_fraction)
            global_state: OrderedDict = self.server.get_global_state()

            client_states = []
            sample_counts = []
            round_losses = []
            updated_c_locals = []

            for cid in selected_ids:
                client = self.clients[cid]
                c_global = self.server.c_global if self.algorithm == "scaffold" else None
                c_local = (
                    self.server.get_client_control_variate(cid)
                    if self.algorithm == "scaffold"
                    else None
                )
                state, n_samples, local_loss, c_local_new = client.train(
                    global_state=global_state,
                    algorithm=self.algorithm,
                    mu=self.mu,
                    c_global=c_global,
                    c_local=c_local,
                )
                client_states.append(state)
                sample_counts.append(n_samples)
                round_losses.append(local_loss)
                
                # Track per-client loss
                client_history[cid]["loss"].append(local_loss)
                client_history[cid]["round"].append(rnd)

                if self.algorithm == "scaffold" and c_local_new is not None:
                    updated_c_locals.append(c_local_new)

            self.server.aggregate(
                selected_ids=selected_ids,
                client_states=client_states,
                sample_counts=sample_counts,
                updated_c_locals=updated_c_locals if self.algorithm == "scaffold" else None,
            )

            global_model = self.server.model
            eval_metrics = Evaluator.evaluate(
                global_model, self.global_test_loader, self.device, self.anomaly_cfg
            )
            avg_round_loss = float(sum(round_losses) / max(len(round_losses), 1))

            # Client robustness testing
            robustness_results = []
            for cid in selected_ids:
                rob = self.clients[cid].test_anomaly_robustness()
                if rob:
                    robustness_results.append(rob)
                    # Track per-client robustness
                    client_history[cid]["recall"].append(rob["detection_recall"])
                    client_history[cid]["precision"].append(rob["detection_precision"])
            
            avg_recall = 0.0
            avg_precision = 0.0
            if robustness_results:
                avg_recall = sum(r["detection_recall"] for r in robustness_results) / len(robustness_results)
                avg_precision = sum(r["detection_precision"] for r in robustness_results) / len(robustness_results)

            history["loss"].append(avg_round_loss)
            for k in ["mae", "rmse", "mse", "r2", "total_energy_kwh", "avg_energy_kwh"]:
                history[k].append(eval_metrics[k])

            anomaly_count = eval_metrics.get("anomaly_count", 0)
            history["anomaly_count"].append(anomaly_count)
            history["detection_recall"].append(avg_recall)
            history["detection_precision"].append(avg_precision)

            self.logger.info(
                f"[FL:{self.algorithm}] Round {rnd}/{self.num_rounds} | "
                f"loss={avg_round_loss:.6f} mae={eval_metrics['mae']:.6f} "
                f"energy_total={eval_metrics['total_energy_kwh']:.2f} "
                f"anomalies={anomaly_count} "
                f"robustness_recall={avg_recall:.2f} "
                f"| selected_clients={len(selected_ids)}"
            )

        return {"global": history, "clients": client_history}

    def save_round_metrics_csv(self, history: Dict[str, List[float]], run_id: str, prefix: str) -> str:
        Path(self.metrics_output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(history)
        df["round"] = list(range(1, len(history["loss"]) + 1))
        out_path = str(Path(self.metrics_output_dir) / f"{prefix}_round_metrics_{run_id}.csv")
        df.to_csv(out_path, index=False)
        return out_path

    @staticmethod
    def save_plots(history: Dict[str, List[float]], output_dir: str, prefix: str) -> None:
        x = range(1, len(history["loss"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(x, history["loss"], label="Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Loss vs Rounds")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_loss_vs_rounds.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(x, history["mae"], label="MAE", color="orange")
        plt.xlabel("Rounds")
        plt.ylabel("MAE")
        plt.title("MAE vs Rounds")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_mae_vs_rounds.png")
        plt.close()

        if "detection_recall" in history and history["detection_recall"]:
            plt.figure(figsize=(8, 5))
            plt.plot(x, history["detection_recall"], label="Recall", marker='o')
            plt.plot(x, history["detection_precision"], label="Precision", marker='s')
            plt.xlabel("Rounds")
            plt.ylabel("Score")
            plt.title("Anomaly Detection Robustness")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{prefix}_anomaly_robustness.png")
            plt.close()

    @staticmethod
    def save_client_plots(
        global_history: Dict[str, List[float]],
        client_history: Dict[str, Dict[str, List[float]]], 
        output_dir: str, 
        prefix: str
    ) -> None:
        """Plot individual client metrics and a comparison summary."""
        client_plots_dir = Path(output_dir) / "clients"
        client_plots_dir.mkdir(parents=True, exist_ok=True)

        global_loss = global_history.get("loss", [])
        global_rounds = list(range(1, len(global_loss) + 1))

        # 1. Comprehensive individual plots for each client
        for cid, history in client_history.items():
            if not history["loss"]:
                continue
            
            # Create a figure with 2 subplots (top: Loss, bottom: Robustness)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            # Subplot 1: Loss (Local vs Global)
            ax1.plot(history["round"], history["loss"], label=f"Client {cid} (Local)", marker='x', color='tab:blue', linewidth=2)
            if global_loss:
                ax1.plot(global_rounds, global_loss, label="Global Average", linestyle='--', color='tab:red', alpha=0.7)
            
            ax1.set_ylabel("Loss")
            ax1.set_title(f"Client {cid}: Training Performance")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            # Subplot 2: Anomaly Detection Robustness
            if history.get("recall"):
                ax2.plot(history["round"], history["recall"], label="Recall", marker='o', color='tab:green')
                ax2.plot(history["round"], history["precision"], label="Precision", marker='s', color='tab:orange')
                ax2.set_ylabel("Score")
                ax2.set_xlabel("Round")
                ax2.set_title(f"Client {cid}: Anomaly Detection Robustness")
                ax2.set_ylim(-0.05, 1.05)
                ax2.grid(True, linestyle='--', alpha=0.6)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "No Robustness Data Available", ha='center', va='center')
                ax2.set_xlabel("Round")

            plt.tight_layout()
            plt.savefig(f"{client_plots_dir}/{prefix}_client_{cid}_full_metrics.png")
            plt.close()

    def save_client_prediction_plots(self, output_dir: str, prefix: str) -> None:
        """Generate Actual vs Prediction plots for each individual client."""
        client_plots_dir = Path(output_dir) / "clients" / "predictions"
        client_plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating individual prediction plots for {len(self.clients)} clients...")
        
        for cid, client in self.clients.items():
            # Use the final global model to see how it performs on this specific client
            Evaluator.save_prediction_plots(
                model=self.server.model,
                data_loader=client.data_loader,
                device=self.device,
                output_dir=str(client_plots_dir),
                prefix=f"{prefix}_client_{cid}",
                anomaly_cfg=self.anomaly_cfg
            )

