from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch import nn

from evaluation.evaluator import Evaluator


class CentralizedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        test_loader,
        device: torch.device,
        learning_rate: float,
        epochs: int,
        logger,
        anomaly_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.logger = logger
        self.anomaly_cfg = anomaly_cfg or {"enabled": False}
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self) -> Dict[str, List[float]]:
        history = {
            "loss": [],
            "mae": [],
            "rmse": [],
            "mse": [],
            "r2": [],
            "total_energy_kwh": [],
            "avg_energy_kwh": [],
            "anomaly_count": [],
        }
        # Get anomaly config if available, else default
        anomaly_cfg = getattr(self, "anomaly_cfg", {"enabled": False})

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            metrics = Evaluator.evaluate(self.model, self.test_loader, self.device, anomaly_cfg)
            history["loss"].append(avg_loss)
            for k in ["mae", "rmse", "mse", "r2", "total_energy_kwh", "avg_energy_kwh"]:
                history[k].append(metrics[k])

            anomaly_count = metrics.get("anomaly_count", 0)
            history["anomaly_count"].append(anomaly_count)

            self.logger.info(
                f"[Centralized] Epoch {epoch}/{self.epochs} | "
                f"loss={avg_loss:.6f} mae={metrics['mae']:.6f} "
                f"energy_total={metrics['total_energy_kwh']:.2f} "
                f"anomalies={anomaly_count}"
            )
        return history

    @staticmethod
    def save_epoch_metrics_csv(history: Dict[str, List[float]], out_path: str) -> None:
        df = pd.DataFrame(history)
        df["epoch"] = list(range(1, len(history["loss"]) + 1))
        df.to_csv(out_path, index=False)

