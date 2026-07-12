from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from utils.metrics import (
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    regression_metrics,
    inject_noise,
)


class Evaluator:
    @staticmethod
    def evaluate(
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        anomaly_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        metrics = regression_metrics(y_true, y_pred)

        if anomaly_cfg and anomaly_cfg.get("enabled", False):
            method = anomaly_cfg.get("method", "zscore")
            threshold = anomaly_cfg.get("threshold", 3.0)

            if method == "zscore":
                _, anomaly_info = detect_anomalies_zscore(y_true, threshold)
            elif method == "iqr":
                _, anomaly_info = detect_anomalies_iqr(y_true)
            else:
                anomaly_info = {}

            # Add anomaly metrics with prefix
            for k, v in anomaly_info.items():
                metrics[f"anomaly_{k}"] = v

        return metrics

    @staticmethod
    def save_prediction_plots(
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        output_dir: str,
        prefix: str,
        anomaly_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generate two plots: 1 day and 1 week.
        Each plot contains: Actual, Actual + Noise, and User Prediction.
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        y_pred = np.vstack(all_preds).flatten()
        y_true = np.vstack(all_targets).flatten()

        # Inject noise to create "Actual + Noise (Anomaly)"
        noise_level = 0.5
        sample_fraction = 0.1
        if anomaly_cfg and "robustness_test" in anomaly_cfg:
            noise_level = anomaly_cfg["robustness_test"].get("noise_level", 0.5)
            sample_fraction = anomaly_cfg["robustness_test"].get("sample_fraction", 0.1)
        
        y_noisy, _ = inject_noise(y_true, noise_level=noise_level, sample_fraction=sample_fraction)

        # 30-min intervals: 1 day = 48 points, 1 week = 336 points
        intervals = {"1day": 48, "1week": 336}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, count in intervals.items():
            if len(y_true) < count:
                continue
            
            # Take the first 'count' samples for the plot
            actual = y_true[:count]
            actual_noise = y_noisy[:count]
            prediction = y_pred[:count]
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual, label="Actual", color="blue", alpha=0.7)
            plt.plot(actual_noise, label="Actual + Noise (Anomaly)", color="red", linestyle="--", alpha=0.6)
            plt.plot(prediction, label="User Prediction", color="green", alpha=0.8)
            
            plt.title(f"Energy Consumption Prediction - {name} ({prefix})")
            plt.xlabel("Time (30-min intervals)")
            plt.ylabel("Energy (kWh)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = output_path / f"{prefix}_prediction_{name}.png"
            plt.savefig(plot_file)
            plt.close()

