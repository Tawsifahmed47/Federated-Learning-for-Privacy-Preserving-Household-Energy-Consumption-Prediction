from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (
    calculate_detection_robustness,
    detect_anomalies_iqr,
    detect_anomalies_zscore,
    inject_noise,
)


class FLClient:
    def __init__(
        self,
        client_id: str,
        data_loader: DataLoader,
        model_fn,
        device: torch.device,
        learning_rate: float,
        local_epochs: int,
        malicious: bool = False,
        malicious_probability: float = 0.0,
        enable_dp: bool = False,
        dp_noise_std: float = 0.0,
        anomaly_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client_id = client_id
        self.data_loader = data_loader
        self.model_fn = model_fn
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.criterion = nn.MSELoss()
        self.malicious = malicious
        self.malicious_probability = malicious_probability
        self.enable_dp = enable_dp
        self.dp_noise_std = dp_noise_std
        self.anomaly_cfg = anomaly_cfg or {"enabled": False}

    def _is_malicious_round(self) -> bool:
        if not self.malicious:
            return False
        return torch.rand(1).item() < self.malicious_probability

    def _randomize_state(self, state_dict: OrderedDict) -> OrderedDict:
        random_state = OrderedDict()
        for k, v in state_dict.items():
            random_state[k] = torch.randn_like(v)
        return random_state

    def _preprocess_local_data(self) -> DataLoader:
        """Filter anomalies if configured."""
        if (
            not self.anomaly_cfg.get("enabled", False)
            or self.anomaly_cfg.get("action") != "filter"
        ):
            return self.data_loader

        dataset = self.data_loader.dataset
        x, y = dataset.tensors
        y_np = y.numpy().flatten()

        method = self.anomaly_cfg.get("method", "zscore")
        threshold = self.anomaly_cfg.get("threshold", 3.0)

        if method == "zscore":
            anomalies, _ = detect_anomalies_zscore(y_np, threshold)
        elif method == "iqr":
            anomalies, _ = detect_anomalies_iqr(y_np)
        else:
            anomalies = torch.zeros_like(y, dtype=torch.bool).numpy().flatten()

        keep_mask = ~anomalies
        x_filtered = x[keep_mask]
        y_filtered = y[keep_mask]

        return DataLoader(
            TensorDataset(x_filtered, y_filtered),
            batch_size=self.data_loader.batch_size,
            shuffle=True,
        )

    def test_anomaly_robustness(self) -> Dict[str, float]:
        """
        Test the detector's capability by injecting noise and checking if it catches it.
        Follows the user request: detect -> inject noise -> detect again.
        """
        robustness_cfg = self.anomaly_cfg.get("robustness_test", {})
        if not robustness_cfg.get("enabled", False):
            return {}

        dataset = self.data_loader.dataset
        _, y = dataset.tensors
        y_np = y.numpy().flatten()

        method = self.anomaly_cfg.get("method", "zscore")
        threshold = self.anomaly_cfg.get("threshold", 3.0)

        # 1. Initial detection (Normal state)
        if method == "zscore":
            initial_anomalies, _ = detect_anomalies_zscore(y_np, threshold)
        else:
            initial_anomalies, _ = detect_anomalies_iqr(y_np)

        # 2. Inject noise (Simulate abnormal behavior)
        noise_level = robustness_cfg.get("noise_level", 0.5)
        sample_fraction = robustness_cfg.get("sample_fraction", 0.1)
        noisy_y, injected_mask = inject_noise(y_np, noise_level, sample_fraction)

        # 3. Detect again (Abnormal state)
        if method == "zscore":
            post_noise_anomalies, _ = detect_anomalies_zscore(noisy_y, threshold)
        else:
            post_noise_anomalies, _ = detect_anomalies_iqr(noisy_y)

        # 4. Calculate robustness (How many of the INJECTED ones were caught)
        # We only care about the ones we explicitly injected to see if the detector is sensitive enough
        results = calculate_detection_robustness(injected_mask, post_noise_anomalies)
        
        return results

    def train(
        self,
        global_state: OrderedDict,
        algorithm: str,
        mu: float = 0.0,
        c_global: Optional[Dict[str, torch.Tensor]] = None,
        c_local: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[OrderedDict, int, float, Optional[Dict[str, torch.Tensor]]]:
        model = self.model_fn().to(self.device)
        model.load_state_dict(deepcopy(global_state))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Filter data if needed
        active_loader = self._preprocess_local_data()

        global_params = {k: v.detach().clone().to(self.device) for k, v in global_state.items()}
        if c_global is None:
            c_global = {k: torch.zeros_like(v).to(self.device) for k, v in global_state.items()}
        if c_local is None:
            c_local = {k: torch.zeros_like(v).to(self.device) for k, v in global_state.items()}

        total_loss = 0.0
        total_batches = 0
        steps = 0
        initial_state = deepcopy(model.state_dict())

        model.train()
        for _ in range(self.local_epochs):
            for x_batch, y_batch in active_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = self.criterion(preds, y_batch)

                if algorithm == "fedprox":
                    prox = 0.0
                    for name, param in model.named_parameters():
                        prox += torch.sum((param - global_params[name]) ** 2)
                    loss = loss + (mu / 2.0) * prox

                loss.backward()

                if self.enable_dp and self.dp_noise_std > 0.0:
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.grad is not None:
                                noise = torch.normal(
                                    mean=0.0,
                                    std=self.dp_noise_std,
                                    size=param.grad.shape,
                                    device=param.grad.device,
                                )
                                param.grad.add_(noise)

                if algorithm == "scaffold":
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            param.grad = param.grad + c_local[name] - c_global[name]

                optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1
                steps += 1

        new_state = model.state_dict()
        avg_loss = total_loss / max(total_batches, 1)

        if self._is_malicious_round():
            return self._randomize_state(new_state), len(active_loader.dataset), avg_loss, None

        c_local_new = None
        if algorithm == "scaffold":
            c_local_new = {}
            lr = self.learning_rate
            for k in new_state.keys():
                delta = initial_state[k].to(self.device) - new_state[k].to(self.device)
                c_local_new[k] = c_local[k] - c_global[k] + (delta / max(steps * lr, 1e-12))

        return new_state, len(active_loader.dataset), avg_loss, c_local_new

