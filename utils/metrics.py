from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    total_consumption = float(np.sum(y_true))
    avg_consumption = float(np.mean(y_true))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "total_energy_kwh": total_consumption,
        "avg_energy_kwh": avg_consumption,
    }


def detect_anomalies_zscore(
    data: np.ndarray, threshold: float = 3.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect anomalies using Z-score."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data, dtype=bool), {"mean": mean, "std": std}

    z_scores = np.abs((data - mean) / std)
    anomalies = z_scores > threshold

    info = {
        "mean": float(mean),
        "std": float(std),
        "count": int(np.sum(anomalies)),
        "percentage": float(np.mean(anomalies) * 100),
    }
    return anomalies, info


def detect_anomalies_iqr(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect anomalies using Interquartile Range."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    anomalies = (data < lower_bound) | (data > upper_bound)

    info = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "count": int(np.sum(anomalies)),
        "percentage": float(np.mean(anomalies) * 100),
    }
    return anomalies, info


def inject_noise(
    data: np.ndarray, noise_level: float = 0.5, sample_fraction: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject Gaussian noise into a fraction of the data to simulate anomalies.
    Returns (noisy_data, injected_mask).
    """
    noisy_data = data.copy()
    n_samples = len(data)
    n_noise = int(n_samples * sample_fraction)

    if n_noise == 0:
        return noisy_data, np.zeros(n_samples, dtype=bool)

    # Randomly select indices to inject noise
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    # Calculate noise scale based on data distribution
    std = np.std(data)
    if std == 0: std = 1.0
    
    # Add noise (spikes) to selected indices
    # We use 3 * std * noise_level to make it a clear but not extreme anomaly
    noise = np.random.normal(loc=std * 3, scale=std * noise_level, size=n_noise)
    noisy_data[noise_indices] += noise
    
    injected_mask = np.zeros(n_samples, dtype=bool)
    injected_mask[noise_indices] = True
    
    return noisy_data, injected_mask


def calculate_detection_robustness(
    injected_mask: np.ndarray, detected_mask: np.ndarray
) -> Dict[str, float]:
    """Calculate how well the detector caught the injected anomalies."""
    if np.sum(injected_mask) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = np.sum(injected_mask & detected_mask)
    false_positives = np.sum((~injected_mask) & detected_mask)
    false_negatives = np.sum(injected_mask & (~detected_mask))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "caught_count": int(true_positives),
        "injected_count": int(np.sum(injected_mask)),
    }

