"""Evaluation statistics shared by the current tracked-policy pipeline."""

from __future__ import annotations

from typing import Dict

import numpy as np


def bootstrap_confidence_intervals(
    model_times: np.ndarray,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> Dict[str, float]:
    """Return percentile-bootstrap 95% intervals for the model mean."""
    count = len(model_times)
    if count == 0:
        raise ValueError("model evaluation must be non-empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(bootstrap_samples, count))
    model_means = model_times[indices].mean(axis=1)
    low, high = np.percentile(model_means, (2.5, 97.5))
    return {
        "model_mean_ci95_low": float(low),
        "model_mean_ci95_high": float(high),
    }


def interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) < 4:
        return float(ordered.mean())
    lower = int(np.floor(len(ordered) * 0.25))
    upper = int(np.ceil(len(ordered) * 0.75))
    return float(ordered[lower:upper].mean())


def lower_tail_mean(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, int(np.ceil(len(ordered) * float(fraction))))
    return float(ordered[:count].mean())


def wilson_lower_bound(successes: int, count: int, z: float = 1.96) -> float:
    """Return the lower bound of a two-sided Wilson score interval."""
    if count <= 0:
        return 0.0
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    margin = z * np.sqrt(
        probability * (1.0 - probability) / count
        + z * z / (4.0 * count * count)
    ) / denominator
    return float(max(0.0, center - margin))
