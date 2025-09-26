"""Bootstrap utilities for metrics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    *,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(data.mean())
    if data.size == 1:
        return mean, mean, mean

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, data.size, size=(n_samples, data.size))
    samples = data[indices].mean(axis=1)

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return mean, lower, upper


__all__ = ["bootstrap_mean_ci"]
