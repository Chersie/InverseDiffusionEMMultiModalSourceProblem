"""Coefficient-space metrics."""

from __future__ import annotations

import numpy as np


def packed_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error in packed-coefficient space."""
    return float(((pred - target) ** 2).mean())


def per_sample_packed_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample MSE in packed-coefficient space.

    Returns a ``(B,)`` array where ``out[i] = mean_k (pred[i, k] - target[i, k])²``.
    The companion to :func:`packed_mse` for the violin/histogram distribution
    plots.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected (B, 4 K); got {pred.shape}")
    return ((pred.astype(np.float64) - target.astype(np.float64)) ** 2).mean(axis=1)


def packed_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination ``R^2`` over the flattened coefficient axis."""
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-18)
