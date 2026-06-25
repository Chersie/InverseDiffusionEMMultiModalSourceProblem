"""Scatter plot of predicted vs target packed coefficients."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def build_coef_scatter_figure(
    pred: np.ndarray,
    target: np.ndarray,
    title: str = "Predicted vs target coefficients",
    n_max: int = 20000,
) -> Figure:
    """One scatter per packed-block (Re a^E, Im a^E, Re a^M, Im a^M).

    Parameters
    ----------
    pred, target : np.ndarray
        Real arrays of shape ``(B, 4 K)`` in canonical packed layout.
    n_max : int
        Sub-sample the joint set if ``B * 4 K`` exceeds this.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    K4 = pred.shape[-1]
    if K4 % 4 != 0:
        raise ValueError("packed dim must be 4 K")
    K = K4 // 4
    block_names = ("Re aE", "Im aE", "Re aM", "Im aM")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=False, sharey=False)
    for i, name in enumerate(block_names):
        ax = axes[i // 2, i % 2]
        p = pred[:, i * K : (i + 1) * K].ravel()
        t = target[:, i * K : (i + 1) * K].ravel()
        if p.size > n_max:
            idx = np.random.default_rng(0).choice(p.size, n_max, replace=False)
            p, t = p[idx], t[idx]
        lo = float(min(p.min(), t.min()))
        hi = float(max(p.max(), t.max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        ax.scatter(t, p, s=2, alpha=0.4)
        ax.set_xlabel(f"target {name}")
        ax.set_ylabel(f"predicted {name}")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
