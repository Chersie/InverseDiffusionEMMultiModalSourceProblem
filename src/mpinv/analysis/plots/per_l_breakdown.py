"""Per-l error bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from mpinv.core.packing import iter_modes


def build_per_l_breakdown_figure(
    pred: np.ndarray,
    target: np.ndarray,
    l_max: int,
    title: str = "Per-l coefficient MSE",
) -> Figure:
    """Bar chart of mean-squared coefficient error per multipole order ``l``."""
    K = l_max * (l_max + 2)
    assert pred.shape[1] == 4 * K, f"expected 4K={4 * K}, got {pred.shape[1]}"
    diff_sq = (pred - target) ** 2
    per_l = np.zeros(l_max)
    counts = np.zeros(l_max)
    for block in range(4):
        sub = diff_sq[:, block * K : (block + 1) * K]
        for k, (l, _m) in enumerate(iter_modes(l_max)):
            per_l[l - 1] += float(sub[:, k].sum())
            counts[l - 1] += sub.shape[0]
    per_l_mse = per_l / np.maximum(counts, 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(1, l_max + 1), per_l_mse)
    ax.set_xlabel("l")
    ax.set_ylabel("MSE summed over (m, family, batch)")
    ax.set_title(title)
    ax.set_xticks(np.arange(1, l_max + 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
