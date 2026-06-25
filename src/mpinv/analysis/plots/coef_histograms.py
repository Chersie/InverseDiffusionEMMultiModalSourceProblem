"""Per-block histogram of packed coefficients (data-gen diagnostic)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def build_coef_histograms_figure(packed: np.ndarray, bins: int = 60) -> Figure:
    """Histograms of the four packed blocks Re aE, Im aE, Re aM, Im aM."""
    if packed.ndim != 2 or packed.shape[1] % 4 != 0:
        raise ValueError(f"expected (B, 4 K), got {packed.shape}")
    K = packed.shape[1] // 4
    block_names = ("Re aE", "Im aE", "Re aM", "Im aM")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, name in enumerate(block_names):
        ax = axes[i // 2, i % 2]
        block = packed[:, i * K : (i + 1) * K].ravel()
        ax.hist(block, bins=bins, alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.set_title(f"{name} (mean={block.mean():.3f}, std={block.std():.3f})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
