"""PCA explained-variance plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def build_pca_explained_variance_figure(
    explained_variance_ratio: np.ndarray,
    title: str = "PCA explained variance",
) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cumulative = np.cumsum(explained_variance_ratio)
    x = np.arange(1, explained_variance_ratio.size + 1)
    ax.bar(x, explained_variance_ratio, alpha=0.7, label="per component")
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, "C1-", label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax2.set_ylabel("cumulative")
    ax2.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
