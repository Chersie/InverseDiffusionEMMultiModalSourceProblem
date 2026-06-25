"""Single-mode (``dummy_*``) probe diagnostic.

Plot the predicted packed coefficients for inputs constructed from a single active
``(l, m, family)`` mode. A correct model should localise its prediction to the
active mode (perhaps with the §1.7 reflected-conjugate ambiguity allowed).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def build_dummy_probe_figure(
    pred_packed: np.ndarray,
    active_indices: list[int],
    title: str = "Single-mode probe response",
) -> Figure:
    """Show ``|pred|`` per coefficient with the active indices highlighted.

    Parameters
    ----------
    pred_packed : np.ndarray
        Real array of shape ``(B, 4 K)``; row ``i`` corresponds to the input that
        had only ``active_indices[i]`` set.
    active_indices : list[int]
        Indices into the packed coefficient vector that were active.
    """
    if pred_packed.shape[0] != len(active_indices):
        raise ValueError("pred and active_indices must have matching length")
    n = pred_packed.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.4 * n + 1), sharex=True, squeeze=False)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        magnitudes = np.abs(pred_packed[i])
        ax.bar(np.arange(magnitudes.size), magnitudes, color="C0", alpha=0.7)
        ax.axvline(active_indices[i], color="C3", linewidth=1)
        ax.set_ylabel(f"trial {i}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("packed coefficient index")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
