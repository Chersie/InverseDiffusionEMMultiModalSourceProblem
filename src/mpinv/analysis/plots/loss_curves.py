"""Loss-curve plot.

Stable contract: ``build_loss_curves_figure(history, **opts) -> matplotlib.Figure``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def build_loss_curves_figure(
    history: Mapping[str, Iterable[tuple[int, float]]],
    title: str = "Training curves",
    log_y: bool = True,
) -> Figure:
    """Plot one curve per series in ``history``.

    Parameters
    ----------
    history : mapping ``name -> [(step, value), ...]``
    title : figure title.
    log_y : whether to use a log y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, points in history.items():
        pts = list(points)
        if not pts:
            continue
        xs = [s for s, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, label=name, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
