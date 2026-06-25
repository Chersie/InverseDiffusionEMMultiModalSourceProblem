"""Field-space evaluation plot: P_pred / P_true / residual heatmaps."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _row_vmax(Pt: np.ndarray, Pp: np.ndarray, vmax_strategy: str) -> tuple[float, bool]:
    """Pick ``vmax`` for one (P_true, P_pred) pair and report if P_pred is clipped.

    Returns ``(vmax, clipped)``.

    Strategies:

    * ``"true_max"`` — use ``P_true.max()``. Truth is always fully visible;
      runaway predictions saturate the colorbar (which is informative).
      Default — chosen so that catastrophically off-scale predictions never
      visually erase a small truth.
    * ``"shared_max"`` — legacy: ``max(P_pred, P_true)``. Lets you compare
      magnitudes directly when both panels are in the same regime, but
      collapses ``P_true`` to black when ``P_pred >> P_true``.
    * ``"pred_max"`` — use ``P_pred.max()``. Useful only when ``P_pred`` is
      the trustworthy reference (rarely).
    """
    pt_max = float(max(Pt.max(), 1e-30))
    pp_max = float(max(Pp.max(), 1e-30))
    if vmax_strategy == "shared_max":
        return float(max(pt_max, pp_max)), False
    if vmax_strategy == "pred_max":
        return pp_max, pt_max > pp_max * 1.001
    # default: "true_max"
    return pt_max, pp_max > pt_max * 1.001


def build_field_comparison_figure(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    sample_idx: int = 0,
    title: str | None = None,
    vmax_strategy: str = "true_max",
) -> Figure:
    """Side-by-side ``P_pred``, ``P_true``, and signed residual.

    Parameters
    ----------
    P_pred, P_true : np.ndarray
        Real arrays of shape ``(B, n_theta, n_phi)``.
    sample_idx : int
        Which batch row to plot.
    vmax_strategy : str
        See :func:`_row_vmax`. Default ``"true_max"`` keeps ``P_true`` always
        visible; ``"shared_max"`` reproduces the pre-2026-05-13 behaviour.
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    Pp = P_pred[sample_idx]
    Pt = P_true[sample_idx]
    res = Pp - Pt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    vmax, clipped = _row_vmax(Pt, Pp, vmax_strategy)
    im0 = axes[0].imshow(Pt, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    pt_max_str = f"max={Pt.max():.3g}"
    axes[0].set_title(f"P_true ({pt_max_str})")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(Pp, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    pp_max_str = f"max={Pp.max():.3g}"
    axes[1].set_title(
        f"P_pred ({pp_max_str})" + (" — clipped to vmax" if clipped else "")
    )
    plt.colorbar(im1, ax=axes[1], shrink=0.85)
    # Residual on the same scale as the P_true / P_pred panels: the diverging
    # colormap saturates exactly when |P_pred - P_true| exceeds vmax.
    res_amp_actual = float(max(np.abs(res).max(), 1e-30))
    res_clipped = res_amp_actual > vmax * 1.001
    im2 = axes[2].imshow(res, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    res_title = f"residual (P_pred - P_true), |max|={res_amp_actual:.3g}"
    if res_clipped:
        res_title += f" — clipped to ±{vmax:.3g}"
    axes[2].set_title(res_title)
    plt.colorbar(im2, ax=axes[2], shrink=0.85)
    for ax in axes:
        ax.set_xlabel(r"$\varphi$ index")
        ax.set_ylabel(r"$\theta$ index")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def build_field_comparison_grid_figure(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    *,
    sample_indices: Sequence[int] | None = None,
    n_samples: int = 8,
    sample_ids: Sequence[str] | None = None,
    per_sample_nrmse: Sequence[float] | None = None,
    per_sample_metric: Sequence[float] | None = None,
    metric_label: str = "NRMSE",
    metric_fmt: str = "{:.3g}",
    title: str | None = None,
    vmax_strategy: str = "true_max",
) -> Figure:
    """Multi-row variant of :func:`build_field_comparison_figure`.

    Each row shows one sample's ``P_true``, ``P_pred``, and signed residual
    side-by-side, with a per-row ``vmax`` so high-dynamic-range samples don't
    wash out the others. Use this when you want to inspect the model's
    behaviour across a representative slice of a split rather than picking
    a single sample.

    Parameters
    ----------
    P_pred, P_true : np.ndarray
        Real arrays of shape ``(B, n_theta, n_phi)``.
    sample_indices : sequence of int, optional
        Specific batch rows to plot. If ``None``, the first ``n_samples`` rows
        are taken (capped at ``B``).
    n_samples : int
        Number of rows to draw if ``sample_indices`` is not given.
    sample_ids : sequence of str, optional
        Optional human-readable IDs to label each row.
    per_sample_nrmse : sequence of float, optional
        **Deprecated**, kept for backward compatibility — same as passing
        ``per_sample_metric`` with ``metric_label="NRMSE"``.
    per_sample_metric : sequence of float, optional
        Per-sample scalar to display alongside each row label. Generic — pass
        whatever metric you want (NRMSE, R², per-sample MSE, etc.) and
        annotate the meaning via ``metric_label`` / ``metric_fmt``.
    metric_label : str
        Short label rendered before the metric value (e.g. ``"R²"``).
    metric_fmt : str
        Python-format spec for the metric value, e.g. ``"{:.3f}"``.
    title : str, optional
        Overall figure title.
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    B = P_pred.shape[0]
    if B == 0:
        raise ValueError("empty batch: nothing to plot")
    if sample_indices is None:
        n = min(n_samples, B)
        sample_indices = list(range(n))
    else:
        sample_indices = [int(i) for i in sample_indices]
    n = len(sample_indices)
    if n == 0:
        raise ValueError("sample_indices is empty")
    # Accept the legacy `per_sample_nrmse` arg as a synonym for the new generic
    # `per_sample_metric`.
    if per_sample_metric is None and per_sample_nrmse is not None:
        per_sample_metric = per_sample_nrmse
        metric_label = "NRMSE"
    fig, axes = plt.subplots(
        n, 3, figsize=(11, 3.0 * n), squeeze=False
    )
    # Store image objects for global colorbars
    im_t_list, im_p_list, im_r_list = [], [], []
    for row, idx in enumerate(sample_indices):
        Pt = P_true[idx]
        Pp = P_pred[idx]
        res = Pp - Pt
        vmax, clipped = _row_vmax(Pt, Pp, vmax_strategy)
        # Residual on the same scale as the P_true / P_pred panels — the
        # diverging colormap saturates exactly when |P_pred - P_true|
        # exceeds vmax. This keeps the residual visually comparable to the
        # data magnitude in the other two panels and avoids the case where
        # an off-scale prediction makes its own residual the only visible
        # feature (with everything else collapsed near zero).
        res_amp_actual = float(max(np.abs(res).max(), 1e-30))
        res_clipped = res_amp_actual > vmax * 1.001
        ax_t, ax_p, ax_r = axes[row]
        im_t = ax_t.imshow(Pt, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        im_p = ax_p.imshow(Pp, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        im_r = ax_r.imshow(
            res, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        im_t_list.append(im_t)
        im_p_list.append(im_p)
        im_r_list.append(im_r)
        if row == 0:
            ax_t.set_title(f"P_true (vmax={vmax_strategy})")
            ax_p.set_title("P_pred")
            ax_r.set_title("residual (P_pred - P_true)  [±vmax]")
        # Per-row max annotations on the column titles for the *second* row
        # would clutter every PDF; instead, embed compact stats in the row
        # label below alongside the metric.
        label = f"#{idx}"
        if sample_ids is not None and 0 <= idx < len(sample_ids):
            label = f"{sample_ids[idx]}"
        if per_sample_metric is not None and 0 <= idx < len(per_sample_metric):
            try:
                value_str = metric_fmt.format(float(per_sample_metric[idx]))
            except (TypeError, ValueError):
                value_str = str(per_sample_metric[idx])
            label = f"{label}\n{metric_label}={value_str}"
        # Always show the per-row absolute scales so off-scale predictions are
        # impossible to miss in the figure.
        scale_note = (
            f"\nmax(P_true)={float(Pt.max()):.2g}\n"
            f"max(P_pred)={float(Pp.max()):.2g}\n"
            f"max|res|={res_amp_actual:.2g}"
        )
        flags = []
        if clipped:
            flags.append("P_pred clipped")
        if res_clipped:
            flags.append("res clipped")
        if flags:
            scale_note += "\n(" + ", ".join(flags) + ")"
        label = f"{label}{scale_note}"
        ax_t.set_ylabel(label, fontsize=8)
        for ax in (ax_t, ax_p, ax_r):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[-1, 0].set_xlabel(r"$\varphi$ index")
    axes[-1, 1].set_xlabel(r"$\varphi$ index")
    axes[-1, 2].set_xlabel(r"$\varphi$ index")
    if title:
        fig.suptitle(title)
    # Global colorbars: one per column type, placed at the figure level
    # Use the first row's image as the reference for each colorbar
    if im_t_list and im_p_list and im_r_list:
        fig.colorbar(im_t_list[0], ax=axes[:, 0].tolist(), shrink=0.85, location="bottom")
        fig.colorbar(im_p_list[0], ax=axes[:, 1].tolist(), shrink=0.85, location="bottom")
        fig.colorbar(im_r_list[0], ax=axes[:, 2].tolist(), shrink=0.85, location="bottom")
    # fig.tight_layout()
    return fig
