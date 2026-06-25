"""Cross-split distribution plots for per-sample evaluation metrics.

This module hosts a generic distribution-plot builder
(:func:`build_metric_distribution_figure`) and two thin wrappers around it
for the two metrics that ship out-of-the-box:

* :func:`build_r2_distribution_figure` — sin-θ-weighted R² per sample
  (range ``(-∞, 1]``; ``-inf`` for samples whose target is constant).
* :func:`build_bin_accuracy_distribution_figure` — fraction of pixels
  whose hard quantile bin matches between ``P_pred`` and ``P_true``
  (range ``[0, 1]``).

Both figures share the same layout: a 2x2 (or 1xN) grid of histograms on
top — one panel per split — and a single violin/box plot below comparing
the same per-sample distributions side-by-side.

Why a single helper? Because adding more "distribution-of-X-across-splits"
figures (Spearman ρ, within-1-bin accuracy, etc.) should be a one-line
wrapper — see the pattern in the two wrappers at the bottom of this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_DEFAULT_PALETTE = (
    "#3a86ff",  # train_aug — blue
    "#06d6a0",  # val_real  — teal
    "#ffbe0b",  # holdout_real — amber
    "#fb5607",  # synthetic_test — orange
)


def _summary(values: np.ndarray, positive_threshold: float = 0.0) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "n_finite": int(finite.size),
        "n_below": int((values == -np.inf).sum()),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "frac_above_threshold": (
            float((finite > positive_threshold).mean()) if finite.size else 0.0
        ),
    }


def build_metric_distribution_figure(
    metric_per_split: Mapping[str, np.ndarray],
    *,
    metric_label: str,
    metric_short: str = "metric",
    clip_range: tuple[float, float] = (-3.0, 1.0),
    n_bins: int = 30,
    reference_lines: Sequence[tuple[float, str, str]] = (
        (0.0, "red", "predict-mean baseline"),
        (1.0, "green", "perfect"),
    ),
    positive_threshold: float = 0.0,
    title: str | None = None,
) -> Figure:
    """Generic per-split distribution plot.

    Parameters
    ----------
    metric_per_split : mapping
        ``{split_name: per_sample_metric_array}``. Mapping order is
        preserved as the plotting order (left-to-right, top-to-bottom in
        the histogram grid; left-to-right in the violin). Empty arrays are
        dropped from the figure.
    metric_label : str
        Long-form axis label (e.g. ``"R² (sin θ-weighted, per sample)"``).
    metric_short : str
        Short-form name used inside subtitles (e.g. ``"R²"``).
    clip_range : (lo, hi)
        Histogram x-axis range and violin y-axis range. Values below ``lo``
        (including ``-inf``) are bucketed into a "below clip" hatched bar
        at the far-left of the histogram and excluded from the violin.
    n_bins : int
        Number of histogram bins inside ``clip_range``.
    reference_lines : sequence of ``(value, color, label)``
        Vertical (in histogram) and horizontal (in violin) reference
        lines drawn at meaningful metric values. Defaults are calibrated
        for R²; use e.g. ``[(1/n_q, "red", "chance"), (1.0, "green", "perfect")]``
        for an accuracy-style metric.
    positive_threshold : float
        Threshold for the "P[metric > t]" annotation in subtitle. Defaults
        to ``0`` (matches R²); for accuracy metrics ``1/n_bins`` is more
        meaningful.
    title : str, optional
        Top-level figure title.
    """
    splits = [(name, np.asarray(arr).reshape(-1))
              for name, arr in metric_per_split.items()
              if np.asarray(arr).size > 0]
    if not splits:
        raise ValueError("metric_per_split is empty: nothing to plot")
    n_splits = len(splits)
    n_cols = min(2, n_splits)
    n_rows = (n_splits + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(11, 3.0 * n_rows + 4.0))
    grid = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2.0] * n_rows + [3.0])

    lo, hi = clip_range
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_w = float(bin_edges[1] - bin_edges[0]) if n_bins > 0 else 1.0
    palette = list(_DEFAULT_PALETTE) + [None] * max(0, n_splits - len(_DEFAULT_PALETTE))

    # Top: per-split histogram panels.
    for i, (name, values) in enumerate(splits):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(grid[r, c])
        s = _summary(values, positive_threshold=positive_threshold)
        finite = values[np.isfinite(values)]
        below = (values < lo) | (values == -np.inf)
        plotted = np.clip(finite[finite >= lo], lo, hi)
        if plotted.size:
            ax.hist(plotted, bins=bin_edges, color=palette[i], alpha=0.8,
                    edgecolor="black", linewidth=0.4)
        if int(below.sum()) > 0:
            ax.bar(
                lo - bin_w / 2.0,
                int(below.sum()),
                width=bin_w,
                color=palette[i], alpha=0.4, hatch="//",
                edgecolor="black", linewidth=0.4,
                label=f"<{lo:g}: {int(below.sum())}",
            )
        for value, color, label in reference_lines:
            ax.axvline(value, color=color, linestyle="--", linewidth=1.0,
                       label=label)
        if np.isfinite(s["median"]):
            ax.axvline(s["median"], color="black", linestyle="-",
                       linewidth=1.2, label=f"median={s['median']:.3f}")
        ax.set_xlim(lo - bin_w, hi)
        ax.set_xlabel(metric_label)
        ax.set_ylabel("count")
        ax.set_title(
            f"{name}  (n={s['n']}, median={s['median']:.3f}, "
            f"mean={s['mean']:.3f}, "
            f"P[{metric_short}>{positive_threshold:g}]="
            f"{s['frac_above_threshold']:.2f}, "
            f"below clip={s['n_below']})"
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    # Bottom: single violin / box panel comparing all splits.
    ax_v = fig.add_subplot(grid[n_rows, :])
    finite_per_split: list[np.ndarray] = []
    labels: list[str] = []
    for name, values in splits:
        finite = values[np.isfinite(values)]
        finite = finite[(finite >= lo) & (finite <= hi)]
        finite_per_split.append(finite)
        labels.append(name)
    positions = np.arange(1, len(splits) + 1)
    violin_positions: list[float] = []
    violin_data: list[np.ndarray] = []
    violin_palette: list[str] = []
    skipped: list[tuple[float, str, int]] = []
    for i, finite in enumerate(finite_per_split):
        if finite.size >= 2 and float(np.var(finite)) > 0:
            violin_positions.append(float(positions[i]))
            violin_data.append(finite)
            violin_palette.append(palette[i])
        else:
            skipped.append((float(positions[i]), labels[i], int(finite.size)))
    if violin_data:
        parts = ax_v.violinplot(
            violin_data, positions=violin_positions, showmedians=True,
            showextrema=False, widths=0.6,
        )
        for body, color in zip(parts["bodies"], violin_palette):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
    for i, finite in enumerate(finite_per_split):
        if finite.size:
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, size=finite.size)
            ax_v.scatter(
                positions[i] + jitter, finite, s=8, color=palette[i],
                edgecolor="black", linewidth=0.3, alpha=0.6,
            )
    y_annot = lo + 0.04 * (hi - lo)
    for x, name, n in skipped:
        msg = "no in-range samples" if n == 0 else (
            "1 in-range sample" if n == 1 else "no variance"
        )
        ax_v.annotate(
            f"({msg})", xy=(x, y_annot), ha="center", va="bottom",
            fontsize=8, color="grey",
        )
    for value, color, _label in reference_lines:
        ax_v.axhline(value, color=color, linestyle="--", linewidth=1.0,
                     alpha=0.7)
    ax_v.set_xticks(positions)
    ax_v.set_xticklabels(labels)
    ax_v.set_ylabel(metric_label)
    ax_v.set_ylim(lo, hi + 0.05)
    ref_line_summary = "; ".join(
        f"{c} dashed: {label}" for _v, c, label in reference_lines
    )
    ax_v.set_title(
        f"Per-sample {metric_short} distribution across splits"
        + (f"  ({ref_line_summary})" if ref_line_summary else "")
    )
    ax_v.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def build_r2_distribution_figure(
    r2_per_split: Mapping[str, np.ndarray],
    *,
    clip_range: tuple[float, float] = (-3.0, 1.0),
    n_bins: int = 30,
    title: str | None = None,
) -> Figure:
    """Backwards-compatible thin wrapper for the R² distribution figure."""
    return build_metric_distribution_figure(
        r2_per_split,
        metric_label="R² (sin θ-weighted, per sample)",
        metric_short="R²",
        clip_range=clip_range,
        n_bins=n_bins,
        reference_lines=(
            (0.0, "red", "R²=0 (predict-mean baseline)"),
            (1.0, "green", "R²=1 (perfect)"),
        ),
        positive_threshold=0.0,
        title=title,
    )


def build_bin_accuracy_distribution_figure(
    accuracy_per_split: Mapping[str, np.ndarray],
    *,
    n_bins_metric: int,
    clip_range: tuple[float, float] = (0.0, 1.0),
    n_bins: int = 30,
    title: str | None = None,
) -> Figure:
    """Distribution figure for per-sample hard rank-bin accuracy on P.

    Companion to the soft-rank-bin training loss in
    :mod:`mpinv.losses.rank_bin`. Reference lines mark chance level
    (``1 / n_bins_metric``) and perfect (``1.0``).

    Parameters
    ----------
    accuracy_per_split : mapping
        ``{split_name: per_sample_bin_accuracy_array}``. Each entry is the
        output of
        :func:`mpinv.analysis.metrics.field_metrics.per_sample_bin_accuracy_P`
        for that split.
    n_bins_metric : int
        Number of rank bins used by the metric (typically ``2 * l_max + 1``).
        Used to draw the chance-level reference line at ``1 / n_bins_metric``.
    clip_range : (lo, hi)
        Histogram x-axis. Bin accuracy is bounded in [0, 1] but you can
        clip tighter if all your splits land high.
    n_bins : int
        Histogram resolution.
    title : str, optional
        Top-level figure title.
    """
    chance = 1.0 / max(n_bins_metric, 1)
    return build_metric_distribution_figure(
        accuracy_per_split,
        metric_label="bin accuracy (per sample)",
        metric_short="bin acc",
        clip_range=clip_range,
        n_bins=n_bins,
        reference_lines=(
            (chance, "red", f"chance = 1/{n_bins_metric} ≈ {chance:.3f}"),
            (1.0, "green", "perfect"),
        ),
        positive_threshold=chance,
        title=title,
    )


def build_spearman_distribution_figure(
    rho_per_split: Mapping[str, np.ndarray],
    *,
    clip_range: tuple[float, float] = (-1.0, 1.0),
    n_bins: int = 30,
    title: str | None = None,
) -> Figure:
    """Distribution figure for per-sample Spearman rho on the power pattern.

    Reference lines mark zero correlation (``red``) and perfect rank
    agreement (``green``). Negative-rho samples (anti-correlation) appear in
    the lower half of the violin.

    Parameters
    ----------
    rho_per_split : mapping
        ``{split_name: per_sample_spearman_rho_P_array}``. Each entry is
        :func:`mpinv.analysis.metrics.field_metrics.per_sample_spearman_rho_P`
        applied to that split. NaN entries (degenerate-rank samples) are
        dropped from the violin and bucketed into the "below clip" bar.
    """
    return build_metric_distribution_figure(
        rho_per_split,
        metric_label="Spearman rho (per sample)",
        metric_short="rho",
        clip_range=clip_range,
        n_bins=n_bins,
        reference_lines=(
            (0.0, "red", "rho=0 (no rank correlation)"),
            (1.0, "green", "rho=1 (perfect rank agreement)"),
        ),
        positive_threshold=0.0,
        title=title,
    )


def build_nrmse_distribution_figure(
    nrmse_per_split: Mapping[str, np.ndarray],
    *,
    clip_range: tuple[float, float] = (0.0, 3.0),
    n_bins: int = 30,
    title: str | None = None,
) -> Figure:
    """Distribution figure for per-sample sin-θ-weighted NRMSE on P.

    NRMSE ≥ 0; lower is better. ``1.0`` means residuals on the order of the
    target, ``> 1.0`` is worse than predicting zero. Reference lines mark
    ``NRMSE = 1`` (predict-zero baseline, red) and ``0`` (perfect, green).

    Parameters
    ----------
    nrmse_per_split : mapping
        ``{split_name: per_sample_weighted_nrmse_P_array}``.
    """
    return build_metric_distribution_figure(
        nrmse_per_split,
        metric_label="NRMSE_w (per sample)",
        metric_short="NRMSE",
        clip_range=clip_range,
        n_bins=n_bins,
        reference_lines=(
            (0.0, "green", "NRMSE=0 (perfect)"),
            (1.0, "red", "NRMSE=1 (predict-zero baseline)"),
        ),
        positive_threshold=1.0,
        title=title,
    )


def build_coef_mse_distribution_figure(
    coef_mse_per_split: Mapping[str, np.ndarray],
    *,
    clip_range: tuple[float, float] = (0.0, 5.0),
    n_bins: int = 30,
    title: str | None = None,
) -> Figure:
    """Distribution figure for per-sample MSE in packed-coefficient space.

    Lower is better. Reference line marks ``coef_mse = 0`` (perfect). The
    upper clip is set generously by default to accommodate the
    high-variance dummy split (one-hot probes); tighten for production
    splits via ``clip_range`` if the violins compress.

    Parameters
    ----------
    coef_mse_per_split : mapping
        ``{split_name: per_sample_packed_mse_array}``.
    """
    return build_metric_distribution_figure(
        coef_mse_per_split,
        metric_label="coef MSE (per sample)",
        metric_short="coef MSE",
        clip_range=clip_range,
        n_bins=n_bins,
        reference_lines=(
            (0.0, "green", "coef_mse=0 (perfect)"),
        ),
        positive_threshold=0.0,
        title=title,
    )
