"""Per-run analysis report.

Given numpy arrays from a finished run (or freshly evaluated), builds a fixed set
of figures and metrics and writes them to ``output_dir`` plus an optional
``MLflowSink`` for artifact logging.

Two entry points:

- :func:`build_run_report` — the legacy single-split path that
  :mod:`scripts.run_real_augmented` uses via ``_figures_for(out_root, tag, ...)``;
  unchanged signature.
- :func:`build_split_report` — per-split report used by ``mpinv-train``: writes
  to ``<output_dir>/<split>/`` and emits an additional ``field_comparison_grid.pdf``
  (worst-to-best ranked by per-sample R-squared). When ``dummy_active_indices`` is
  provided, additionally emits ``dummy_probe.pdf`` and skips the degenerate
  ``coef_histograms.pdf``. Returns both the aggregated metrics dict and the
  per-sample arrays (R-squared, bin accuracy, Spearman rho, NRMSE, coef-MSE) so
  the caller can build cross-split distribution figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from mpinv.analysis.metrics.coefficient_metrics import (
    packed_mse,
    packed_r2,
    per_sample_packed_mse,
)
from mpinv.analysis.metrics.field_metrics import (
    bin_accuracy_P,
    per_sample_bin_accuracy_P,
    per_sample_spearman_rho_P,
    per_sample_weighted_nrmse_P,
    per_sample_weighted_r2_P,
    spearman_rho_P,
    weighted_mse_P,
    weighted_nrmse_P,
)
from mpinv.analysis.metrics.mode_metrics import reflected_conjugate_aware_loss
from mpinv.analysis.plots.coef_histograms import build_coef_histograms_figure
from mpinv.analysis.plots.coef_scatter import build_coef_scatter_figure
from mpinv.analysis.plots.dummy_probe import build_dummy_probe_figure
from mpinv.analysis.plots.feature_importance_pca import build_pca_explained_variance_figure
from mpinv.analysis.plots.field_comparison import (
    build_field_comparison_figure,
    build_field_comparison_grid_figure,
)
from mpinv.analysis.plots.per_l_breakdown import build_per_l_breakdown_figure
from mpinv.core.grid import GRID_DEFAULT, GridSpec


@dataclass(slots=True)
class RunArtifacts:
    """Inputs the report builder needs from a finished run."""

    pred_packed: np.ndarray  # (N, 4 K)
    target_packed: np.ndarray  # (N, 4 K)
    P_pred: np.ndarray  # (N, n_theta, n_phi)
    P_true: np.ndarray  # (N, n_theta, n_phi)
    l_max: int
    grid: GridSpec = GRID_DEFAULT
    pca_explained_variance_ratio: np.ndarray | None = None


def build_run_report(
    art: RunArtifacts,
    output_dir: str | Path,
    sink: Any | None = None,
) -> dict[str, float]:
    """Build the standard per-run figures + metrics.

    Returns a metric dictionary (metrics are also logged via ``sink`` if given).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Figure] = {
        "coef_histograms.pdf": build_coef_histograms_figure(art.target_packed),
        "coef_scatter.pdf": build_coef_scatter_figure(art.pred_packed, art.target_packed),
        "per_l_breakdown.pdf": build_per_l_breakdown_figure(
            art.pred_packed, art.target_packed, art.l_max
        ),
        "field_comparison.pdf": build_field_comparison_figure(art.P_pred, art.P_true),
    }
    if art.pca_explained_variance_ratio is not None:
        figs["pca_explained_variance.pdf"] = build_pca_explained_variance_figure(
            art.pca_explained_variance_ratio
        )
    for name, fig in figs.items():
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        if sink is not None:
            try:
                sink.log_figure(fig, f"plots/{name}")
            except Exception:
                pass
        # Release pyplot's reference so figures don't accumulate when this
        # report is built many times in one process (e.g. the per-epoch
        # figures callback in scripts/run_real_augmented.py).
        plt.close(fig)

    metrics = {
        "report/coef_mse": packed_mse(art.pred_packed, art.target_packed),
        "report/coef_r2": packed_r2(art.pred_packed, art.target_packed),
        "report/coef_mse_amb_aware": reflected_conjugate_aware_loss(
            art.pred_packed, art.target_packed, art.l_max
        ),
        "report/field_mse_w": weighted_mse_P(art.P_pred, art.P_true, grid=art.grid),
        "report/field_nrmse_w": weighted_nrmse_P(art.P_pred, art.P_true, grid=art.grid),
    }
    if sink is not None:
        try:
            sink.log_metrics(metrics)
        except Exception:
            pass
    return metrics


def _pick_grid_indices(per_sample_r2: np.ndarray, n_grid: int) -> list[int]:
    """Pick worst/middle/best samples sorted by per-sample R² ascending.

    Mirrors the S5 selection rule (``scripts.run_real_augmented._emit_all_figures``):
    one third worst, one third best (reversed), one third middle band, top up
    with random remaining if we still have room. Returns indices sorted by
    R² ascending so the rendered grid reads worst → best top-to-bottom.
    """
    B = per_sample_r2.shape[0]
    n_grid = max(1, min(n_grid, B))
    order = np.argsort(per_sample_r2)
    n_third = max(1, n_grid // 3)
    picks: list[int] = []
    picks.extend(int(i) for i in order[:n_third])
    picks.extend(int(i) for i in order[-n_third:][::-1])
    mid = len(order) // 2
    mid_band = order[max(0, mid - n_third // 2):
                     min(len(order), mid - n_third // 2 + n_third)]
    picks.extend(int(i) for i in mid_band if int(i) not in picks)
    if len(picks) < n_grid:
        remaining = [int(i) for i in range(B) if int(i) not in picks]
        rng = np.random.default_rng(0)
        if remaining:
            extra = rng.choice(
                len(remaining), size=min(n_grid - len(picks), len(remaining)),
                replace=False,
            )
            picks.extend(int(remaining[int(p)]) for p in extra)
    picks = picks[:n_grid]
    picks.sort(key=lambda i: float(per_sample_r2[i]))
    return picks


def build_split_report(
    art: RunArtifacts,
    output_dir: str | Path,
    split: str,
    *,
    sink: Any | None = None,
    n_grid_samples: int = 8,
    dummy_active_indices: list[int] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Per-split report: 4 PDFs + worst-to-best ``field_comparison_grid.pdf``.

    Outputs land in ``<output_dir>/<split>/``. Files emitted:

    - ``coef_scatter.pdf`` — pred vs target packed scatter.
    - ``per_l_breakdown.pdf`` — per-l error decomposition.
    - ``field_comparison_grid.pdf`` — ``n_grid_samples`` rows ranked worst→best
      by per-sample sin-θ-weighted R².
    - ``coef_histograms.pdf`` — marginal histograms of target coefficients.
      Skipped when ``dummy_active_indices`` is given (target is one-hot —
      degenerate).
    - ``dummy_probe.pdf`` — only when ``dummy_active_indices`` is given:
      ``|pred|`` per packed slot with the active slot highlighted, mirroring
      :func:`mpinv.analysis.plots.dummy_probe.build_dummy_probe_figure`.

    Returns
    -------
    metrics : dict[str, float]
        Aggregated metrics keyed ``report/<split>/<metric>`` so the caller
        can union them straight into ``metrics.json``.
    per_sample : dict[str, np.ndarray]
        ``{"r2", "bin_accuracy", "spearman_rho", "nrmse", "coef_mse"}``
        per-sample arrays of shape ``(B,)``. ``coef_mse`` is over packed
        coefficients; the field metrics are sin-θ-weighted on P. Used by the
        cli/train.py loop to build cross-split violin/histogram figures.
    """
    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    is_dummy = dummy_active_indices is not None
    figs: dict[str, Figure] = {
        "coef_scatter.pdf": build_coef_scatter_figure(art.pred_packed, art.target_packed),
        "per_l_breakdown.pdf": build_per_l_breakdown_figure(
            art.pred_packed, art.target_packed, art.l_max
        ),
    }
    if not is_dummy:
        figs["coef_histograms.pdf"] = build_coef_histograms_figure(art.target_packed)

    # Per-sample metrics (computed once, reused for ranking + the caller's
    # cross-split distribution figures).
    sample_r2 = per_sample_weighted_r2_P(art.P_pred, art.P_true, grid=art.grid)
    n_bins_metric = 2 * art.l_max + 1
    sample_bin_acc = per_sample_bin_accuracy_P(art.P_pred, art.P_true, n_bins_metric)
    sample_rho = per_sample_spearman_rho_P(art.P_pred, art.P_true)
    sample_nrmse = per_sample_weighted_nrmse_P(art.P_pred, art.P_true, grid=art.grid)
    sample_coef_mse = per_sample_packed_mse(art.pred_packed, art.target_packed)

    # Worst-to-best grid figure.
    if art.P_pred.shape[0] > 0:
        picks = _pick_grid_indices(sample_r2, n_grid_samples)
        figs["field_comparison_grid.pdf"] = build_field_comparison_grid_figure(
            art.P_pred,
            art.P_true,
            sample_indices=picks,
            per_sample_metric=sample_r2.tolist(),
            metric_label="R²",
            metric_fmt="{:+.3f}",
            title=(
                f"{split}: {len(picks)} samples ranked by sin-θ-weighted R² "
                f"(rows top→bottom: worst → best)"
            ),
        )
    else:
        # Still emit a single-sample placeholder so the directory layout is
        # uniform across splits.
        figs["field_comparison.pdf"] = build_field_comparison_figure(art.P_pred, art.P_true)

    if is_dummy:
        figs["dummy_probe.pdf"] = build_dummy_probe_figure(
            art.pred_packed,
            list(dummy_active_indices),
            title=f"{split}: single-mode probe response (|pred| per packed slot)",
        )

    if art.pca_explained_variance_ratio is not None and split == "val":
        figs["pca_explained_variance.pdf"] = build_pca_explained_variance_figure(
            art.pca_explained_variance_ratio
        )

    for name, fig in figs.items():
        path = out_dir / name
        fig.savefig(path, bbox_inches="tight")
        if sink is not None:
            try:
                sink.log_figure(fig, f"plots/{split}/{name}")
            except Exception:
                pass
        plt.close(fig)

    metrics = {
        f"report/{split}/coef_mse": packed_mse(art.pred_packed, art.target_packed),
        f"report/{split}/coef_r2": packed_r2(art.pred_packed, art.target_packed),
        f"report/{split}/coef_mse_amb_aware": reflected_conjugate_aware_loss(
            art.pred_packed, art.target_packed, art.l_max
        ),
        f"report/{split}/field_mse_w": weighted_mse_P(
            art.P_pred, art.P_true, grid=art.grid
        ),
        f"report/{split}/field_nrmse_w": weighted_nrmse_P(
            art.P_pred, art.P_true, grid=art.grid
        ),
        f"report/{split}/spearman_rho_P": spearman_rho_P(art.P_pred, art.P_true),
        f"report/{split}/bin_accuracy_P": bin_accuracy_P(
            art.P_pred, art.P_true, n_bins_metric
        ),
    }
    if sink is not None:
        try:
            sink.log_metrics(metrics)
        except Exception:
            pass

    per_sample: dict[str, np.ndarray] = {
        "r2": np.asarray(sample_r2, dtype=np.float64),
        "bin_accuracy": np.asarray(sample_bin_acc, dtype=np.float64),
        "spearman_rho": np.asarray(sample_rho, dtype=np.float64),
        "nrmse": np.asarray(sample_nrmse, dtype=np.float64),
        "coef_mse": np.asarray(sample_coef_mse, dtype=np.float64),
    }
    return metrics, per_sample
