"""Analysis suite: plots, reports, metrics."""

from mpinv.analysis.plots.coef_histograms import build_coef_histograms_figure
from mpinv.analysis.plots.coef_scatter import build_coef_scatter_figure
from mpinv.analysis.plots.dummy_probe import build_dummy_probe_figure
from mpinv.analysis.plots.feature_importance_pca import build_pca_explained_variance_figure
from mpinv.analysis.plots.field_comparison import build_field_comparison_figure
from mpinv.analysis.plots.loss_curves import build_loss_curves_figure
from mpinv.analysis.plots.per_l_breakdown import build_per_l_breakdown_figure

__all__ = [
    "build_coef_histograms_figure",
    "build_coef_scatter_figure",
    "build_dummy_probe_figure",
    "build_field_comparison_figure",
    "build_loss_curves_figure",
    "build_pca_explained_variance_figure",
    "build_per_l_breakdown_figure",
]
