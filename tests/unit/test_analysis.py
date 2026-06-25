"""Smoke tests for analysis plots and metrics."""

from __future__ import annotations

import numpy as np

from mpinv.analysis.metrics import (
    bin_accuracy_P,
    bin_within_k_accuracy_P,
    hard_rank_bin_mse_P,
    packed_mse,
    packed_r2,
    per_lm_mse,
    per_sample_spearman_rho_P,
    per_sample_weighted_r2_P,
    per_theta_band_error,
    reflected_conjugate_aware_loss,
    spearman_rho_P,
    weighted_mse_P,
    weighted_nrmse_P,
    weighted_r2_P,
)
from mpinv.analysis.plots.coef_histograms import build_coef_histograms_figure
from mpinv.analysis.plots.coef_scatter import build_coef_scatter_figure
from mpinv.analysis.plots.dummy_probe import build_dummy_probe_figure
from mpinv.analysis.plots.feature_importance_pca import build_pca_explained_variance_figure
from mpinv.analysis.plots.field_comparison import build_field_comparison_figure
from mpinv.analysis.plots.loss_curves import build_loss_curves_figure
from mpinv.analysis.plots.per_l_breakdown import build_per_l_breakdown_figure
from mpinv.analysis.plots.r2_distribution import build_r2_distribution_figure


def test_loss_curves_smoke():
    fig = build_loss_curves_figure({"train/loss": [(i, 1 / (i + 1)) for i in range(10)]})
    assert fig is not None


def test_coef_scatter_smoke():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((4, 24)).astype(np.float32)
    target = pred + rng.standard_normal((4, 24)).astype(np.float32) * 0.1
    fig = build_coef_scatter_figure(pred, target)
    assert fig is not None


def test_coef_histograms_smoke():
    fig = build_coef_histograms_figure(np.random.default_rng(0).standard_normal((50, 24)))
    assert fig is not None


def test_per_l_breakdown_smoke():
    K = 4 * 6
    pred = np.zeros((4, 4 * K), dtype=np.float32)
    target = np.ones((4, 4 * K), dtype=np.float32)
    fig = build_per_l_breakdown_figure(pred, target, l_max=4)
    assert fig is not None


def test_field_comparison_smoke():
    P = np.abs(np.random.default_rng(0).standard_normal((2, 8, 12))).astype(np.float32)
    fig = build_field_comparison_figure(P, P + 0.01)
    assert fig is not None


def test_r2_distribution_smoke_handles_inf_and_empty():
    import pytest

    rng = np.random.default_rng(0)
    data = {
        "train_aug":      0.7 + 0.2 * rng.standard_normal(200),
        "val_real":       0.1 + 0.4 * rng.standard_normal(50),
        "holdout_real":  -0.5 + 0.7 * rng.standard_normal(20),
        "synthetic_test": np.concatenate(
            [rng.uniform(-2, -1, 25), [-np.inf, -np.inf]]
        ),
    }
    fig = build_r2_distribution_figure(data, clip_range=(-3.0, 1.0))
    assert fig is not None
    with pytest.raises(ValueError, match="empty"):
        build_r2_distribution_figure({})


def test_r2_distribution_handles_unviolinable_splits():
    """Regression: 2026-05-13 — matplotlib's KDE rejects splits with <2 finite
    values inside the clip range. The violin call must skip those splits
    rather than feed them to GaussianKDE.
    """
    rng = np.random.default_rng(1)
    data = {
        # all -inf (silent antennas) — must NOT be passed to violinplot
        "all_silent": np.array([-np.inf] * 5),
        # all finite but identical (zero variance) — KDE also fails on this
        "all_constant": np.full(20, 0.5),
        # exactly one finite value within clip range — too few for KDE
        "single_in_range": np.array([0.3, -np.inf, -np.inf]),
        # well-populated split that should still get a violin
        "ok": rng.standard_normal(50) * 0.3,
    }
    fig = build_r2_distribution_figure(data, clip_range=(-3.0, 1.0))
    assert fig is not None


def test_pca_explained_variance_smoke():
    fig = build_pca_explained_variance_figure(np.array([0.5, 0.3, 0.1, 0.05, 0.05]))
    assert fig is not None


def test_dummy_probe_smoke():
    fig = build_dummy_probe_figure(np.random.default_rng(0).standard_normal((3, 24)), [0, 5, 12])
    assert fig is not None


def test_metrics_basic():
    from mpinv.core.grid import GridSpec

    pred = np.zeros((4, 24), dtype=np.float32)
    target = np.ones((4, 24), dtype=np.float32)
    assert packed_mse(pred, target) == 1.0
    assert packed_r2(pred, target) <= 0  # constant target -> denominator is zero, r2 ≤ 0
    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    P = np.ones((2, 8, 12), dtype=np.float32)
    assert weighted_mse_P(P, P, grid=grid) == 0.0
    assert weighted_nrmse_P(P, P, grid=grid) == 0.0
    bands = per_theta_band_error(P, P + 0.5, n_bands=4)
    assert bands.shape == (4,)
    assert np.allclose(bands, 0.5)


def test_weighted_r2_P_perfect_and_mean_baselines():
    from mpinv.core.grid import GridSpec

    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    rng = np.random.default_rng(0)
    P_true = np.abs(rng.standard_normal((4, 8, 12))).astype(np.float32) + 0.1
    # Perfect prediction => R² = 1 per sample
    r2 = per_sample_weighted_r2_P(P_true, P_true, grid=grid)
    assert r2.shape == (4,)
    np.testing.assert_allclose(r2, 1.0, atol=1e-6)
    assert weighted_r2_P(P_true, P_true, grid=grid) > 0.999

    # Predicting each sample's own (unweighted) mean approximately gives R² ≈ 0
    # under the unweighted definition. Weighted R² with sin-θ weights differs
    # slightly from 0 because the weighted mean ≠ the unweighted mean, but for
    # smooth random fields the weighted R² of the unweighted mean stays well
    # below the perfect-fit value.
    sample_means = P_true.mean(axis=(-2, -1), keepdims=True)
    P_const = np.broadcast_to(sample_means, P_true.shape).copy()
    r2_const = per_sample_weighted_r2_P(P_const, P_true, grid=grid)
    assert (r2_const < 0.5).all(), r2_const

    # A plainly worse-than-mean predictor (large constant shift) gives R² < 0.
    P_off = P_true + 10.0
    r2_off = per_sample_weighted_r2_P(P_off, P_true, grid=grid)
    assert (r2_off < 0).all(), r2_off
    assert weighted_r2_P(P_off, P_true, grid=grid) < 0


def test_weighted_r2_P_silent_sample_returns_neg_inf():
    from mpinv.core.grid import GridSpec

    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    P_true = np.zeros((3, 8, 12), dtype=np.float32)
    P_pred = np.ones((3, 8, 12), dtype=np.float32)
    r2 = per_sample_weighted_r2_P(P_pred, P_true, grid=grid)
    assert r2.shape == (3,)
    assert np.all(np.isinf(r2)) and np.all(r2 < 0)
    # batch-aggregated R² with all degenerate samples returns NaN
    assert np.isnan(weighted_r2_P(P_pred, P_true, grid=grid))


def _smooth_P(B: int = 3, n_theta: int = 16, n_phi: int = 32) -> np.ndarray:
    rng_local = np.random.default_rng(42)
    out = []
    for _ in range(B):
        u = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        v = np.linspace(0.05, np.pi - 0.05, n_theta)
        f = np.outer(np.sin(v) ** 2, 0.3 + np.cos(u) ** 2)
        f = f + 0.05 * rng_local.standard_normal(f.shape)
        out.append(f - f.min() + 1e-6)
    return np.stack(out).astype(np.float32)


def test_spearman_rho_perfect_and_reversed():
    P = _smooth_P()
    rho = per_sample_spearman_rho_P(P, P)
    assert rho.shape == (P.shape[0],)
    np.testing.assert_allclose(rho, 1.0, atol=1e-12)
    assert spearman_rho_P(P, P) > 0.999

    # Reversing the per-sample ranks gives ρ = -1 exactly.
    P_rev = -P  # negation reverses the rank order
    rho_rev = per_sample_spearman_rho_P(P_rev, P)
    np.testing.assert_allclose(rho_rev, -1.0, atol=1e-12)


def test_spearman_rho_scale_shift_invariance():
    P = _smooth_P()
    rho_scaled = per_sample_spearman_rho_P(7.0 * P + 1e-3, P)
    np.testing.assert_allclose(rho_scaled, 1.0, atol=1e-12)
    rho_shifted = per_sample_spearman_rho_P(P + 5.0, P)
    np.testing.assert_allclose(rho_shifted, 1.0, atol=1e-12)


def test_spearman_rho_handles_constant_target():
    P_const = np.zeros((2, 8, 8), dtype=np.float32)
    P_pred = np.random.default_rng(0).standard_normal((2, 8, 8)).astype(np.float32)
    rho = per_sample_spearman_rho_P(P_pred, P_const)
    assert np.all(np.isnan(rho))
    assert np.isnan(spearman_rho_P(P_pred, P_const))


def test_bin_accuracy_perfect_and_random():
    P = _smooth_P()
    n_bins = 11
    assert bin_accuracy_P(P, P, n_bins) == 1.0
    assert bin_within_k_accuracy_P(P, P, n_bins, k=1) == 1.0
    assert hard_rank_bin_mse_P(P, P, n_bins) == 0.0

    # Permuting pixels within each sample drops bin agreement to roughly 1/n_bins.
    rng_local = np.random.default_rng(0)
    P_perm = P.copy()
    flat = P_perm.reshape(P.shape[0], -1)
    for i in range(flat.shape[0]):
        rng_local.shuffle(flat[i])
    P_perm = flat.reshape(P.shape)
    acc = bin_accuracy_P(P_perm, P, n_bins)
    # Random permutation: expected accuracy = 1/n_bins. Allow generous slack.
    assert 0.5 / n_bins < acc < 2.5 / n_bins, acc


def test_bin_within_1_is_more_forgiving_than_exact():
    """within_k=1 ≥ exact, with strict inequality whenever any pixel is off
    by exactly one bin."""
    P_true = _smooth_P()
    rng_local = np.random.default_rng(1)
    perturb = 0.05 * rng_local.standard_normal(P_true.shape)
    P_pred = (P_true.astype(np.float64) + perturb).astype(np.float32)
    n_bins = 11
    exact = bin_accuracy_P(P_pred, P_true, n_bins)
    within_1 = bin_within_k_accuracy_P(P_pred, P_true, n_bins, k=1)
    assert within_1 >= exact
    # The mild perturbation should at least nudge a few pixels by one bin.
    assert within_1 > exact + 1e-3


def test_per_lm_mse_dict_size():
    K = 4 * 6  # L=4
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((3, 4 * K)).astype(np.float32)
    target = pred + 0.1
    d = per_lm_mse(pred, target, l_max=4)
    assert len(d) == 4 * K  # 4 blocks * K modes


def test_reflected_conjugate_aware_loss():
    rng = np.random.default_rng(0)
    K = 4 * 6
    a = rng.standard_normal((2, 4 * K)).astype(np.float32)
    err_self = reflected_conjugate_aware_loss(a, a, l_max=4)
    assert err_self == 0.0
    err_off = reflected_conjugate_aware_loss(a, a + 1.0, l_max=4)
    assert err_off > 0
