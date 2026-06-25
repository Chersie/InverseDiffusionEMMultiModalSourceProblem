"""Tests for the new per-sample metric helpers and the three new violin wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.analysis.metrics import (
    per_sample_packed_mse,
    per_sample_weighted_nrmse_P,
)
from mpinv.analysis.plots.r2_distribution import (
    build_coef_mse_distribution_figure,
    build_nrmse_distribution_figure,
    build_spearman_distribution_figure,
)
from mpinv.core.grid import GridSpec

_TINY_GRID = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)


# ---------------------------------------------------------------------------
# per_sample_weighted_nrmse_P
# ---------------------------------------------------------------------------


def test_per_sample_nrmse_zero_when_perfect():
    P = np.abs(np.random.default_rng(0).standard_normal((4, 8, 12))).astype(np.float32) + 0.1
    out = per_sample_weighted_nrmse_P(P, P, grid=_TINY_GRID)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_per_sample_nrmse_positive_when_off():
    rng = np.random.default_rng(0)
    P_true = np.abs(rng.standard_normal((4, 8, 12))).astype(np.float32) + 0.1
    P_pred = P_true * 2.0
    out = per_sample_weighted_nrmse_P(P_pred, P_true, grid=_TINY_GRID)
    assert out.shape == (4,)
    assert (out > 0).all()


def test_per_sample_nrmse_constant_target_returns_sentinel():
    """When P_true is identically zero, NRMSE is undefined (0/0). The function
    returns the sentinel (default NaN) so the violin plot drops those samples."""
    P_true = np.zeros((3, 8, 12), dtype=np.float32)
    P_pred = np.ones_like(P_true)
    out = per_sample_weighted_nrmse_P(P_pred, P_true, grid=_TINY_GRID)
    assert np.isnan(out).all()


# ---------------------------------------------------------------------------
# per_sample_packed_mse
# ---------------------------------------------------------------------------


def test_per_sample_packed_mse_shape_and_values():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((5, 24)).astype(np.float32)
    target = pred + rng.standard_normal((5, 24)).astype(np.float32) * 0.1
    out = per_sample_packed_mse(pred, target)
    assert out.shape == (5,)
    expected = ((pred - target) ** 2).mean(axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_per_sample_packed_mse_zero_on_identity():
    a = np.ones((3, 12), dtype=np.float32)
    out = per_sample_packed_mse(a, a)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_per_sample_packed_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        per_sample_packed_mse(np.zeros((2, 8)), np.zeros((2, 12)))


def test_per_sample_packed_mse_rejects_3d():
    with pytest.raises(ValueError, match=r"\(B, 4 K\)"):
        per_sample_packed_mse(np.zeros((2, 8, 4)), np.zeros((2, 8, 4)))


# ---------------------------------------------------------------------------
# Violin-wrapper smoke tests
# ---------------------------------------------------------------------------


def _toy_metric_dict(rng_seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    return {
        "train_aug": rng.uniform(0.5, 1.0, size=64),
        "val": rng.uniform(0.3, 0.95, size=32),
        "holdout": rng.uniform(-0.2, 0.7, size=32),
        "test": rng.uniform(0.0, 0.8, size=32),
        "dummy": rng.uniform(-0.5, 0.5, size=140),
    }


def test_build_spearman_distribution_figure_smoke():
    fig = build_spearman_distribution_figure(_toy_metric_dict(0), title="rho test")
    assert fig is not None
    assert len(fig.axes) >= 2


def test_build_nrmse_distribution_figure_smoke():
    rng = np.random.default_rng(1)
    nrmse = {tag: rng.uniform(0.05, 2.5, size=32) for tag in ("train_aug", "val", "holdout")}
    fig = build_nrmse_distribution_figure(nrmse, title="NRMSE test")
    assert fig is not None


def test_build_coef_mse_distribution_figure_smoke():
    rng = np.random.default_rng(2)
    cmse = {tag: rng.uniform(0.0, 3.0, size=32) for tag in ("train_aug", "val", "holdout", "test", "dummy")}
    fig = build_coef_mse_distribution_figure(cmse, title="coef MSE test")
    assert fig is not None


def test_violins_handle_nan_entries():
    """Spearman rho returns NaN for degenerate samples; the wrapper must not crash."""
    rho = {
        "val": np.array([0.5, np.nan, 0.7, np.nan, 0.9]),
        "test": np.array([np.nan, np.nan, np.nan]),  # all degenerate
        "holdout": np.array([0.3, 0.4, 0.5, 0.6]),
    }
    fig = build_spearman_distribution_figure(rho)
    assert fig is not None
