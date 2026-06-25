"""Tests for :func:`mpinv.analysis.reports.run_report.build_split_report`."""

from __future__ import annotations

import numpy as np

from mpinv.analysis.reports.run_report import RunArtifacts, build_split_report
from mpinv.core.grid import GridSpec

_TINY_GRID = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)


def _make_art(B: int = 6, l_max: int = 3, *, rng_seed: int = 0) -> RunArtifacts:
    K = l_max * (l_max + 2)
    rng = np.random.default_rng(rng_seed)
    target_packed = rng.standard_normal((B, 4 * K)).astype(np.float32)
    pred_packed = target_packed + 0.1 * rng.standard_normal((B, 4 * K)).astype(np.float32)
    P_true = (
        np.abs(rng.standard_normal((B, _TINY_GRID.n_theta, _TINY_GRID.n_phi))).astype(np.float32)
        + 0.1
    )
    P_pred = P_true + 0.05 * rng.standard_normal(P_true.shape).astype(np.float32)
    return RunArtifacts(
        pred_packed=pred_packed,
        target_packed=target_packed,
        P_pred=P_pred,
        P_true=P_true,
        l_max=l_max,
        grid=_TINY_GRID,
    )


def test_split_report_writes_expected_pdfs(tmp_path):
    art = _make_art()
    metrics, per_sample = build_split_report(
        art, output_dir=tmp_path, split="val", n_grid_samples=4,
    )
    out = tmp_path / "val"
    assert out.is_dir()
    assert (out / "coef_histograms.pdf").exists()
    assert (out / "coef_scatter.pdf").exists()
    assert (out / "per_l_breakdown.pdf").exists()
    assert (out / "field_comparison_grid.pdf").exists()
    # No dummy_probe.pdf when dummy_active_indices is None
    assert not (out / "dummy_probe.pdf").exists()

    # Metric keys carry the split tag.
    for k in metrics:
        assert k.startswith("report/val/")
    # Required keys present.
    expected = {
        "report/val/coef_mse", "report/val/coef_r2", "report/val/coef_mse_amb_aware",
        "report/val/field_mse_w", "report/val/field_nrmse_w",
        "report/val/spearman_rho_P", "report/val/bin_accuracy_P",
    }
    assert expected <= set(metrics)

    # Per-sample arrays of correct shape.
    B = art.P_true.shape[0]
    for key in ("r2", "bin_accuracy", "spearman_rho", "nrmse", "coef_mse"):
        assert per_sample[key].shape == (B,), f"{key}: {per_sample[key].shape}"


def test_split_report_dummy_branch_skips_histograms_and_emits_probe(tmp_path):
    """For the dummy split: target_packed is one-hot per row, so coef_histograms
    is degenerate and skipped; dummy_probe.pdf is emitted instead."""
    K = 3 * (3 + 2)  # 15
    n = 4 * K
    target_packed = np.eye(n, dtype=np.float32)
    pred_packed = target_packed + 0.05 * np.random.default_rng(0).standard_normal(target_packed.shape).astype(np.float32)
    rng = np.random.default_rng(1)
    P_true = np.abs(rng.standard_normal((n, _TINY_GRID.n_theta, _TINY_GRID.n_phi))).astype(np.float32) + 0.1
    P_pred = P_true + 0.05 * rng.standard_normal(P_true.shape).astype(np.float32)
    art = RunArtifacts(
        pred_packed=pred_packed, target_packed=target_packed,
        P_pred=P_pred, P_true=P_true, l_max=3, grid=_TINY_GRID,
    )
    metrics, per_sample = build_split_report(
        art, output_dir=tmp_path, split="dummy",
        dummy_active_indices=list(range(n)),
        n_grid_samples=6,
    )
    out = tmp_path / "dummy"
    assert (out / "dummy_probe.pdf").exists()
    assert not (out / "coef_histograms.pdf").exists()
    assert (out / "coef_scatter.pdf").exists()
    assert (out / "per_l_breakdown.pdf").exists()
    assert (out / "field_comparison_grid.pdf").exists()
    assert "report/dummy/coef_mse" in metrics
    assert per_sample["r2"].shape == (n,)


def test_split_report_emits_pca_only_for_val(tmp_path):
    art = _make_art()
    art = RunArtifacts(
        pred_packed=art.pred_packed, target_packed=art.target_packed,
        P_pred=art.P_pred, P_true=art.P_true, l_max=art.l_max, grid=art.grid,
        pca_explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
    )
    build_split_report(art, output_dir=tmp_path, split="val")
    assert (tmp_path / "val" / "pca_explained_variance.pdf").exists()
    build_split_report(art, output_dir=tmp_path, split="train_aug")
    assert not (tmp_path / "train_aug" / "pca_explained_variance.pdf").exists()


def test_split_report_grid_samples_clipped_to_batch(tmp_path):
    """If B < n_grid_samples we still get a valid grid figure with B rows."""
    art = _make_art(B=3)
    build_split_report(art, output_dir=tmp_path, split="holdout", n_grid_samples=8)
    assert (tmp_path / "holdout" / "field_comparison_grid.pdf").exists()
