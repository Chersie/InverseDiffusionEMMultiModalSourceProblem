"""Tests for :func:`mpinv.data.real_augmented_pipeline.build_real_augmented_pipeline`.

We exercise the smoke-test code path (synthetic stand-in for the real-antenna
corpus) so the test is hermetic and does not require ``data/raw/real_antenna``.
"""

from __future__ import annotations

import numpy as np

from mpinv.data.real_augmented_pipeline import build_real_augmented_pipeline

_GRID = {
    "n_phi": 12,
    "n_theta": 8,
    "theta_start_deg": 15.0,
    "theta_end_deg": 165.0,
}


def _build_smoke(**overrides):
    kwargs: dict = {
        "grid": _GRID,
        "l_max": 4,
        "n_source": 20,
        "n_train_sources": 15,
        "n_augmented": 32,
        "n_holdout_samples": 0,
        "dropout_prob": 0.0,
        "field_sigma": 0.0,
        "scale_factor": 1.0,
        "aug_chunk_size": 16,
        "include_synthetic_test": True,
        "n_synthetic_test": 8,
        "smoke_test": True,
        "batch_size": 4,
        "num_workers": 0,
    }
    kwargs.update(overrides)
    return build_real_augmented_pipeline(**kwargs)


def test_pipeline_emits_dummy_split_by_default():
    out = _build_smoke()
    K = 4 * (4 + 2)
    assert "P_dummy" in out
    assert "packed_dummy" in out
    assert "dummy_active_indices" in out
    assert out["P_dummy"].shape == (4 * K, _GRID["n_theta"], _GRID["n_phi"])
    assert out["packed_dummy"].shape == (4 * K, 4 * K)
    assert out["dummy_active_indices"] == list(range(4 * K))
    assert out["n_dummy"] == 4 * K


def test_pipeline_dummy_can_be_disabled():
    out = _build_smoke(include_dummy_probe=False)
    assert "P_dummy" not in out
    assert "packed_dummy" not in out
    assert "dummy_active_indices" not in out


def test_pipeline_dummy_amplitude_scales_packed():
    amp = 2.0
    out = _build_smoke(dummy_amplitude=amp)
    K = 4 * (4 + 2)
    np.testing.assert_array_equal(
        out["packed_dummy"], np.eye(4 * K, dtype=np.float32) * amp
    )


def test_pipeline_other_splits_unchanged_when_dummy_enabled():
    """Adding the dummy split must not perturb train/val/test/holdout shapes."""
    out = _build_smoke()
    assert out["P_train"].shape[0] == 32  # n_augmented
    assert out["P_val"].shape[0] == 5     # n_source - n_train_sources
    assert out["P_test"].shape[0] == 8    # n_synthetic_test
