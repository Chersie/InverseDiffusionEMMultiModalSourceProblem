"""Tests for the feature pipeline."""

from __future__ import annotations

import numpy as np

from mpinv.features.modes import InputMode, select_channels
from mpinv.features.normalisers import StandardScaler
from mpinv.features.pca import RandomizedPCA
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig


def test_select_channels_power(rng):
    P = rng.uniform(0, 1, size=(4, 8, 12)).astype(np.float32)
    out = select_channels(None, P, InputMode.POWER)
    assert out.shape == (4, 1, 8, 12)
    assert out.dtype == np.float32


def test_select_channels_magnitude_and_complex(rng):
    E = rng.standard_normal((4, 2, 8, 12)) + 1j * rng.standard_normal((4, 2, 8, 12))
    mag = select_channels(E, None, InputMode.MAGNITUDE)
    assert mag.shape == (4, 2, 8, 12)
    cpx = select_channels(E, None, InputMode.COMPLEX)
    assert cpx.shape == (4, 4, 8, 12)


def test_standard_scaler_round_trip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5)).astype(np.float32) * 3.0 + 2.0
    s = StandardScaler()
    Z = s.fit_transform(X)
    assert np.isclose(Z.mean(), 0.0, atol=1e-3)
    assert np.isclose(Z.std(), 1.0, atol=1e-3)
    Xb = s.inverse_transform(Z)
    assert np.allclose(Xb, X, atol=1e-5)


def test_randomized_pca_dim():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 50)).astype(np.float32)
    p = RandomizedPCA(n_components=8, random_state=0)
    Z = p.fit_transform(X)
    assert Z.shape == (200, 8)
    assert p.explained_variance_ratio_.shape == (8,)


def test_power_pca_pipeline_fit_transform(tiny_generator, rng):
    P_train, _ = tiny_generator.generate_batch(64, rng)
    pipe = PowerPCAPipeline(cfg=PowerPCAPipelineConfig(pca_components=4))
    pipe.fit(P_train=P_train)
    z = pipe.transform(P=P_train)
    assert z.shape == (64, 4)
    assert pipe.feature_dim == 4
