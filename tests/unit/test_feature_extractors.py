"""Tests for the FFT radial / HOG / SH-power / composite feature extractors."""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.features.composite import CompositeFeaturesConfig, CompositePipeline
from mpinv.features.fft_radial import FFTRadial, FFTRadialConfig
from mpinv.features.modes import InputMode
from mpinv.features.power_pipeline import PowerPCAPipelineConfig
from mpinv.features.raw_flat import RawFlattenPipeline, RawFlattenPipelineConfig
from mpinv.features.sh_power import SHPower, SHPowerConfig
from mpinv.features.subsample import SubsampleGridPipeline, SubsampleGridPipelineConfig


def test_fft_radial_shape(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(8, rng)
    f = FFTRadial(FFTRadialConfig(n_bins=8, input_mode=InputMode.POWER))
    f.fit(P_train=P)
    out = f.transform(P=P)
    # 1 channel for POWER mode * 8 bins = 8
    assert out.shape == (8, 8)
    assert f.feature_dim == 8


def test_fft_radial_complex_mode(tiny_generator, rng):
    E, _, _ = tiny_generator.generate_batch_with_field(4, rng)
    f = FFTRadial(FFTRadialConfig(n_bins=4, input_mode=InputMode.COMPLEX))
    f.fit(E_train=E)
    out = f.transform(E=E)
    # 4 channels * 4 bins
    assert out.shape == (4, 16)


def test_sh_power_shape(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(4, rng)
    f = SHPower(SHPowerConfig(l_max=tiny_generator.cfg.l_max), grid=tiny_generator.cfg.grid)
    f.fit(P_train=P)
    out = f.transform(P=P)
    # 1 channel * l_max
    assert out.shape == (4, tiny_generator.cfg.l_max)


def test_composite_pca_plus_cv(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(64, rng)
    pipe = CompositePipeline(
        cfg=CompositeFeaturesConfig(pca=PowerPCAPipelineConfig(pca_components=8)),
        extractors=[
            FFTRadial(FFTRadialConfig(n_bins=4, input_mode=InputMode.POWER)),
            SHPower(SHPowerConfig(l_max=tiny_generator.cfg.l_max), grid=tiny_generator.cfg.grid),
        ],
    )
    pipe.fit(P_train=P)
    out = pipe.transform(P=P)
    assert out.shape[0] == 64
    # PCA(8) + FFT(4) + SH(l_max=4) = 16
    assert out.shape[1] == 8 + 4 + tiny_generator.cfg.l_max
    assert pipe.feature_dim == out.shape[1]


def test_raw_flat_shape(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(8, rng)
    f = RawFlattenPipeline(RawFlattenPipelineConfig(input_mode=InputMode.POWER))
    f.fit(P_train=P)
    out = f.transform(P=P)
    n_theta = tiny_generator.cfg.grid.n_theta
    n_phi = tiny_generator.cfg.grid.n_phi
    assert out.shape == (8, n_theta * n_phi)
    assert f.feature_dim == n_theta * n_phi


def test_raw_flat_complex_mode(tiny_generator, rng):
    E, _, _ = tiny_generator.generate_batch_with_field(4, rng)
    f = RawFlattenPipeline(RawFlattenPipelineConfig(input_mode=InputMode.COMPLEX))
    f.fit(E_train=E)
    out = f.transform(E=E)
    n_theta = tiny_generator.cfg.grid.n_theta
    n_phi = tiny_generator.cfg.grid.n_phi
    # 4 real channels (Re/Im of E_theta and E_phi)
    assert out.shape == (4, 4 * n_theta * n_phi)


def test_raw_flat_normalisation_zero_mean(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(64, rng)
    f = RawFlattenPipeline(RawFlattenPipelineConfig(normalise_features=True))
    f.fit(P_train=P)
    z = f.transform(P=P)
    # Per-feature mean should be ~0 after StandardScaler
    assert np.abs(z.mean(axis=0)).max() < 1e-4


def test_subsample_stride_shape(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(8, rng)
    n_theta = tiny_generator.cfg.grid.n_theta
    n_phi = tiny_generator.cfg.grid.n_phi
    f = SubsampleGridPipeline(
        SubsampleGridPipelineConfig(theta_stride=2, phi_stride=3, normalise_features=False)
    )
    f.fit(P_train=P)
    out = f.transform(P=P)
    expected_theta = (n_theta + 1) // 2
    expected_phi = (n_phi + 2) // 3
    assert out.shape == (8, expected_theta * expected_phi)


def test_subsample_random_mask_reproducible(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(8, rng)
    cfg = SubsampleGridPipelineConfig(
        random_fraction=0.25,
        mask_seed=123,
        normalise_features=False,
    )
    f1 = SubsampleGridPipeline(cfg)
    f2 = SubsampleGridPipeline(cfg)
    f1.fit(P_train=P)
    f2.fit(P_train=P)
    out1 = f1.transform(P=P)
    out2 = f2.transform(P=P)
    assert out1.shape == out2.shape
    assert np.array_equal(out1, out2)


def test_subsample_random_mask_size(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(4, rng)
    n_theta = tiny_generator.cfg.grid.n_theta
    n_phi = tiny_generator.cfg.grid.n_phi
    n_pix = n_theta * n_phi
    f = SubsampleGridPipeline(
        SubsampleGridPipelineConfig(
            random_fraction=0.5,
            mask_seed=0,
            normalise_features=False,
        )
    )
    f.fit(P_train=P)
    out = f.transform(P=P)
    # 1 channel, 50% kept
    assert out.shape[1] == round(0.5 * n_pix)


def test_hog_smoke(tiny_generator, rng):
    P, _ = tiny_generator.generate_batch(2, rng)
    try:
        from mpinv.features.hog import HOGConfig, HOGExtractor

        # tiny grid doesn't divide nicely with default cells_per_block; use 2x2 cells
        f = HOGExtractor(
            HOGConfig(pixels_per_cell=(4, 4), cells_per_block=(1, 1), input_mode=InputMode.POWER)
        )
        f.fit(P_train=P)
        out = f.transform(P=P)
        assert out.shape[0] == 2
        assert f.feature_dim > 0
    except RuntimeError as exc:
        pytest.skip(f"scikit-image not available: {exc}")
