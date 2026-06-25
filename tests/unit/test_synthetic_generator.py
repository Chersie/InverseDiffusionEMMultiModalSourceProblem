"""Tests for the synthetic generator + VSH basis precompute."""

from __future__ import annotations

import numpy as np

from mpinv.data._basis_cache import build_basis
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig


def test_basis_shape(tiny_basis, tiny_grid, tiny_l_max):
    K = tiny_l_max * (tiny_l_max + 2)
    assert tiny_basis.basis.shape == (K, 2, 2, tiny_grid.n_theta, tiny_grid.n_phi)
    assert tiny_basis.basis.dtype == np.complex64


def test_generator_outputs_match_field(tiny_generator, rng):
    E, P, packed = tiny_generator.generate_batch_with_field(n=4, rng=rng)
    K = tiny_generator.n_modes
    assert P.shape == (4, tiny_generator.cfg.grid.n_theta, tiny_generator.cfg.grid.n_phi)
    assert packed.shape == (4, 4 * K)
    P_from_E = (E.real**2 + E.imag**2).sum(axis=1)
    assert np.allclose(P, P_from_E, rtol=1e-5, atol=1e-6)


def test_generator_seed_reproducible(tiny_generator):
    P1, packed1 = tiny_generator.generate_batch(8, np.random.default_rng(123))
    P2, packed2 = tiny_generator.generate_batch(8, np.random.default_rng(123))
    assert np.allclose(P1, P2)
    assert np.allclose(packed1, packed2)


def test_generator_modes(tiny_grid, tiny_l_max, tiny_basis):
    for mode in ("gaussian", "uniform", "colored", "sparse"):
        cfg = SyntheticGeneratorConfig(grid=tiny_grid, l_max=tiny_l_max, mode=mode)
        gen = SyntheticGenerator(cfg=cfg, basis=tiny_basis)
        P, packed = gen.generate_batch(2, np.random.default_rng(0))
        assert np.all(np.isfinite(P))
        assert P.shape[0] == 2


def test_basis_grid_mismatch_raises(tiny_grid, tiny_l_max):
    other = build_basis(tiny_grid, tiny_l_max + 1)
    cfg = SyntheticGeneratorConfig(grid=tiny_grid, l_max=tiny_l_max)
    try:
        SyntheticGenerator(cfg=cfg, basis=other)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
