"""Tests for :mod:`mpinv.data.basis_decomposer`.

The contract is: ``decompose_field_to_packed`` is the analytic inverse of the
synthetic generator's forward operator on the project's basis. So for a
random ``packed_in``::

    E = synth(packed_in)
    packed_out = decompose_field_to_packed(E_theta, E_phi)
    assert allclose(packed_out, packed_in, ...)

The check is performed on the tiny fixture grid so the discretisation error is
bounded by the grid resolution alone (no truncation, since we synthesise at
the same ``l_max`` we project onto).
"""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import VSHBasis
from mpinv.data.basis_decomposer import (
    decompose_field_to_packed,
    decomposition_residual,
)
from mpinv.data.synthetic_generator import SyntheticGenerator


def _synthesise_E(
    a_e: np.ndarray, a_m: np.ndarray, basis: VSHBasis
) -> tuple[np.ndarray, np.ndarray]:
    E_e = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E = E_e + E_m
    return E[:, 0], E[:, 1]


def test_decomposer_roundtrip_single_mode(
    tiny_basis: VSHBasis, tiny_grid: GridSpec, tiny_l_max: int
):
    """Project synthesised single-mode field; recover unit at the right slot.

    The tiny fixture grid (12 phi x 8 theta for L=4) is barely above Nyquist
    on phi, so a few coefficients drift by O(few %) from discretisation alone.
    A 7% tolerance is sufficient to catch any structural bug (sign flip,
    wrong family ordering, missing 1/sqrt(l(l+1))) — those would manifest as
    O(100 %) errors. The full-grid roundtrip is accurate to O(1e-4) and is
    covered separately by the importer's per-sample residual log.
    """
    K = tiny_l_max * (tiny_l_max + 2)
    for k_target in (0, 3, 7, K - 1):
        for fam in (0, 1):
            a_e = np.zeros((1, K), dtype=np.complex64)
            a_m = np.zeros((1, K), dtype=np.complex64)
            if fam == 0:
                a_e[0, k_target] = 1.0 + 0.0j
            else:
                a_m[0, k_target] = 1.0 + 0.0j
            E_theta, E_phi = _synthesise_E(a_e, a_m, tiny_basis)
            packed = decompose_field_to_packed(
                E_theta[0], E_phi[0], basis=tiny_basis, grid=tiny_grid
            )
            from mpinv.core.packing import unpack_coefficients

            a_e_rec, a_m_rec = unpack_coefficients(packed[None, :])
            target_e = np.zeros(K, dtype=np.complex64)
            target_m = np.zeros(K, dtype=np.complex64)
            if fam == 0:
                target_e[k_target] = 1.0
            else:
                target_m[k_target] = 1.0
            np.testing.assert_allclose(a_e_rec[0], target_e, rtol=0.07, atol=0.05)
            np.testing.assert_allclose(a_m_rec[0], target_m, rtol=0.07, atol=0.05)


def test_decomposer_roundtrip_random(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(42)
    P, packed_in = tiny_generator.generate_batch(4, rng)
    from mpinv.core.packing import unpack_coefficients

    a_e, a_m = unpack_coefficients(packed_in)
    E_theta, E_phi = _synthesise_E(a_e, a_m, tiny_generator.basis)

    packed_out = decompose_field_to_packed(
        E_theta, E_phi,
        basis=tiny_generator.basis,
        grid=tiny_generator.cfg.grid,
    )
    # See `test_decomposer_roundtrip_single_mode` docstring on tolerance choice.
    np.testing.assert_allclose(packed_in, packed_out, rtol=0.07, atol=0.07)


def test_decomposer_residual_is_small_at_full_lmax(
    tiny_generator: SyntheticGenerator,
):
    rng = np.random.default_rng(7)
    _, packed_in = tiny_generator.generate_batch(3, rng)
    from mpinv.core.packing import unpack_coefficients

    a_e, a_m = unpack_coefficients(packed_in)
    E_theta, E_phi = _synthesise_E(a_e, a_m, tiny_generator.basis)

    diag = decomposition_residual(
        E_theta, E_phi,
        basis=tiny_generator.basis,
        grid=tiny_generator.cfg.grid,
    )
    assert diag["e_rel_residual"] < 0.1, diag
    assert diag["p_rel_rmse"] < 0.2, diag


def test_decomposer_rejects_shape_mismatch(
    tiny_basis: VSHBasis, tiny_grid: GridSpec
):
    bad_e_theta = np.zeros((tiny_grid.n_theta + 1, tiny_grid.n_phi), dtype=np.complex64)
    bad_e_phi = np.zeros_like(bad_e_theta)
    with pytest.raises(ValueError, match="does not match grid"):
        decompose_field_to_packed(
            bad_e_theta, bad_e_phi, basis=tiny_basis, grid=tiny_grid
        )

    e_theta = np.zeros((tiny_grid.n_theta, tiny_grid.n_phi), dtype=np.complex64)
    e_phi = np.zeros((tiny_grid.n_theta, tiny_grid.n_phi + 1), dtype=np.complex64)
    with pytest.raises(ValueError, match="shape mismatch"):
        decompose_field_to_packed(
            e_theta, e_phi, basis=tiny_basis, grid=tiny_grid
        )
