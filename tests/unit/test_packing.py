"""Tests for ``mpinv.core.packing``."""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.core.packing import (
    K_MODES,
    L_MAX,
    PACKED_DIM,
    flat_index,
    iter_modes,
    pack_coefficients,
    pack_to_sht_grid,
    unpack_coefficients,
    unpack_from_sht_grid,
    zero_above_band,
)


def test_constants_consistent():
    assert K_MODES == L_MAX * (L_MAX + 2)
    assert PACKED_DIM == 4 * K_MODES


def test_iter_modes_count_and_order():
    modes = list(iter_modes(L_MAX))
    assert len(modes) == K_MODES
    assert modes[0] == (1, -1)
    assert modes[-1] == (L_MAX, L_MAX)


def test_flat_index_round_trip():
    for k, (l, m) in enumerate(iter_modes(L_MAX)):
        assert flat_index(l, m) == k


def test_flat_index_out_of_range():
    with pytest.raises(ValueError):
        flat_index(0, 0)
    with pytest.raises(ValueError):
        flat_index(2, 3)


def test_pack_unpack_round_trip():
    rng = np.random.default_rng(0)
    a_e = (rng.standard_normal((4, K_MODES)) + 1j * rng.standard_normal((4, K_MODES))).astype(
        np.complex64
    )
    a_m = (rng.standard_normal((4, K_MODES)) + 1j * rng.standard_normal((4, K_MODES))).astype(
        np.complex64
    )
    packed = pack_coefficients(a_e, a_m)
    assert packed.shape == (4, PACKED_DIM)
    a_e2, a_m2 = unpack_coefficients(packed)
    assert np.allclose(a_e, a_e2, atol=1e-6)
    assert np.allclose(a_m, a_m2, atol=1e-6)


def test_pack_unpack_arbitrary_l():
    rng = np.random.default_rng(0)
    K = 4 * (4 + 2)  # L=4 -> K=24
    a_e = (rng.standard_normal((2, K)) + 1j * rng.standard_normal((2, K))).astype(np.complex64)
    a_m = (rng.standard_normal((2, K)) + 1j * rng.standard_normal((2, K))).astype(np.complex64)
    packed = pack_coefficients(a_e, a_m)
    assert packed.shape == (2, 4 * K)
    a_e2, a_m2 = unpack_coefficients(packed)
    assert np.allclose(a_e, a_e2)
    assert np.allclose(a_m, a_m2)


def test_pack_to_sht_grid_shape_and_only_m_nonneg_stored():
    rng = np.random.default_rng(0)
    K = K_MODES
    a_e = (rng.standard_normal((3, K)) + 1j * rng.standard_normal((3, K))).astype(np.complex64)
    a_m = (rng.standard_normal((3, K)) + 1j * rng.standard_normal((3, K))).astype(np.complex64)
    packed = pack_coefficients(a_e, a_m)
    g = pack_to_sht_grid(packed)
    assert g.shape == (3, 2, L_MAX + 1, L_MAX + 1)
    assert g.dtype == np.complex64
    # l=0 row stays all zeros
    assert np.all(g[:, :, 0, :] == 0)


def test_hermitian_round_trip_through_sht_grid():
    """``unpack_from_sht_grid(pack_to_sht_grid(a)) == a`` only when ``a`` is
    Hermitian-symmetric (``a_{l,-m} = (-1)^m * conj(a_{l,m})``). Asymmetric inputs
    lose the ``m < 0`` information by construction.
    """
    rng = np.random.default_rng(0)
    K = K_MODES
    # construct a Hermitian-symmetric coefficient set
    a_e = np.zeros((2, K), dtype=np.complex64)
    a_m = np.zeros((2, K), dtype=np.complex64)
    for k, (l, m) in enumerate(iter_modes(L_MAX)):
        if m >= 0:
            v_e = rng.standard_normal() + 1j * rng.standard_normal()
            v_m = rng.standard_normal() + 1j * rng.standard_normal()
            a_e[:, k] = v_e
            a_m[:, k] = v_m
    for k, (l, m) in enumerate(iter_modes(L_MAX)):
        if m < 0:
            # find the matching m=-m entry
            for k2, (l2, m2) in enumerate(iter_modes(L_MAX)):
                if l2 == l and m2 == -m:
                    a_e[:, k] = ((-1) ** m) * np.conj(a_e[:, k2])
                    a_m[:, k] = ((-1) ** m) * np.conj(a_m[:, k2])
                    break

    packed = pack_coefficients(a_e, a_m)
    g = pack_to_sht_grid(packed)
    packed_back = unpack_from_sht_grid(g)
    assert np.allclose(packed, packed_back, atol=1e-5)


# re-export the symbol used in the construction above
from mpinv.core.packing import iter_modes  # noqa: F811

# ---------------------------------------------------------------------------
# zero_above_band
# ---------------------------------------------------------------------------


def test_zero_above_band_keeps_low_l_zeroes_high_l():
    """For l_max=5 and k=2, the function zeroes coefficients with l ∈ {3, 4, 5}
    and leaves l ∈ {1, 2} untouched, in every quarter of the packed layout."""
    l_max = 5
    K = l_max * (l_max + 2)  # 35
    rng = np.random.default_rng(0)
    packed = rng.standard_normal((3, 4 * K)).astype(np.float32)
    out = zero_above_band(packed, k=2, l_max=l_max)

    boundary = 2 * (2 + 2)  # k*(k+2) = 8
    for q in range(4):
        # l ∈ {1, 2} block survives unchanged.
        np.testing.assert_array_equal(
            out[..., q * K : q * K + boundary], packed[..., q * K : q * K + boundary]
        )
        # l ∈ {3, 4, 5} block is zero.
        np.testing.assert_array_equal(
            out[..., q * K + boundary : (q + 1) * K], 0.0
        )


def test_zero_above_band_k_equals_lmax_is_noop():
    l_max = 5
    K = l_max * (l_max + 2)
    rng = np.random.default_rng(1)
    packed = rng.standard_normal((2, 4 * K)).astype(np.float32)
    out = zero_above_band(packed, k=l_max, l_max=l_max)
    np.testing.assert_array_equal(out, packed)
    # not the same object: it is a copy
    assert out.base is not packed
    assert out is not packed


def test_zero_above_band_validates_inputs():
    l_max = 5
    K = l_max * (l_max + 2)
    packed = np.zeros((4 * K,), dtype=np.float32)
    with pytest.raises(ValueError):
        zero_above_band(packed, k=-1, l_max=l_max)
    with pytest.raises(ValueError):
        zero_above_band(packed, k=l_max + 1, l_max=l_max)
    bad = np.zeros((4 * K - 1,), dtype=np.float32)
    with pytest.raises(ValueError):
        zero_above_band(bad, k=2, l_max=l_max)


def test_zero_above_band_works_for_torch_tensors():
    """The same call path supports torch tensors via duck-typed ``.clone()``."""
    import torch

    l_max = 4
    K = l_max * (l_max + 2)  # 24
    t = torch.randn(2, 4 * K)
    out = zero_above_band(t, k=2, l_max=l_max)
    assert isinstance(out, torch.Tensor)
    boundary = 2 * (2 + 2)
    for q in range(4):
        torch.testing.assert_close(
            out[..., q * K : q * K + boundary], t[..., q * K : q * K + boundary]
        )
        assert torch.all(out[..., q * K + boundary : (q + 1) * K] == 0.0)
