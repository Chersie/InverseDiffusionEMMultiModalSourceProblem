"""Tests for :func:`mpinv.data.dummy_probe.build_single_mode_probe`."""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import build_basis
from mpinv.data.dummy_probe import build_single_mode_probe


@pytest.fixture(scope="module")
def tiny_basis():
    """A small VSH basis on a 12 x 8 grid; cheap for tests."""
    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    return build_basis(grid, l_max=4)


def test_build_single_mode_probe_shapes(tiny_basis):
    l_max = 4
    K = l_max * (l_max + 2)  # 24
    P_dum, packed_dum, active = build_single_mode_probe(tiny_basis, l_max)
    assert P_dum.shape == (4 * K, 8, 12)
    assert packed_dum.shape == (4 * K, 4 * K)
    assert active == list(range(4 * K))
    assert P_dum.dtype == np.float32
    assert packed_dum.dtype == np.float32


def test_packed_dummy_is_amplitude_scaled_identity(tiny_basis):
    l_max = 4
    K = l_max * (l_max + 2)
    amp = 2.5
    _, packed_dum, _ = build_single_mode_probe(tiny_basis, l_max, amplitude=amp)
    # exactly amp on the diagonal, zeros elsewhere
    np.testing.assert_array_equal(packed_dum, np.eye(4 * K, dtype=np.float32) * amp)


def test_decoded_P_matches_differentiable_decoder(tiny_basis):
    """Decoding through the einsum path must match the torch decoder so the
    cli/train.py report's ``decoder(packed_dummy)`` matches the precomputed
    ``P_dummy`` (no train/test discrepancy)."""
    import torch

    from mpinv.losses.differentiable_field import DifferentiableMultipoleField

    l_max = 4
    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    P_dum, packed_dum, _ = build_single_mode_probe(tiny_basis, l_max)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=l_max, basis=tiny_basis)
    with torch.no_grad():
        P_th = decoder(torch.from_numpy(packed_dum)).cpu().numpy()
    rel = np.abs(P_th - P_dum).max() / max(np.abs(P_dum).max(), 1e-12)
    assert rel < 1e-4, f"decoder mismatch: rel max abs = {rel:g}"


def test_validation_errors(tiny_basis):
    with pytest.raises(ValueError, match="l_max must be"):
        build_single_mode_probe(tiny_basis, l_max=0)
    with pytest.raises(ValueError, match=r"basis\.l_max"):
        build_single_mode_probe(tiny_basis, l_max=5)


def test_amplitude_zero_yields_zero_field(tiny_basis):
    l_max = 4
    P_dum, packed_dum, _ = build_single_mode_probe(tiny_basis, l_max, amplitude=0.0)
    assert np.all(packed_dum == 0.0)
    assert np.all(P_dum == 0.0)


def test_active_index_is_truly_one_hot(tiny_basis):
    """For each row k, the only nonzero packed entry is at column k."""
    l_max = 4
    K = l_max * (l_max + 2)
    _, packed_dum, _ = build_single_mode_probe(tiny_basis, l_max, amplitude=1.0)
    for k in range(4 * K):
        nonzero = np.flatnonzero(packed_dum[k])
        assert nonzero.tolist() == [k], f"row {k} nonzero at {nonzero.tolist()}"
