"""Phase C — physics-layer regression tests.

Four required checks per the plan:

1. Reciprocity: numpy einsum forward (data generator) and torch einsum forward
   (DifferentiableMultipoleField) must agree pointwise on ``P`` to float32 noise.
2. Gradient flow at the full project grid (179, 360): a single forward + backward
   pass through ``DifferentiableMultipoleField`` produces non-zero gradients on
   the input packed coefficients. This closes the open question raised in the
   legacy ``gradient_flow_investigation_results.md``.
3. Reflected-conjugate ambiguity: the analytical map ``a -> a'`` defined in
   presentation/ch1_full.md §1.7 ((-1)^m * conj(a_{l, -m})) produces a coefficient
   set whose power pattern matches the original to noise floor.
4. Single-mode injection: for every (l, m, family) at L = 4, set exactly one
   coefficient to 1+0j, run the decoder, and check the output power has the right
   per-mode magnitude (cross-check against the data generator's einsum on the
   same basis).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mpinv.core.grid import GRID_DEFAULT
from mpinv.core.packing import K_MODES, L_MAX, iter_modes, pack_coefficients
from mpinv.data._basis_cache import build_basis
from mpinv.losses.differentiable_field import DifferentiableMultipoleField


@pytest.mark.physics
def test_reciprocity_tiny(tiny_generator, tiny_basis):
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    P_np, packed = tiny_generator.generate_batch(8, np.random.default_rng(0))
    P_th = decoder(torch.from_numpy(packed)).detach().numpy()
    rel = np.abs(P_th - P_np).max() / max(np.abs(P_np).max(), 1e-12)
    assert rel < 1e-4, f"reciprocity rel error too large: {rel:.3e}"


@pytest.mark.physics
@pytest.mark.slow
def test_gradient_flow_full_grid():
    """Differentiability check on the project's (179, 360) grid at L = 15.

    This is the regression test that pins the legacy gradient_flow_investigation
    claim: the new layer produces non-zero gradients on the full project grid.
    Marked slow because the basis precompute on full L=15 takes a few seconds.
    """
    grid = GRID_DEFAULT
    basis = build_basis(grid, l_max=L_MAX)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=L_MAX, basis=basis)
    K = K_MODES
    packed = torch.randn(2, 4 * K, requires_grad=True)
    P = decoder(packed)
    assert P.shape == (2, grid.n_theta, grid.n_phi)
    P.pow(2).sum().backward()
    assert packed.grad.abs().sum().item() > 0


@pytest.mark.physics
def test_reflected_conjugate_ambiguity(tiny_generator, tiny_basis):
    """Verify the §1.7 reflected-conjugate ambiguity: the map
    ``a'_{l,m} = (-1)^m * conj(a_{l,-m})`` produces an identical power pattern.
    """
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    rng = np.random.default_rng(0)
    a_e, a_m = tiny_generator.sample_coefficients(2, rng)

    # construct a' via the reflected-conjugate formula
    K = tiny_generator.n_modes
    a_e_p = np.zeros_like(a_e)
    a_m_p = np.zeros_like(a_m)
    mode_index: dict[tuple[int, int], int] = {
        (l, m): k for k, (l, m) in enumerate(iter_modes(tiny_generator.cfg.l_max))
    }
    for k, (l, m) in enumerate(iter_modes(tiny_generator.cfg.l_max)):
        k_neg = mode_index[(l, -m)]
        sign = (-1.0) ** m
        a_e_p[:, k] = sign * np.conj(a_e[:, k_neg])
        a_m_p[:, k] = sign * np.conj(a_m[:, k_neg])

    packed = pack_coefficients(a_e, a_m)
    packed_p = pack_coefficients(a_e_p, a_m_p)
    P = decoder(torch.from_numpy(packed)).detach().numpy()
    P_p = decoder(torch.from_numpy(packed_p)).detach().numpy()
    rel = np.abs(P - P_p).max() / max(np.abs(P).max(), 1e-12)
    assert rel < 1e-4, f"reflected-conjugate ambiguity broken: rel={rel:.3e}"


@pytest.mark.physics
def test_single_mode_injection_matches_generator(tiny_generator, tiny_basis):
    """For every (l, m, family) at the tiny truncation, set exactly that single
    coefficient and verify the decoder's power matches the generator's
    einsum-based power.
    """
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    K = tiny_generator.n_modes
    for k, (l, m) in enumerate(iter_modes(tiny_generator.cfg.l_max)):
        for family_idx in (0, 1):
            a_e = np.zeros((1, K), dtype=np.complex64)
            a_m = np.zeros((1, K), dtype=np.complex64)
            (a_e if family_idx == 0 else a_m)[0, k] = 1.0 + 0.0j
            packed = pack_coefficients(a_e, a_m)
            P_th = decoder(torch.from_numpy(packed)).detach().numpy()[0]
            _, P_np = tiny_generator.synthesize(a_e, a_m)
            P_np = P_np[0]
            rel = np.abs(P_th - P_np).max() / max(np.abs(P_np).max(), 1e-12)
            assert rel < 1e-4, f"mismatch at (l={l}, m={m}, family={family_idx}): rel={rel:.3e}"


@pytest.mark.physics
def test_decoder_basis_grid_mismatch_raises(tiny_grid, tiny_l_max):
    other = build_basis(tiny_grid, tiny_l_max + 1)
    with pytest.raises(ValueError):
        DifferentiableMultipoleField(grid=tiny_grid, l_max=tiny_l_max, basis=other)
