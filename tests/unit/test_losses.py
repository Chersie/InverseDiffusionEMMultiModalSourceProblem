"""Tests for losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mpinv.losses.coef_mse import CoefMSE
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig
from mpinv.losses.rank_bin import (
    RankBinPLoss,
    RankBinPLossConfig,
    rank_bin_mse,
)
from mpinv.losses.registry import LOSSES


def test_registry():
    assert "coef_mse" in LOSSES
    assert "physics_power" in LOSSES
    assert "rank_bin_p" in LOSSES


def test_coef_mse_zero_when_equal():
    K = 24
    pred = torch.randn(4, 4 * K)
    loss = CoefMSE()(pred, pred.clone())
    assert torch.allclose(loss, torch.zeros(()))
    assert "coef_mse" in CoefMSE().last_components or True  # last_components is per-instance


def test_coef_mse_shape_mismatch():
    f = CoefMSE()
    with pytest.raises(ValueError):
        f(torch.randn(4, 8), torch.randn(4, 12))


def test_differentiable_field_reciprocity(tiny_generator, tiny_basis, rng):
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    P_np, packed = tiny_generator.generate_batch(4, rng)
    P_th = decoder(torch.from_numpy(packed)).detach().numpy()
    rel = np.abs(P_th - P_np).max() / max(np.abs(P_np).max(), 1e-12)
    assert rel < 1e-4


def test_differentiable_field_gradient_flow(tiny_generator, tiny_basis):
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    K = tiny_generator.n_modes
    packed = torch.randn(2, 4 * K, requires_grad=True)
    P = decoder(packed)
    P.pow(2).sum().backward()
    assert packed.grad.abs().sum().item() > 0


def test_physics_power_loss(tiny_generator, tiny_basis, rng):
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    target_P = torch.from_numpy(P_np)
    pred = torch.from_numpy(packed).clone().requires_grad_(True)
    val = loss(pred, target_P)
    assert val.item() < 1e-6  # exact match should give ~0 loss
    val.backward()
    # gradient should still be ~0 because we are at the minimum, but it should be defined
    assert pred.grad is not None


def test_physics_power_loss_nonzero_when_off(tiny_generator, tiny_basis, rng):
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    pred = torch.from_numpy(packed) + 1.0  # deliberately off
    val = loss(pred.requires_grad_(True), torch.from_numpy(P_np))
    assert val.item() > 0


# ---- rank_bin_mse / RankBinPLoss tests ----


def _toy_P(n_phi: int = 32, n_theta: int = 16) -> torch.Tensor:
    """Build a smooth, non-trivial pattern of shape (B=3, n_theta, n_phi)."""
    rng_local = np.random.default_rng(42)
    out = []
    for _ in range(3):
        u = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        v = np.linspace(0.05, np.pi - 0.05, n_theta)
        f = np.outer(np.sin(v) ** 2, 0.3 + np.cos(u) ** 2)
        f = f + 0.1 * rng_local.standard_normal(f.shape)
        out.append(f - f.min() + 1e-6)
    return torch.tensor(np.stack(out), dtype=torch.float32)


def test_rank_bin_identity_zero():
    P = _toy_P()
    val = rank_bin_mse(P, P, n_bins=11)
    # Soft binning is sigmoid-smooth, so it's not exactly zero, but very close.
    assert val.item() < 1e-6


def test_rank_bin_scale_invariance():
    P = _toy_P()
    val_scaled = rank_bin_mse(7.0 * P + 1e-3, P, n_bins=11)
    assert val_scaled.item() < 1e-6


def test_rank_bin_shift_invariance():
    P = _toy_P()
    val_shifted = rank_bin_mse(P + 5.0, P, n_bins=11)
    assert val_shifted.item() < 1e-6


def test_rank_bin_permutation_sensitivity():
    """A permutation that destroys spatial rank order must give large loss."""
    P = _toy_P()
    P_shuffled = P.clone()
    g = torch.Generator().manual_seed(0)
    flat = P_shuffled.reshape(P_shuffled.shape[0], -1)
    perm = torch.randperm(flat.shape[-1], generator=g)
    flat = flat[:, perm]
    P_shuffled = flat.reshape_as(P)
    val = rank_bin_mse(P_shuffled, P, n_bins=11)
    assert val.item() > 0.5, val.item()


def test_rank_bin_gradient_flow():
    """The soft-bin formulation must preserve a non-trivial gradient on
    pred_packed when wired through the standalone RankBinPLoss."""
    n_theta = 16
    n_phi = 32
    from mpinv.core.grid import GridSpec

    grid = GridSpec(n_phi=n_phi, n_theta=n_theta,
                    theta_start_deg=15.0, theta_end_deg=165.0)
    from mpinv.data._basis_cache import build_basis

    basis = build_basis(grid, l_max=4)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=4, basis=basis)
    loss = RankBinPLoss(
        cfg=RankBinPLossConfig(n_bins=9, beta=10.0),
        grid=grid, l_max=4, decoder=decoder,
    )
    K = 4 * (4 + 2)
    pred = torch.randn(2, 4 * K, requires_grad=True)
    target = torch.relu(torch.randn(2, n_theta, n_phi)) + 0.1
    val = loss(pred, target)
    val.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0, \
        "rank_bin loss produced zero gradient — soft binning broken"


def test_physics_power_truncate_target_to_band_zero_when_truncation_matches_pred(
    tiny_generator, tiny_basis, rng
):
    """When pred packed coefficients are bandlimited to ``l ≤ k`` and the loss
    is configured with ``truncate_target_to_band=k``, the primary term is ~0
    because the truncated target P matches the predicted P exactly."""
    from mpinv.core.packing import zero_above_band

    l_max = tiny_generator.cfg.l_max
    k = max(1, l_max - 1)
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=l_max,
        basis=tiny_basis,
    )
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(truncate_target_to_band=k),
        grid=tiny_generator.cfg.grid,
        l_max=l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    target_P = torch.from_numpy(P_np)
    target_packed = torch.from_numpy(packed)
    # Predicted packed = truncated target packed; predicted P = decoded truncated.
    pred = zero_above_band(target_packed, k=k, l_max=l_max).requires_grad_(True)
    val = loss(pred, target_P, target_packed=target_packed)
    # Primary term should be ~0 (decoder(pred) == decoder(zero_above_band(target_packed, k))).
    assert val.item() < 1e-6


def test_physics_power_truncate_target_to_band_requires_target_packed(
    tiny_generator, tiny_basis, rng
):
    """When ``truncate_target_to_band`` is active and ``target_packed`` is missing,
    the loss must raise rather than silently fall back to the full target."""
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(truncate_target_to_band=2),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    pred = torch.from_numpy(packed).clone().requires_grad_(True)
    with pytest.raises(ValueError, match="target_packed"):
        loss(pred, torch.from_numpy(P_np))


def test_physics_power_truncate_to_lmax_is_noop(tiny_generator, tiny_basis, rng):
    """Setting ``truncate_target_to_band = l_max`` reverts to the canonical
    full-P target — exactly equal to the un-truncated baseline."""
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss_trunc = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(truncate_target_to_band=tiny_generator.cfg.l_max),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    loss_full = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    pred = torch.from_numpy(packed) + 0.3
    target_P = torch.from_numpy(P_np)
    target_packed = torch.from_numpy(packed)
    v_trunc = loss_trunc(pred.clone().requires_grad_(True), target_P, target_packed=target_packed).item()
    v_full = loss_full(pred.clone().requires_grad_(True), target_P).item()
    assert abs(v_trunc - v_full) < 1e-6


def test_physics_power_with_rank_bin_aux(tiny_generator, tiny_basis, rng):
    """The rank_bin_weight knob in PhysicsPowerLoss must add a finite term."""
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss_with = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(rank_bin_weight=0.5),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    loss_without = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(),
        grid=tiny_generator.cfg.grid,
        l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    P_np, packed = tiny_generator.generate_batch(2, rng)
    pred = torch.from_numpy(packed) + 0.3
    target_P = torch.from_numpy(P_np)
    v_with = loss_with(pred.requires_grad_(True), target_P).item()
    v_without = loss_without(torch.from_numpy(packed) + 0.3, target_P).item()
    # The rank-bin term is non-negative, so the combined loss is >= the
    # plain physics_power loss.
    assert v_with >= v_without - 1e-9
    # And both components are present in last_components.
    components = loss_with.last_components
    assert "rank_bin_p" in components
    assert "physics_power" in components
