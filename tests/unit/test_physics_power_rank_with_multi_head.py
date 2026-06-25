"""End-to-end smoke test: ``physics_power_rank`` through :class:`MultiHeadMLP`.

The plan asserts that the rank-regularised physics-power loss works *unchanged*
on multi-head outputs because heads only change the origin of the
``(B, 4 K)`` packed tensor. This file verifies that:

- The full forward + loss is finite and connects back-propagation to *exactly*
  the parameters that are ``requires_grad=True`` (active head + backbone if
  unfrozen) and to **none** of the frozen / zero-frozen heads.
- The decoded ``P_pred`` is non-negative everywhere (physical constraint of
  the squared-magnitude operator).
- A few SGD steps reduce the loss when only one head is active, which is the
  basic gradient-flow sanity that the staged trainer relies on.
"""

from __future__ import annotations

import torch

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import VSHBasis
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig
from mpinv.models.multi_head_mlp import (
    MultiHeadMLP,
    MultiHeadMLPConfig,
    expected_output_dim,
)
from mpinv.training.staged import apply_stage_policy


def _build_model(L: int, in_dim: int) -> MultiHeadMLP:
    return MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=in_dim, output_dim=expected_output_dim(L), l_max=L,
        hidden_size=8, n_hidden_layers=1, activation="silu",
    ))


def _build_data(
    tiny_grid: GridSpec, tiny_basis: VSHBasis, l_max: int, n: int = 4
):
    cfg = SyntheticGeneratorConfig(grid=tiny_grid, l_max=l_max, mode="gaussian")
    gen = SyntheticGenerator(cfg=cfg, basis=tiny_basis)
    rng = torch.Generator()
    rng.manual_seed(0)
    import numpy as np

    P, packed = gen.generate_batch(n, np.random.default_rng(0))
    return torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(
        packed, dtype=torch.float32
    )


def test_forward_and_loss_runs(
    tiny_grid: GridSpec, tiny_l_max: int, tiny_basis: VSHBasis
):
    L = tiny_l_max
    model = _build_model(L, in_dim=12)
    decoder = DifferentiableMultipoleField(grid=tiny_grid, l_max=L, basis=tiny_basis)
    loss_fn = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(rank_bin_weight=0.1),
        grid=tiny_grid, l_max=L, decoder=decoder,
    )
    P, packed = _build_data(tiny_grid, tiny_basis, L)
    x = torch.randn(P.size(0), 12)
    pred = model(x)
    assert pred.shape == packed.shape
    out = loss_fn(pred, P, target_packed=packed)
    assert torch.isfinite(out)
    assert out.item() > 0


def test_grad_only_flows_to_active_head(
    tiny_grid: GridSpec, tiny_l_max: int, tiny_basis: VSHBasis
):
    """In stage k under freeze_after_stage1 (k > 1), gradients reach exactly
    the active head's parameters and nothing else.
    """
    L = tiny_l_max
    model = _build_model(L, in_dim=12)
    decoder = DifferentiableMultipoleField(grid=tiny_grid, l_max=L, basis=tiny_basis)
    loss_fn = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(rank_bin_weight=0.1),
        grid=tiny_grid, l_max=L, decoder=decoder,
    )
    apply_stage_policy(model, stage_idx=2, backbone_policy="freeze_after_stage1")

    P, packed = _build_data(tiny_grid, tiny_basis, L)
    x = torch.randn(P.size(0), 12)

    model.zero_grad(set_to_none=True)
    pred = model(x)
    out = loss_fn(pred, P, target_packed=packed)
    out.backward()

    # Active head (index 1) MUST have non-zero gradients on weight.
    assert model.heads[1].weight.grad is not None
    assert model.heads[1].weight.grad.abs().sum() > 0
    # Frozen / zero-frozen heads have grad = None (they are excluded from the
    # graph by requires_grad=False).
    for j in [0, 2, 3]:
        assert model.heads[j].weight.grad is None
    # Backbone is frozen at stage 2 under freeze_after_stage1.
    for p in model.backbone.parameters():
        assert p.grad is None


def test_predicted_power_pattern_is_nonnegative(
    tiny_grid: GridSpec, tiny_l_max: int, tiny_basis: VSHBasis
):
    L = tiny_l_max
    model = _build_model(L, in_dim=12)
    decoder = DifferentiableMultipoleField(
        grid=tiny_grid, l_max=L, basis=tiny_basis
    )
    P, _ = _build_data(tiny_grid, tiny_basis, L)
    x = torch.randn(P.size(0), 12)
    with torch.no_grad():
        P_pred = decoder(model(x))
    assert torch.all(P_pred >= 0.0)


def test_a_few_sgd_steps_reduce_loss(
    tiny_grid: GridSpec, tiny_l_max: int, tiny_basis: VSHBasis
):
    L = tiny_l_max
    model = _build_model(L, in_dim=12)
    decoder = DifferentiableMultipoleField(
        grid=tiny_grid, l_max=L, basis=tiny_basis
    )
    loss_fn = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(rank_bin_weight=0.1),
        grid=tiny_grid, l_max=L, decoder=decoder,
    )
    apply_stage_policy(model, stage_idx=1, backbone_policy="freeze_after_stage1")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    P, packed = _build_data(tiny_grid, tiny_basis, L, n=8)
    x = torch.randn(P.size(0), 12)

    losses: list[float] = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        out = loss_fn(pred, P, target_packed=packed)
        out.backward()
        opt.step()
        losses.append(float(out.item()))
    assert losses[-1] < losses[0]
