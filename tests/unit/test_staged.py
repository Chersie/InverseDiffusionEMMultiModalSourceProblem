"""Tests for :mod:`mpinv.training.staged`.

Covers the stage-protocol contracts called out in
[.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md](../../.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md):

- Per-stage requires_grad layout matches the policy ("backbone freeze after
  stage 1" / "trainable always" / "lower lr").
- ``apply_stage_policy`` zeroes future heads and reinitialises the active head
  (verified by Frobenius norm > 0 after ``reinit_active_head=True``).
- ``build_stage_optimiser`` covers exactly the trainable parameters and uses
  separate param groups when ``lower_lr_after_stage1`` is requested.
- The framework's ``sanity_check_optimiser_coverage`` accepts every stage.
- A multi-stage ``StagedTrainer.fit`` reduces ``coef_mse`` for stage 1 and
  keeps zero-frozen future heads at zero across all stages.
- Transplant + ``starting_stage`` interaction: stages 1..starting_stage-1
  are skipped, transplanted heads stay frozen.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mpinv.callbacks.base import Callback
from mpinv.losses.coef_mse import CoefMSE, CoefMSEConfig
from mpinv.models.multi_head_mlp import (
    MultiHeadMLP,
    MultiHeadMLPConfig,
    expected_output_dim,
    transplant_heads,
)
from mpinv.training.optim import OptimiserConfig
from mpinv.training.sanity import sanity_check_optimiser_coverage
from mpinv.training.staged import (
    StagedTrainer,
    StagedTrainerConfig,
    apply_stage_policy,
    build_stage_optimiser,
)


def _make_mh(L: int = 3, *, hidden: int = 8, layers: int = 1) -> MultiHeadMLP:
    return MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(L), l_max=L,
        hidden_size=hidden, n_hidden_layers=layers, activation="silu",
    ))


# ---------------------------------------------------------------------------
# apply_stage_policy
# ---------------------------------------------------------------------------


def test_apply_stage_policy_freeze_after_stage1():
    m = _make_mh(L=4)
    apply_stage_policy(m, stage_idx=1, backbone_policy="freeze_after_stage1")
    s = m.trainable_summary()
    assert s == {"backbone": True, "head_0": True, "head_1": False, "head_2": False, "head_3": False}

    apply_stage_policy(m, stage_idx=2, backbone_policy="freeze_after_stage1")
    s = m.trainable_summary()
    assert s == {"backbone": False, "head_0": False, "head_1": True, "head_2": False, "head_3": False}

    apply_stage_policy(m, stage_idx=4, backbone_policy="freeze_after_stage1")
    s = m.trainable_summary()
    assert s == {"backbone": False, "head_0": False, "head_1": False, "head_2": False, "head_3": True}


def test_apply_stage_policy_trainable_always():
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="trainable_always")
    s = m.trainable_summary()
    assert s["backbone"] is True
    assert s["head_0"] is False  # earlier head frozen at trained weights
    assert s["head_1"] is True
    assert s["head_2"] is False  # future head zero-frozen


def test_apply_stage_policy_lower_lr_keeps_backbone_trainable():
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="lower_lr_after_stage1")
    # Backbone is "trainable" (requires_grad=True); the per-group LR scaling
    # happens inside build_stage_optimiser, not here.
    assert m.trainable_summary()["backbone"] is True


def test_apply_stage_policy_all_trainable_active_boost_keeps_previous_heads():
    """Under ``all_trainable_active_boost`` previous heads stay trainable
    (they get a lower LR via the optimiser, but they continue to learn)."""
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="all_trainable_active_boost")
    s = m.trainable_summary()
    assert s["backbone"] is True
    assert s["head_0"] is True  # previous head NOT frozen
    assert s["head_1"] is True  # active head trainable
    assert s["head_2"] is False  # future head zero-frozen


def test_apply_stage_policy_zeroes_future_heads_when_requested():
    m = _make_mh(L=4)
    apply_stage_policy(m, stage_idx=1, zero_init_future_heads=True)
    for j in [1, 2, 3]:
        assert torch.all(m.heads[j].weight == 0.0)
        if m.heads[j].bias is not None:
            assert torch.all(m.heads[j].bias == 0.0)


def test_apply_stage_policy_skips_zero_when_disabled():
    m = _make_mh(L=3)
    # Capture initial random weights for head 1 and 2 before the call.
    before_h1 = m.heads[1].weight.detach().clone()
    apply_stage_policy(m, stage_idx=1, zero_init_future_heads=False)
    # Future heads keep their default-initialised values, just frozen.
    assert torch.equal(m.heads[1].weight, before_h1)
    assert all(not p.requires_grad for p in m.heads[1].parameters())


def test_apply_stage_policy_reinits_active_head():
    m = _make_mh(L=3)
    # Zero head 1 first; with reinit_active_head=True, activating it should
    # restore non-zero weights via Linear.reset_parameters.
    m.zero_init_head(1)
    apply_stage_policy(m, stage_idx=2, reinit_active_head=True)
    # Frobenius norm > 0 after reinit.
    assert m.heads[1].weight.detach().norm().item() > 0.0


def test_apply_stage_policy_skips_reinit_when_disabled():
    m = _make_mh(L=3)
    m.zero_init_head(1)
    apply_stage_policy(m, stage_idx=2, reinit_active_head=False)
    assert torch.all(m.heads[1].weight == 0.0)


# ---------------------------------------------------------------------------
# build_stage_optimiser + optimiser coverage
# ---------------------------------------------------------------------------


def test_build_stage_optimiser_covers_trainable_params():
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="freeze_after_stage1")
    opt = build_stage_optimiser(
        m, OptimiserConfig(name="adamw", lr=1e-3),
        stage_idx=2, backbone_policy="freeze_after_stage1",
    )
    sanity_check_optimiser_coverage(m, opt)


def test_build_stage_optimiser_covers_at_every_stage():
    m = _make_mh(L=4)
    for stage_idx in range(1, m.n_heads + 1):
        apply_stage_policy(m, stage_idx, backbone_policy="freeze_after_stage1")
        opt = build_stage_optimiser(
            m, OptimiserConfig(name="adamw", lr=1e-3),
            stage_idx=stage_idx, backbone_policy="freeze_after_stage1",
        )
        sanity_check_optimiser_coverage(m, opt)


def test_build_stage_optimiser_lower_lr_uses_split_groups():
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="lower_lr_after_stage1")
    opt = build_stage_optimiser(
        m, OptimiserConfig(name="adamw", lr=1e-3),
        stage_idx=2, backbone_policy="lower_lr_after_stage1",
        backbone_lr_factor=0.1,
    )
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[0] == 1e-4
    assert lrs[1] == 1e-3
    sanity_check_optimiser_coverage(m, opt)


def test_build_stage_optimiser_lower_lr_stage1_is_single_group():
    """At stage 1 even under lower_lr policy the backbone runs at the head LR
    (the user's LR halving only kicks in *after* stage 1)."""
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=1, backbone_policy="lower_lr_after_stage1")
    opt = build_stage_optimiser(
        m, OptimiserConfig(name="adamw", lr=1e-3),
        stage_idx=1, backbone_policy="lower_lr_after_stage1",
        backbone_lr_factor=0.1,
    )
    assert len(opt.param_groups) == 1


def test_build_stage_optimiser_active_boost_uses_two_groups_with_low_and_full_lr():
    """``all_trainable_active_boost`` at stage > 1 builds two param groups: a
    low-LR group covering backbone + previous heads, and a full-LR group for
    the active head. Sanity check covers every trainable parameter."""
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=2, backbone_policy="all_trainable_active_boost")
    opt = build_stage_optimiser(
        m, OptimiserConfig(name="adamw", lr=1e-3),
        stage_idx=2, backbone_policy="all_trainable_active_boost",
        backbone_lr_factor=0.1,
    )
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[0] == 1e-4
    assert lrs[1] == 1e-3
    sanity_check_optimiser_coverage(m, opt)


def test_build_stage_optimiser_active_boost_stage1_is_single_group():
    """At stage 1 the active-boost policy collapses to a single full-lr group
    (no previous heads exist)."""
    m = _make_mh(L=3)
    apply_stage_policy(m, stage_idx=1, backbone_policy="all_trainable_active_boost")
    opt = build_stage_optimiser(
        m, OptimiserConfig(name="adamw", lr=1e-3),
        stage_idx=1, backbone_policy="all_trainable_active_boost",
        backbone_lr_factor=0.1,
    )
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-3
    sanity_check_optimiser_coverage(m, opt)


# ---------------------------------------------------------------------------
# Multi-stage StagedTrainer integration
# ---------------------------------------------------------------------------


def _make_dataset(L: int, n: int = 16, in_dim: int = 4):
    torch.manual_seed(0)
    x = torch.randn(n, in_dim)
    y_packed = torch.randn(n, expected_output_dim(L)) * 0.1
    y_pattern = torch.zeros(n, 179, 360)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return (x[i], y_packed[i], y_pattern[i])

    return _DS(), DataLoader(_DS(), batch_size=4)


def test_staged_fit_runs_all_stages_and_reports():
    L = 3
    m = _make_mh(L)
    _, loader = _make_dataset(L)
    loss = CoefMSE(CoefMSEConfig())
    trainer = StagedTrainer(StagedTrainerConfig(
        stage_max_epochs=1, backbone_policy="freeze_after_stage1",
    ))
    reports = trainer.fit(
        m, loader, loss, OptimiserConfig(name="adamw", lr=1e-2),
        loss_kind="coef",
    )
    assert len(reports) == m.n_heads
    for k, r in enumerate(reports, start=1):
        assert r.stage_idx == k
        assert r.active_head_idx == k - 1
        assert r.epochs_run >= 1


def test_staged_fit_zero_frozen_heads_remain_zero_across_stage():
    L = 3
    m = _make_mh(L)
    _, loader = _make_dataset(L)
    loss = CoefMSE(CoefMSEConfig())
    trainer = StagedTrainer(StagedTrainerConfig(
        stage_max_epochs=2, backbone_policy="freeze_after_stage1",
        zero_init_future_heads=True,
    ))
    # Run only stage 1; heads 1, 2 must stay zero.
    trainer.fit_one_stage(
        m, stage_idx=1,
        train_loader=loader,
        loss_fn=loss, optim_cfg=OptimiserConfig(name="adamw", lr=1e-2),
        loss_kind="coef",
    )
    for j in [1, 2]:
        assert torch.all(m.heads[j].weight == 0.0)
        if m.heads[j].bias is not None:
            assert torch.all(m.heads[j].bias == 0.0)


def test_staged_fit_decreases_loss_in_stage1():
    L = 3
    m = _make_mh(L, hidden=16, layers=2)
    _, loader = _make_dataset(L, n=32)
    loss = CoefMSE(CoefMSEConfig())
    # Fit a bunch of epochs in stage 1 only and assert the *final* training
    # batch loss is strictly lower than the *initial* one. We capture both via
    # a tiny callback rather than fight pytorch's internal logging surface.
    losses: list[float] = []

    class _Capture(Callback):
        def on_loss_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
            losses.append(ctx.last_loss)

    trainer = StagedTrainer(StagedTrainerConfig(
        stage_max_epochs=8, backbone_policy="freeze_after_stage1",
    ))
    trainer.fit_one_stage(
        m, stage_idx=1,
        train_loader=loader, loss_fn=loss,
        optim_cfg=OptimiserConfig(name="adamw", lr=5e-2),
        loss_kind="coef",
        callbacks=[_Capture()],
    )
    assert len(losses) > 4
    assert losses[-1] < losses[0]


def test_staged_truncate_target_to_active_band_updates_loss_per_stage():
    """When ``truncate_target_to_active_band=True`` is set on the StagedTrainerConfig,
    the per-stage loop mutates ``loss_fn.cfg.truncate_target_to_band`` to
    ``max(group)`` of the active head's l-band group at the start of each stage.

    Verified by running stage 1 then stage 2 on a tiny multi-head model and
    asserting the cfg reflects the new band each time.
    """
    from mpinv.core.grid import GridSpec
    from mpinv.data._basis_cache import build_basis
    from mpinv.losses.differentiable_field import DifferentiableMultipoleField
    from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig

    L = 3
    grid = GridSpec(n_phi=32, n_theta=16, theta_start_deg=15.0, theta_end_deg=165.0)
    basis = build_basis(grid, l_max=L)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=L, basis=basis)
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(),
        grid=grid, l_max=L, decoder=decoder,
    )
    m = MultiHeadMLP(MultiHeadMLPConfig(
        input_dim=4, output_dim=expected_output_dim(L), l_max=L,
        hidden_size=8, n_hidden_layers=1, activation="silu",
    ))

    # Build a tiny dataset against this grid (P pattern shape must match).
    torch.manual_seed(0)
    n = 4
    x = torch.randn(n, 4)
    y_packed = torch.randn(n, expected_output_dim(L)) * 0.1
    y_pattern = torch.zeros(n, grid.n_theta, grid.n_phi)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return (x[i], y_packed[i], y_pattern[i])

    loader = DataLoader(_DS(), batch_size=2)

    trainer = StagedTrainer(StagedTrainerConfig(
        stage_max_epochs=1, backbone_policy="freeze_after_stage1",
        truncate_target_to_active_band=True,
    ))
    trainer.fit_one_stage(
        m, stage_idx=1, train_loader=loader, loss_fn=loss,
        optim_cfg=OptimiserConfig(name="adamw", lr=1e-3),
        loss_kind="physics",
    )
    assert loss.cfg.truncate_target_to_band == max(m.groups[0])

    trainer.fit_one_stage(
        m, stage_idx=2, train_loader=loader, loss_fn=loss,
        optim_cfg=OptimiserConfig(name="adamw", lr=1e-3),
        loss_kind="physics",
    )
    assert loss.cfg.truncate_target_to_band == max(m.groups[1])


def test_staged_starting_stage_skips_earlier_stages():
    """``starting_stage > 1`` runs only the listed stages and leaves head
    weights from heads with index < starting_stage-1 untouched (here we
    explicitly transplant them first)."""
    L = 4
    src = _make_mh(L, hidden=8, layers=1)
    dst = _make_mh(L, hidden=8, layers=1)
    info = transplant_heads(src, dst, freeze_src_heads=True)
    assert info["transplanted_head_indices"] == [0, 1, 2, 3]

    # Snapshot head 0 weights — transplanted-and-frozen, so they must NOT
    # change after staged training starting from stage 3.
    head0_w = dst.heads[0].weight.detach().clone()
    head1_w = dst.heads[1].weight.detach().clone()

    _, loader = _make_dataset(L)
    loss = CoefMSE(CoefMSEConfig())
    trainer = StagedTrainer(StagedTrainerConfig(
        stage_max_epochs=1, backbone_policy="freeze_after_stage1",
        starting_stage=3,
    ))
    reports = trainer.fit(
        dst, loader, loss, OptimiserConfig(name="adamw", lr=1e-2),
        loss_kind="coef",
    )
    # Only stages 3, 4 run.
    assert [r.stage_idx for r in reports] == [3, 4]
    # Frozen heads' weights are unchanged.
    assert torch.equal(dst.heads[0].weight, head0_w)
    assert torch.equal(dst.heads[1].weight, head1_w)
