"""Smoke tests for callbacks."""

from __future__ import annotations

import torch

from mpinv.callbacks.checkpoint_cb import CheckpointCallback
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.training.trainer import TrainingContext


class _StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)


def _make_ctx() -> TrainingContext:
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    return TrainingContext(
        model=model, optimiser=opt, scheduler=None, loss_fn=None, loss_kind="coef"
    )


def test_grad_clip_clips(monkeypatch):
    ctx = _make_ctx()
    for p in ctx.model.parameters():
        p.grad = torch.full_like(p, 100.0)
    cb = GradClipCallback(max_norm=1.0)
    cb.on_backward_end(ctx)
    norms = [p.grad.abs().max().item() for p in ctx.model.parameters()]
    assert all(n < 100 for n in norms)


def test_early_stopping_triggers():
    cb = EarlyStoppingCallback(metric="val/loss", patience=2, min_delta=0.0)
    ctx = _make_ctx()
    ctx.last_eval_metrics = {"val/loss": 1.0}
    cb.on_epoch_end(ctx)
    ctx.last_eval_metrics = {"val/loss": 1.0}
    cb.on_epoch_end(ctx)
    ctx.last_eval_metrics = {"val/loss": 1.0}
    cb.on_epoch_end(ctx)
    assert ctx.stop_requested is True


def test_checkpoint_saves(tmp_path):
    cb = CheckpointCallback(output_dir=str(tmp_path), save_every_n_epochs=1, keep_last=2)
    ctx = _make_ctx()
    ctx.epoch = 1
    cb.on_epoch_end(ctx)
    assert (tmp_path / "epoch_0001.pt").exists()
    cb.on_fit_end(ctx)
    assert (tmp_path / "last.pt").exists()


def test_feature_extractor_pickle_roundtrip():
    """Fitted feature extractors must survive a pickle roundtrip and produce
    bit-equal transform outputs after reload.

    Companion regression test for the run_real_augmented.py checkpointing
    flow added 2026-05-13: feature extractors are saved as a single pickle
    after fit() because they are frozen for the duration of training. If
    pickle ever stops capturing the fitted scaler / PCA / mask state, the
    saved feature_extractor.pkl would still load but transform() would
    produce different outputs, silently corrupting any reloaded inference.
    """
    import pickle

    import numpy as np

    from mpinv.features.raw_flat import RawFlattenPipeline, RawFlattenPipelineConfig

    rng = np.random.default_rng(0)
    P = (rng.standard_normal((8, 8, 12)).astype(np.float32) ** 2 + 0.1)
    feat = RawFlattenPipeline(RawFlattenPipelineConfig())
    feat.fit(P_train=P)
    z_before = feat.transform(P=P)
    blob = pickle.dumps({"feature_extractor": feat})
    feat_reloaded = pickle.loads(blob)["feature_extractor"]
    z_after = feat_reloaded.transform(P=P)
    np.testing.assert_array_equal(z_before, z_after)
    assert feat_reloaded.feature_dim == feat.feature_dim


def test_validation_callback_physics_passes_target_packed(tiny_generator, tiny_basis):
    """Regression: ``ValidationCallback`` must call physics losses with the
    same kwargs the trainer uses. ``PhysicsPowerLoss(coef_aux_weight > 0)``
    raises if ``target_packed`` is missing, so this test guards the bug from
    2026-05-13 (validation_cb dropped the kwarg even though
    ``trainer._step_loss`` passed it).
    """
    import numpy as np
    from torch.utils.data import DataLoader

    from mpinv.callbacks.validation_cb import ValidationCallback
    from mpinv.cli._builders import _ArrayDataset
    from mpinv.losses.differentiable_field import DifferentiableMultipoleField
    from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig

    rng = np.random.default_rng(0)
    P, packed = tiny_generator.generate_batch(4, rng)
    K = tiny_generator.n_modes
    decoder = DifferentiableMultipoleField(
        grid=tiny_generator.cfg.grid, l_max=tiny_generator.cfg.l_max,
        basis=tiny_basis,
    )
    loss = PhysicsPowerLoss(
        cfg=PhysicsPowerLossConfig(coef_aux_weight=0.1),
        grid=tiny_generator.cfg.grid, l_max=tiny_generator.cfg.l_max,
        decoder=decoder,
    )
    val_loader = DataLoader(
        _ArrayDataset(np.zeros((4, 4 * K), dtype=np.float32), packed, P),
        batch_size=2, shuffle=False, num_workers=0,
    )

    class _StubModel2(torch.nn.Module):
        def __init__(self, K_):
            super().__init__()
            self.K = K_

        def forward(self, x):
            return torch.zeros(x.shape[0], 4 * self.K, dtype=torch.float32)

    from mpinv.training.trainer import _default_unpack

    model = _StubModel2(K)
    opt = torch.optim.SGD(list(model.parameters()) + [torch.zeros(1, requires_grad=True)],
                          lr=1e-3)
    ctx = TrainingContext(
        model=model, optimiser=opt, scheduler=None,
        loss_fn=loss, loss_kind="physics",
        val_loader=val_loader,
        unpack_batch=_default_unpack,
    )
    ctx.epoch = 1
    cb = ValidationCallback(every_n_epochs=1)
    # Before the fix this raised:
    #   ValueError: PhysicsPowerLoss configured with coef_aux_weight > 0
    #               needs target_packed
    cb.on_epoch_end(ctx)
    assert ctx.last_eval_metrics is not None
    assert "val/loss" in ctx.last_eval_metrics
