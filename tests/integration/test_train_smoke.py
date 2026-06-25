"""End-to-end smoke training on the tiny grid (no MLflow)."""

from __future__ import annotations

import numpy as np
import pytest

from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import make_loaders
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig
from mpinv.losses.coef_mse import CoefMSE, CoefMSEConfig
from mpinv.models.mlp import MLP, MLPConfig
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig


@pytest.mark.integration
def test_trainer_smoke(tiny_generator):
    rng_t = np.random.default_rng(0)
    rng_v = np.random.default_rng(1)
    P_train, packed_train = tiny_generator.generate_batch(128, rng_t)
    P_val, packed_val = tiny_generator.generate_batch(32, rng_v)

    feat = PowerPCAPipeline(cfg=PowerPCAPipelineConfig(pca_components=8))
    feat.fit(P_train=P_train)
    z_train = feat.transform(P=P_train)
    z_val = feat.transform(P=P_val)

    K = tiny_generator.n_modes
    model = MLP(MLPConfig(input_dim=8, output_dim=4 * K, hidden_size=32, n_hidden_layers=2))
    loss_fn = CoefMSE(CoefMSEConfig())
    opt = build_optimiser(model, OptimiserConfig(name="adamw", lr=1e-3))

    train_loader, val_loader = make_loaders(
        P_train,
        packed_train,
        z_train,
        P_val,
        packed_val,
        z_val,
        batch_size=32,
        num_workers=0,
    )

    trainer = Trainer(TrainerConfig(max_epochs=2, log_every_n_steps=1))
    ctx = trainer.fit(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimiser=opt,
        loss_kind="coef",
        val_loader=val_loader,
        callbacks=[LoggingCallback(log_every_n_steps=1), ValidationCallback(every_n_epochs=1)],
    )
    assert np.isfinite(ctx.last_loss)
    assert (ctx.last_eval_metrics or {}).get("val/loss") is not None
