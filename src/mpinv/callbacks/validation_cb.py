"""Validation callback: run the model on a validation loader at epoch end."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationCallback(Callback):
    """Compute validation loss + a small set of metrics every N epochs."""

    every_n_epochs: int = 1

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        if (ctx.epoch % self.every_n_epochs) != 0:
            return
        if ctx.val_loader is None:
            return
        ctx.model.eval()
        n = 0
        total = 0.0
        coef_mse_total = 0.0
        with torch.no_grad():
            for batch in ctx.val_loader:
                x, y_packed, y_pattern = ctx.unpack_batch(batch)
                pred = ctx.model(x)
                if ctx.loss_kind == "coef":
                    loss = ctx.loss_fn(pred, y_packed)
                else:
                    # Physics losses must receive ``target_packed`` so the
                    # optional auxiliary coef-MSE term works (PhysicsPowerLoss
                    # raises ValueError when coef_aux_weight > 0 and
                    # target_packed is None). Mirrors the same call in
                    # ``mpinv.training.trainer.Trainer._step_loss`` so train
                    # and val see the same loss surface.
                    loss = ctx.loss_fn(pred, y_pattern, target_packed=y_packed)
                total += loss.item() * x.shape[0]
                coef_mse_total += (pred - y_packed).pow(2).mean().item() * x.shape[0]
                n += x.shape[0]
        ctx.model.train()
        if n == 0:
            return
        metrics = {
            "val/loss": total / n,
            "val/coef_mse": coef_mse_total / n,
        }
        ctx.last_eval_metrics = (ctx.last_eval_metrics or {}) | metrics
        ctx.log_metrics(metrics)
        logger.info("validation: %s", metrics)
