"""Custom callback-driven training loop (per practice.pdf §1).

The hooks fire in the order of the practice.pdf "universal training loop" surface:
``on_fit_start, on_epoch_start, on_batch_start, on_forward_end, on_loss_end,
on_backward_end, on_step_end, on_batch_end, on_epoch_end, on_validation_end,
on_fit_end``.

Callbacks mutate the shared :class:`TrainingContext` to communicate; sinks (MLflow,
TensorBoard, etc.) implement the same set of hooks but for output rather than
control.

The loop supports:
- AMP (bf16 default, fp16 with GradScaler, fp32 disabled).
- Gradient accumulation (``accum_steps``).
- Per-step learning-rate scheduler steps.
- Sanity assertions on parameter coverage and loss participation at fit start.
- Configurable batch unpacking via ``unpack_batch`` callable on the context (so the
  Trainer doesn't need to know whether the loss expects packed coefs or P).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from mpinv.callbacks.base import Callback
from mpinv.training.amp import AMPConfig, build_amp
from mpinv.training.sanity import (
    sanity_check_loss_participation,
    sanity_check_optimiser_coverage,
)

logger = logging.getLogger(__name__)

LossKind = Literal["coef", "physics"]


@dataclass(slots=True)
class TrainerConfig:
    """Knobs for :class:`Trainer`."""

    max_epochs: int = 50
    accum_steps: int = 1
    log_every_n_steps: int = 10
    sanity_check: bool = True
    device: str = "cpu"
    amp: AMPConfig = field(default_factory=AMPConfig)


@dataclass(slots=True)
class TrainingContext:
    """Mutable container shared across callbacks and the Trainer.

    Attributes
    ----------
    model : nn.Module
    optimiser : Optimizer
    scheduler : LRScheduler or None
    loss_fn : nn.Module
    loss_kind : "coef" | "physics"
        Determines which fields of the batch are passed to the loss.
    train_loader, val_loader : iterables of batches
    epoch, global_step : counters
    last_loss, last_loss_components, current_lr : last seen scalars
    last_eval_metrics : dict from the last validation pass
    stop_requested : bool flag (set by EarlyStopping)
    sinks : sequence of Sink-like objects with ``log_metrics`` and ``log_metric``
    """

    model: nn.Module
    optimiser: Optimizer | None
    scheduler: LRScheduler | None
    loss_fn: Any
    loss_kind: LossKind
    train_loader: Iterable[Any] | None = None
    val_loader: Iterable[Any] | None = None
    epoch: int = 0
    global_step: int = 0
    last_loss: float = float("nan")
    last_loss_components: dict[str, float] | None = None
    last_eval_metrics: dict[str, float] | None = None
    current_lr: float = 0.0
    last_grad_norm: float = float("nan")
    stop_requested: bool = False
    sinks: Sequence[Any] = field(default_factory=list)
    unpack_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for sink in self.sinks:
            sink.log_metrics(metrics, step=self.global_step)


def _default_unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Default batch layout: ``(x_features, y_packed_coeffs, y_power_pattern)``."""
    if isinstance(batch, dict):
        return batch["x"], batch["y_packed"], batch["y_pattern"]
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(
        f"unrecognised batch type {type(batch).__name__}; provide unpack_batch on the context"
    )


class Trainer:
    """The framework's training loop. One canonical implementation."""

    def __init__(self, cfg: TrainerConfig | None = None):
        self.cfg = cfg or TrainerConfig()

    @staticmethod
    def _step_loss(
        loss_fn: Any,
        loss_kind: LossKind,
        pred: torch.Tensor,
        y_packed: torch.Tensor,
        y_pattern: torch.Tensor,
    ) -> torch.Tensor:
        if loss_kind == "coef":
            return loss_fn(pred, y_packed)
        if loss_kind == "physics":
            return loss_fn(pred, y_pattern, target_packed=y_packed)
        raise ValueError(f"unknown loss kind {loss_kind!r}")

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable[Any],
        loss_fn: Any,
        optimiser: Optimizer,
        loss_kind: LossKind = "coef",
        scheduler: LRScheduler | None = None,
        val_loader: Iterable[Any] | None = None,
        callbacks: Sequence[Callback] = (),
        sinks: Sequence[Any] = (),
        unpack_batch: Callable[
            [Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = _default_unpack,
    ) -> TrainingContext:
        device = torch.device(self.cfg.device)
        amp_ctx, scaler = build_amp(self.cfg.amp, device)

        model.to(device)
        try:
            loss_fn.to(device)  # type: ignore[union-attr]
        except AttributeError:
            pass

        ctx = TrainingContext(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            loss_fn=loss_fn,
            loss_kind=loss_kind,
            train_loader=train_loader,
            val_loader=val_loader,
            sinks=list(sinks),
            unpack_batch=unpack_batch,
            device=device,
        )

        if self.cfg.sanity_check:
            sanity_check_optimiser_coverage(model, optimiser)
            try:
                first = next(iter(train_loader))
                xs, yp, yt = unpack_batch(first)
                xs = xs.to(device)
                yp = yp.to(device)
                yt = yt.to(device)
                model.train()
                if loss_kind == "coef":
                    sanity_check_loss_participation(model, loss_fn, xs, yp)
                else:
                    sanity_check_loss_participation(
                        model, lambda pred, _: loss_fn(pred, yt, target_packed=yp), xs, yp
                    )
            except StopIteration:
                logger.warning("sanity check skipped: empty train_loader")

        for cb in callbacks:
            cb.on_fit_start(ctx)
        for sink in sinks:
            sink.on_fit_start(ctx)

        try:
            for epoch in range(1, self.cfg.max_epochs + 1):
                if ctx.stop_requested:
                    break
                ctx.epoch = epoch
                model.train()
                for cb in callbacks:
                    cb.on_epoch_start(ctx)
                for batch_idx, batch in enumerate(train_loader):
                    for cb in callbacks:
                        cb.on_batch_start(ctx)
                    xs, yp, yt = unpack_batch(batch)
                    xs = xs.to(device, non_blocking=True)
                    yp = yp.to(device, non_blocking=True)
                    yt = yt.to(device, non_blocking=True)

                    with amp_ctx:
                        pred = model(xs)
                        loss = self._step_loss(loss_fn, loss_kind, pred, yp, yt)
                        loss = loss / self.cfg.accum_steps

                    for cb in callbacks:
                        cb.on_forward_end(ctx)
                    ctx.last_loss = loss.detach().item() * self.cfg.accum_steps
                    ctx.last_loss_components = (
                        dict(loss_fn.last_components)
                        if hasattr(loss_fn, "last_components")
                        else None
                    )
                    for cb in callbacks:
                        cb.on_loss_end(ctx)

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # Capture grad norm before any zero_grad so logging callbacks see it.
                    try:
                        from mpinv.training.sanity import grad_norm

                        ctx.last_grad_norm = grad_norm(model.parameters())
                    except Exception:
                        ctx.last_grad_norm = float("nan")
                    for cb in callbacks:
                        cb.on_backward_end(ctx)

                    if (batch_idx + 1) % self.cfg.accum_steps == 0:
                        if scaler is not None:
                            scaler.step(optimiser)
                            scaler.update()
                        else:
                            optimiser.step()
                        optimiser.zero_grad(set_to_none=True)
                        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step()
                        ctx.global_step += 1
                    ctx.current_lr = optimiser.param_groups[0]["lr"]
                    for cb in callbacks:
                        cb.on_step_end(ctx)
                    for cb in callbacks:
                        cb.on_batch_end(ctx)

                for cb in callbacks:
                    cb.on_epoch_end(ctx)
                for sink in sinks:
                    sink.on_epoch_end(ctx)
                if (
                    scheduler is not None
                    and isinstance(scheduler, ReduceLROnPlateau)
                    and ctx.last_eval_metrics
                    and "val/loss" in ctx.last_eval_metrics
                ):
                    scheduler.step(ctx.last_eval_metrics["val/loss"])
        finally:
            for cb in callbacks:
                cb.on_fit_end(ctx)
            for sink in sinks:
                sink.on_run_end("FINISHED" if not ctx.stop_requested else "EARLY_STOPPED")

        return ctx
