"""Logging callback: emit loss / lr / grad-norm / step-time / batch-time every N steps.

Per practice.pdf p. 16. Sinks include stdout (always) and any registered MLflow sink.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoggingCallback(Callback):
    """Emit per-step metrics every ``log_every_n_steps`` steps.

    Metrics emitted: ``loss``, ``lr``, ``grad_norm``, ``step_time_ms``,
    ``batch_time_ms`` (per practice.pdf p. 16). Per-loss-component values from
    ``loss_fn.last_components`` are also forwarded.
    """

    log_every_n_steps: int = 10
    _step_t0: float = 0.0
    _batch_t0: float = 0.0

    def on_batch_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self._batch_t0 = time.perf_counter()

    def on_step_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self._step_t0 = time.perf_counter()

    def on_batch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        if (ctx.global_step % self.log_every_n_steps) != 0:
            return
        step_time_ms = (time.perf_counter() - self._step_t0) * 1000.0
        batch_time_ms = (time.perf_counter() - self._batch_t0) * 1000.0
        metrics: dict[str, float] = {
            "train/loss": ctx.last_loss,
            "train/lr": ctx.current_lr,
            "train/grad_norm": ctx.last_grad_norm,
            "perf/step_time_ms": step_time_ms,
            "perf/batch_time_ms": batch_time_ms,
        }
        for k, v in (ctx.last_loss_components or {}).items():
            metrics[f"train/{k}"] = v
        ctx.log_metrics(metrics)
        logger.info(
            "epoch=%d step=%d loss=%.6f lr=%.2e gn=%.3f bt=%.1fms",
            ctx.epoch,
            ctx.global_step,
            ctx.last_loss,
            ctx.current_lr,
            metrics["train/grad_norm"],
            batch_time_ms,
        )
