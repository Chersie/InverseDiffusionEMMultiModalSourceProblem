"""Checkpoint callback: save model + optimiser + scheduler state on a cadence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckpointCallback(Callback):
    """Save state every N epochs (and a final snapshot at fit end)."""

    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last: int = 3
    save_best_metric: str | None = "val/loss"
    higher_is_better: bool = False

    _best_value: float = float("inf")

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.higher_is_better:
            self._best_value = -float("inf")

    def _save(self, ctx, name: str) -> Path:  # type: ignore[no-untyped-def]
        path = Path(self.output_dir) / f"{name}.pt"
        state = {
            "model": ctx.model.state_dict(),
            "optimiser": ctx.optimiser.state_dict() if ctx.optimiser is not None else None,
            "scheduler": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
            "epoch": ctx.epoch,
            "global_step": ctx.global_step,
            "best_value": self._best_value,
        }
        torch.save(state, path)
        logger.info("saved checkpoint: %s", path)
        return path

    def _prune(self) -> None:
        files = sorted(Path(self.output_dir).glob("epoch_*.pt"))
        for old in files[: -self.keep_last]:
            old.unlink(missing_ok=True)

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        if (ctx.epoch % self.save_every_n_epochs) == 0:
            self._save(ctx, f"epoch_{ctx.epoch:04d}")
            self._prune()
        if self.save_best_metric and self.save_best_metric in (ctx.last_eval_metrics or {}):
            v = ctx.last_eval_metrics[self.save_best_metric]
            improved = v > self._best_value if self.higher_is_better else v < self._best_value
            if improved:
                self._best_value = v
                self._save(ctx, "best")

    def on_fit_end(self, ctx, status: str = "FINISHED") -> None:  # type: ignore[no-untyped-def]
        self._save(ctx, "last")
