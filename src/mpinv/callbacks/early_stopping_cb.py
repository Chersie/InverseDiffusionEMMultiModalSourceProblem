"""Early-stopping callback. Watches a metric on ``ctx.last_eval_metrics``."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EarlyStoppingCallback(Callback):
    """Stop training if ``metric`` does not improve for ``patience`` epochs."""

    metric: str = "val/loss"
    patience: int = 10
    min_delta: float = 1e-6
    higher_is_better: bool = False

    _best: float = float("inf")
    _bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.higher_is_better:
            self._best = -float("inf")

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        metrics = ctx.last_eval_metrics or {}
        if self.metric not in metrics:
            return
        v = metrics[self.metric]
        improved = (
            (v > self._best + self.min_delta)
            if self.higher_is_better
            else (v < self._best - self.min_delta)
        )
        if improved:
            self._best = v
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs >= self.patience:
                logger.info("early stopping after %d epochs without improvement", self._bad_epochs)
                ctx.stop_requested = True
