"""Timing callback: track epoch wall time."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimingCallback(Callback):
    """Log epoch wall time."""

    _t0: float = 0.0

    def on_epoch_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self._t0 = time.perf_counter()

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        elapsed = time.perf_counter() - self._t0
        ctx.log_metrics({"perf/epoch_time_s": elapsed})
        logger.info("epoch %d done in %.2fs", ctx.epoch, elapsed)
