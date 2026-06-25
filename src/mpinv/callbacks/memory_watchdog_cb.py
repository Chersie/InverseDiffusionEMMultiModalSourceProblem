"""Memory watchdog callback: log RSS and CUDA memory each epoch.

If ``hard_limit_mb`` is set and exceeded, sets ``ctx.stop_requested = True`` so the
trainer halts before the OS kills the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mpinv.callbacks.base import Callback

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryWatchdogCallback(Callback):
    """Track RSS/CUDA memory; stop training if a hard limit is exceeded."""

    hard_limit_mb: int | None = None

    def _rss_mb(self) -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return float("nan")

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        rss = self._rss_mb()
        metrics = {"perf/rss_mb": rss}
        try:
            import torch

            if torch.cuda.is_available():
                metrics["perf/cuda_alloc_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                metrics["perf/cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass
        ctx.log_metrics(metrics)
        if self.hard_limit_mb and rss > self.hard_limit_mb:
            logger.warning(
                "memory watchdog: RSS %.0f MB > limit %d MB; stopping", rss, self.hard_limit_mb
            )
            ctx.stop_requested = True
