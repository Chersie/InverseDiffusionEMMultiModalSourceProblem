"""Gradient-clipping callback (per practice.pdf p. 19)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mpinv.callbacks.base import Callback


@dataclass(slots=True)
class GradClipCallback(Callback):
    """Clip gradient norms (or values) before the optimiser step.

    Hooked to ``on_backward_end`` so it runs *after* loss.backward() but *before*
    optimiser.step().
    """

    max_norm: float | None = 1.0
    max_value: float | None = None
    norm_type: float = 2.0

    def on_backward_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        if self.max_value is not None:
            torch.nn.utils.clip_grad_value_(ctx.model.parameters(), clip_value=self.max_value)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                ctx.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
            )
