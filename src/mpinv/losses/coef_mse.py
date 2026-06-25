"""Coefficient-space MSE loss.

The simplest baseline: mean squared error between predicted and target packed
coefficient vectors. Optional per-component weights enable up-weighting of low orders
when the user wants it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from mpinv.core.shapes import assert_packed_coeffs
from mpinv.losses.registry import register_loss


@dataclass(slots=True)
class CoefMSEConfig:
    """Knobs for :class:`CoefMSE`."""

    reduction: str = "mean"


@register_loss("coef_mse")
class CoefMSE(nn.Module):
    """MSE in packed coefficient space."""

    def __init__(self, cfg: CoefMSEConfig | None = None):
        super().__init__()
        self.cfg = cfg or CoefMSEConfig()
        self._last_components: dict[str, float] = {}

    @property
    def last_components(self) -> Mapping[str, float]:
        return self._last_components

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert_packed_coeffs(pred, name="pred")
        assert_packed_coeffs(target, name="target")
        if pred.shape != target.shape:
            raise ValueError(
                f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}"
            )
        diff = pred - target
        sq = diff.pow(2)
        if self.cfg.reduction == "mean":
            loss = sq.mean()
        elif self.cfg.reduction == "sum":
            loss = sq.sum()
        elif self.cfg.reduction == "none":
            loss = sq
        else:
            raise ValueError(f"unknown reduction {self.cfg.reduction!r}")
        self._last_components = {"coef_mse": loss.detach().mean().item()}
        return loss
