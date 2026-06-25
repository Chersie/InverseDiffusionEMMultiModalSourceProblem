"""Linear baselines: a single ``nn.Linear`` regressor.

In the MSE objective, gradient descent on a single ``Linear`` layer converges to the
ordinary-least-squares solution. With ``weight_decay > 0`` (handled by the AdamW
optimiser in the framework) it converges to Ridge regression. With L1
regularisation in the loss, it would converge to Lasso; we do not enable that path
in the baseline because the framework's loss layer is MSE/physics-power.

These models share the framework's :class:`Model` protocol — input shape
``(B, input_dim)``, output shape ``(B, output_dim)`` packed coefficients — so
swapping ``model=mlp_small`` for ``model=linear_baseline`` just changes the
architecture, not the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mpinv.models.base import BaseModelConfig
from mpinv.models.registry import register_model


@dataclass(slots=True)
class LinearBaselineConfig(BaseModelConfig):
    """Knobs for :class:`LinearBaseline`."""

    use_bias: bool = True


@register_model("linear")
class LinearBaseline(nn.Module):
    """Single ``Linear`` layer regressor."""

    def __init__(self, cfg: LinearBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.lin = nn.Linear(cfg.input_dim, cfg.output_dim, bias=cfg.use_bias)

    @property
    def input_dim(self) -> int:
        return self.cfg.input_dim

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
