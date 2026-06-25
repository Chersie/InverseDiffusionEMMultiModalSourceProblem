"""MLP regressor for packed multipole coefficients.

Architectures supported:

- ``flat``: ``[hidden_size] * n_hidden_layers``.
- ``pyramid``: shrinks geometrically from ``hidden_size`` to ``hidden_size_min``.
- ``bottleneck``: a wide head, a narrow waist, a wide tail.
- ``residual``: blocks of two linear layers with a skip every block.

All variants use SiLU activations by default; layer norms optional. Bias on linear
layers is configurable; per practice.pdf p. 19 it is sometimes worth turning off
for stability, especially on the residual variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from mpinv.models.base import BaseModelConfig
from mpinv.models.registry import register_model

Architecture = Literal["flat", "pyramid", "bottleneck", "residual"]


@dataclass(slots=True)
class MLPConfig(BaseModelConfig):
    """Configuration for :class:`MLP`."""

    hidden_size: int = 512
    n_hidden_layers: int = 4
    architecture: Architecture = "flat"
    hidden_size_min: int = 64
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_bias: bool = True
    activation: Literal["silu", "relu", "gelu", "elu"] = "silu"


def _make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"unknown activation {name!r}")


def _layer_widths(cfg: MLPConfig) -> list[int]:
    """Return the list of hidden widths for the chosen architecture."""
    if cfg.architecture == "flat":
        return [cfg.hidden_size] * cfg.n_hidden_layers
    if cfg.architecture == "pyramid":
        if cfg.n_hidden_layers <= 1:
            return [cfg.hidden_size]
        ratio = (cfg.hidden_size_min / cfg.hidden_size) ** (1.0 / (cfg.n_hidden_layers - 1))
        return [
            max(cfg.hidden_size_min, round(cfg.hidden_size * (ratio**i)))
            for i in range(cfg.n_hidden_layers)
        ]
    if cfg.architecture == "bottleneck":
        if cfg.n_hidden_layers < 3:
            return [cfg.hidden_size] * cfg.n_hidden_layers
        mid = cfg.n_hidden_layers // 2
        widths = (
            [cfg.hidden_size] * mid
            + [cfg.hidden_size_min]
            + [cfg.hidden_size] * (cfg.n_hidden_layers - mid - 1)
        )
        return widths
    if cfg.architecture == "residual":
        return [cfg.hidden_size] * cfg.n_hidden_layers
    raise ValueError(f"unknown architecture {cfg.architecture!r}")


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_layer_norm: bool, use_bias: bool, act: str):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.lin1 = nn.Linear(dim, dim, bias=use_bias)
        self.lin2 = nn.Linear(dim, dim, bias=use_bias)
        self.act = _make_activation(act)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.lin1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


def make_backbone(cfg: MLPConfig) -> tuple[nn.Sequential, int]:
    """Build the MLP body up to (but not including) the final output ``Linear``.

    Returns ``(backbone, hidden_dim)`` where ``hidden_dim`` is the width emitted by
    the last hidden layer (the input dim that the per-head ``Linear`` should
    expect). Preserves all four architectures (``flat``, ``pyramid``,
    ``bottleneck``, ``residual``) exactly as :class:`MLP` constructs them.

    This is the single source of truth for the backbone shared between
    :class:`MLP` and :class:`mpinv.models.multi_head_mlp.MultiHeadMLP`. Any
    change here must keep both call-sites in sync (see the round-trip test in
    ``tests/models/test_multi_head_mlp.py``).
    """
    widths = _layer_widths(cfg)
    layers: list[nn.Module] = []
    in_dim = cfg.input_dim
    for w in widths:
        if cfg.architecture == "residual" and in_dim == w:
            layers.append(
                _ResidualBlock(w, cfg.dropout, cfg.use_layer_norm, cfg.use_bias, cfg.activation)
            )
        else:
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(in_dim))
            layers.append(nn.Linear(in_dim, w, bias=cfg.use_bias))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        in_dim = w
    return nn.Sequential(*layers), in_dim


@register_model("mlp")
class MLP(nn.Module):
    """Multilayer perceptron mapping ``(B, input_dim) -> (B, output_dim)`` packed coefs."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        backbone, hidden_dim = make_backbone(cfg)
        head = nn.Linear(hidden_dim, cfg.output_dim, bias=cfg.use_bias)
        self.net = nn.Sequential(*backbone, head)

    @property
    def input_dim(self) -> int:
        return self.cfg.input_dim

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
