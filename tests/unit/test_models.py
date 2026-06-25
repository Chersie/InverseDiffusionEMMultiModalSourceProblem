"""Tests for model layer and registry."""

from __future__ import annotations

import torch

from mpinv.models.mlp import MLP, MLPConfig
from mpinv.models.registry import MODELS


def test_mlp_registered():
    assert "mlp" in MODELS


def test_mlp_flat_forward():
    cfg = MLPConfig(
        input_dim=16, output_dim=32, hidden_size=64, n_hidden_layers=2, architecture="flat"
    )
    mlp = MLP(cfg)
    x = torch.randn(5, 16)
    y = mlp(x)
    assert y.shape == (5, 32)
    assert mlp.num_parameters() > 0


def test_mlp_pyramid_widths():
    cfg = MLPConfig(
        input_dim=16,
        output_dim=32,
        hidden_size=512,
        hidden_size_min=64,
        n_hidden_layers=4,
        architecture="pyramid",
    )
    mlp = MLP(cfg)
    y = mlp(torch.randn(3, 16))
    assert y.shape == (3, 32)


def test_mlp_residual_forward():
    cfg = MLPConfig(
        input_dim=64, output_dim=8, hidden_size=64, n_hidden_layers=4, architecture="residual"
    )
    mlp = MLP(cfg)
    y = mlp(torch.randn(2, 64))
    assert y.shape == (2, 8)
