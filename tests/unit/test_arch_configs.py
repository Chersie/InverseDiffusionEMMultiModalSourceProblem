"""Smoke tests for every model config preset."""

from __future__ import annotations

import pytest
import torch

from mpinv.models.linear_baselines import LinearBaseline, LinearBaselineConfig
from mpinv.models.mlp import MLP, MLPConfig


@pytest.mark.parametrize("arch", ["flat", "pyramid", "bottleneck", "residual"])
def test_mlp_arch_forward(arch):
    cfg = MLPConfig(
        input_dim=32,
        output_dim=64,
        hidden_size=64,
        hidden_size_min=16,
        n_hidden_layers=4,
        architecture=arch,
        use_layer_norm=True,
    )
    m = MLP(cfg)
    x = torch.randn(3, 32)
    y = m(x)
    assert y.shape == (3, 64)
    assert m.num_parameters() > 0


def test_linear_baseline_forward():
    cfg = LinearBaselineConfig(input_dim=16, output_dim=32)
    m = LinearBaseline(cfg)
    y = m(torch.randn(5, 16))
    assert y.shape == (5, 32)
