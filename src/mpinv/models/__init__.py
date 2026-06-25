"""Model layer: registry, base, MLP, multi-head MLP, linear baselines."""

from mpinv.models.base import BaseModelConfig
from mpinv.models.linear_baselines import LinearBaseline, LinearBaselineConfig
from mpinv.models.mlp import MLP, MLPConfig
from mpinv.models.multi_head_mlp import (
    MultiHeadMLP,
    MultiHeadMLPConfig,
    expected_output_dim,
    transplant_heads,
    validate_groups,
)
from mpinv.models.registry import MODELS, register_model

__all__ = [
    "MLP",
    "MODELS",
    "BaseModelConfig",
    "LinearBaseline",
    "LinearBaselineConfig",
    "MLPConfig",
    "MultiHeadMLP",
    "MultiHeadMLPConfig",
    "expected_output_dim",
    "register_model",
    "transplant_heads",
    "validate_groups",
]
