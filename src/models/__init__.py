"""
Modular Model Registry for ML Pipeline.

This module provides a unified registry system for ML models, supporting
different architectures with consistent interfaces for training and inference.
"""

from src.models.registry import ModelRegistry, get_model_registry
from src.models.mlp import MLPModel
from src.models.baseline import BaselineModel

# Export main components
__all__ = [
    "ModelRegistry",
    "get_model_registry", 
    "MLPModel",
    "BaselineModel",
]