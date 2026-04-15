"""
Model Registry for ML Pipeline.

This module provides a centralized registry system for managing different
model architectures, with automatic discovery, factory patterns, and
consistent interfaces across model types.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

from src.core.config import Config, MLConfig
from src.models.base import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """
    Registry for managing ML model architectures and factories.
    
    Provides centralized registration, discovery, and instantiation of models
    with consistent interfaces and configuration management.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Type[BaseModel]] = {}
        self._configs: Dict[str, Type[ModelConfig]] = {}
        self._descriptions: Dict[str, str] = {}
        
        # Register built-in models
        self._register_builtin_models()
        
        logger.debug("Initialized ModelRegistry")
    
    def register_model(
        self,
        model_type: str,
        model_class: Type[BaseModel],
        config_class: Type[ModelConfig],
        description: str = ""
    ) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Unique identifier for the model type
            model_class: Model class (must inherit from BaseModel)
            config_class: Configuration class (must inherit from ModelConfig)
            description: Human-readable description
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel, got {model_class}")
        
        if not issubclass(config_class, ModelConfig):
            raise ValueError(f"Config class must inherit from ModelConfig, got {config_class}")
        
        if model_type in self._models:
            logger.warning(f"Overriding existing model type: {model_type}")
        
        self._models[model_type] = model_class
        self._configs[model_type] = config_class
        self._descriptions[model_type] = description
        
        logger.info(f"Registered model type: {model_type} ({model_class.__name__})")
    
    def unregister_model(self, model_type: str) -> None:
        """
        Unregister a model type.
        
        Args:
            model_type: Model type to remove
        """
        if model_type not in self._models:
            raise ValueError(f"Model type not registered: {model_type}")
        
        del self._models[model_type]
        del self._configs[model_type]
        del self._descriptions[model_type]
        
        logger.info(f"Unregistered model type: {model_type}")
    
    def list_models(self) -> List[str]:
        """
        Get list of registered model types.
        
        Returns:
            List of model type identifiers
        """
        return list(self._models.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a registered model type.
        
        Args:
            model_type: Model type identifier
            
        Returns:
            Dictionary with model information
        """
        if model_type not in self._models:
            raise ValueError(f"Model type not registered: {model_type}")
        
        return {
            "model_type": model_type,
            "model_class": self._models[model_type].__name__,
            "config_class": self._configs[model_type].__name__,
            "description": self._descriptions[model_type],
            "module": self._models[model_type].__module__,
        }
    
    def create_config(
        self,
        model_type: str,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> ModelConfig:
        """
        Create a configuration for a specific model type.
        
        Args:
            model_type: Model type identifier
            input_dim: Input feature dimension
            output_dim: Output dimension
            **kwargs: Additional configuration parameters
            
        Returns:
            Model configuration instance
        """
        if model_type not in self._configs:
            raise ValueError(f"Model type not registered: {model_type}")
        
        config_class = self._configs[model_type]
        
        return config_class(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs
        )
    
    def create_model(
        self,
        model_type: str,
        config: Optional[ModelConfig] = None,
        ml_config: Optional[MLConfig] = None,
        **config_kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Model type identifier
            config: Model configuration (created automatically if None)
            ml_config: General ML configuration
            **config_kwargs: Configuration parameters (used if config is None)
            
        Returns:
            Model instance
        """
        if model_type not in self._models:
            raise ValueError(f"Model type not registered: {model_type}")
        
        model_class = self._models[model_type]
        
        # Create config if not provided
        if config is None:
            if "input_dim" not in config_kwargs or "output_dim" not in config_kwargs:
                raise ValueError("input_dim and output_dim must be provided if config is None")
            
            config = self.create_config(model_type, **config_kwargs)
        
        # Create model instance
        return model_class(config, ml_config)
    
    def create_model_for_maxorder(
        self,
        model_type: str,
        maxorder: int,
        input_dim: Optional[int] = None,
        ml_config: Optional[MLConfig] = None,
        **config_kwargs
    ) -> BaseModel:
        """
        Create a model instance configured for a specific maxorder.
        
        Args:
            model_type: Model type identifier
            maxorder: Maximum multipole order
            input_dim: Input dimension (uses default PCA size if None)
            ml_config: General ML configuration
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Model instance configured for the maxorder
        """
        # Calculate output dimension from maxorder
        n_modes = maxorder * (maxorder + 2)
        output_dim = 4 * n_modes  # Real/imag for both E and M coefficients
        
        # Use default PCA dimension if not provided
        if input_dim is None:
            input_dim = 256  # Default PCA components
        
        return self.create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            ml_config=ml_config,
            **config_kwargs
        )
    
    def _register_builtin_models(self) -> None:
        """Register built-in model types."""
        # Import here to avoid circular imports
        from src.models.mlp import MLPModel, MLPConfig
        from src.models.baseline import BaselineModel, BaselineConfig
        
        # Register MLP model
        self.register_model(
            model_type="mlp",
            model_class=MLPModel,
            config_class=MLPConfig,
            description="Multi-layer perceptron with physics-aware training"
        )
        
        # Register baseline models
        self.register_model(
            model_type="baseline",
            model_class=BaselineModel,
            config_class=BaselineConfig,
            description="Linear baseline models (Ridge, OLS, Lasso)"
        )
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        model_types = ", ".join(self.list_models())
        return f"ModelRegistry({len(self._models)} types: {model_types})"


# =============================================================================
# Global Registry Instance
# =============================================================================

# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        Global ModelRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ModelRegistry()
    
    return _global_registry


def reset_model_registry() -> None:
    """Reset the global model registry (useful for testing)."""
    global _global_registry
    _global_registry = None


# =============================================================================
# Convenience Functions
# =============================================================================

def list_available_models() -> List[str]:
    """
    Get list of available model types.
    
    Returns:
        List of model type identifiers
    """
    registry = get_model_registry()
    return registry.list_models()


def create_model(
    model_type: str,
    maxorder: int = 15,
    input_dim: Optional[int] = None,
    **kwargs
) -> BaseModel:
    """
    Convenience function to create a model.
    
    Args:
        model_type: Model type ("mlp", "baseline")
        maxorder: Maximum multipole order
        input_dim: Input feature dimension (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured model instance
    """
    registry = get_model_registry()
    return registry.create_model_for_maxorder(
        model_type=model_type,
        maxorder=maxorder,
        input_dim=input_dim,
        **kwargs
    )


def create_mlp(maxorder: int = 15, **kwargs) -> BaseModel:
    """
    Create an MLP model with default settings.
    
    Args:
        maxorder: Maximum multipole order
        **kwargs: Additional MLP configuration parameters
        
    Returns:
        Configured MLP model
    """
    return create_model("mlp", maxorder=maxorder, **kwargs)


def create_baseline(
    maxorder: int = 15,
    baseline_type: str = "ridge",
    **kwargs
) -> BaseModel:
    """
    Create a baseline model with default settings.
    
    Args:
        maxorder: Maximum multipole order
        baseline_type: Type of baseline ("ridge", "linear", "lasso")
        **kwargs: Additional baseline configuration parameters
        
    Returns:
        Configured baseline model
    """
    return create_model(
        "baseline", 
        maxorder=maxorder, 
        baseline_type=baseline_type,
        **kwargs
    )


def print_model_info() -> None:
    """Print information about all registered models."""
    registry = get_model_registry()
    
    print("=== Available Models ===")
    for model_type in registry.list_models():
        info = registry.get_model_info(model_type)
        print(f"\n{model_type}:")
        print(f"  Class: {info['model_class']}")
        print(f"  Config: {info['config_class']}")
        print(f"  Description: {info['description']}")
        print(f"  Module: {info['module']}")
    
    print()