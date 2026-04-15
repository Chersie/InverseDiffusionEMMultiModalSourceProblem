"""
Base Model Interface for ML Pipeline.

This module defines the abstract base class and common interfaces that all
models in the registry must implement, ensuring consistent APIs across
different model architectures.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.config import Config, MLConfig
from src.core.dependencies import TORCH, requires_torch, optional_torch

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

ModelOutput = Dict[str, np.ndarray]
ModelState = Dict[str, Any]
TrainingResult = Dict[str, Any]


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Base configuration for all models."""
    
    # Model identification
    model_type: str = "base"
    model_name: str = "base_model"
    
    # Input/output dimensions
    input_dim: int = 256  # PCA components
    output_dim: int = 240  # 4 * n_modes (for maxorder=15: 240 = 4*60)
    
    # Physics parameters (for physics-informed models)
    maxorder: int = 15  # Maximum multipole order
    grid_n_theta: int = 179  # Theta grid resolution
    grid_n_phi: int = 360  # Phi grid resolution
    
    # Training parameters
    device: str = "auto"
    seed: int = 42
    
    # Validation
    validate_inputs: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")


# =============================================================================
# Abstract Base Model
# =============================================================================

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the pipeline.
    
    Defines the common interface that all models must implement, including
    training, inference, serialization, and configuration management.
    """
    
    def __init__(
        self, 
        config: ModelConfig,
        ml_config: Optional[MLConfig] = None
    ):
        """
        Initialize the base model.
        
        Args:
            config: Model-specific configuration
            ml_config: General ML configuration (creates default if None)
        """
        self.config = config
        self.ml_config = ml_config or MLConfig()
        self.is_trained = False
        self._model_state: Optional[ModelState] = None
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.config.model_type} model")
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get model type identifier."""
        pass
    
    @property
    def device(self) -> str:
        """Get the device for computation."""
        if self.config.device == "auto":
            from src.core.dependencies import get_compute_device
            return get_compute_device()
        return self.config.device
    
    @abstractmethod
    def fit(
        self, 
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> TrainingResult:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training input features (n_samples, input_dim)
            y_train: Training targets (n_samples, output_dim) 
            X_val: Validation input features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (n_samples, input_dim)
            **kwargs: Additional inference parameters
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def predict_batch(
        self, 
        X: np.ndarray, 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions in batches for memory efficiency.
        
        Args:
            X: Input features (n_samples, input_dim)
            batch_size: Size of processing batches (auto-determined if None)
            **kwargs: Additional inference parameters
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        from src.data.memory_monitor import estimate_memory_per_sample, get_available_memory, suggest_batch_size
        
        n_samples = X.shape[0]
        
        # Auto-determine batch size if not provided
        if batch_size is None:
            available_memory = get_available_memory()
            sample_memory = estimate_memory_per_sample(
                feature_shape=X.shape[1:],
                target_shape=(self.config.output_dim,),
                overhead_factor=3.0  # Model inference overhead
            )
            
            batch_size = suggest_batch_size(
                base_batch_size=1000,
                available_memory_mb=available_memory,
                sample_memory_mb=sample_memory
            )
            
            logger.info(f"Auto-determined batch size: {batch_size} for {n_samples} samples")
        
        # If batch size >= samples, use direct prediction to avoid overhead
        if batch_size >= n_samples:
            logger.debug(f"Using direct prediction (batch_size={batch_size} >= n_samples={n_samples})")
            return self.predict(X, **kwargs)
        
        logger.info(f"Using batched prediction: {n_samples} samples in batches of {batch_size}")
        
        predictions = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            if batch_idx % 10 == 0:
                logger.debug(f"Processing batch {batch_idx + 1}/{n_batches}")
            
            batch_X = X[start_idx:end_idx]
            batch_pred = self.predict(batch_X, **kwargs)
            predictions.append(batch_pred)
            
            # Clear batch data
            del batch_X, batch_pred
            
            # Periodic garbage collection
            if batch_idx % 5 == 0:
                import gc
                gc.collect()
        
        result = np.concatenate(predictions, axis=0)
        del predictions
        
        return result
    
    def predict_safe(self, X: np.ndarray, force_batch: bool = False, **kwargs) -> np.ndarray:
        """
        Safe prediction that automatically chooses between direct and batched prediction.
        
        Args:
            X: Input features (n_samples, input_dim)
            force_batch: Force batched prediction regardless of size
            **kwargs: Additional inference parameters
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        n_samples = X.shape[0]
        
        # Memory-based threshold for auto-batching
        memory_threshold_samples = 5000
        if hasattr(self.config, 'auto_batch_threshold'):
            memory_threshold_samples = self.config.auto_batch_threshold
        
        # Use batching if forced or if dataset is large
        if force_batch or n_samples > memory_threshold_samples:
            logger.info(f"Using batched prediction for {n_samples} samples (threshold: {memory_threshold_samples})")
            return self.predict_batch(X, **kwargs)
        else:
            logger.debug(f"Using direct prediction for {n_samples} samples")
            return self.predict(X, **kwargs)
    
    def validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data shapes and types.
        
        Args:
            X: Input features
            y: Target values (optional)
        """
        if not self.config.validate_inputs:
            return
        
        # Check X
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if X.shape[1] != self.config.input_dim:
            raise ValueError(
                f"X has wrong feature dimension: expected {self.config.input_dim}, "
                f"got {X.shape[1]}"
            )
        
        # Check y if provided
        if y is not None:
            # Handle both 2D (coefficients) and 3D (P field) targets
            if y.ndim == 2:
                # Standard 2D targets (coefficients)
                if y.shape[1] != self.config.output_dim:
                    raise ValueError(
                        f"y has wrong output dimension: expected {self.config.output_dim}, "
                        f"got {y.shape[1]}"
                    )
            elif y.ndim == 3:
                # 3D targets (P field for physics loss) - skip output dimension check
                # The output_dim refers to model predictions (coefficients), not P field targets
                pass
            else:
                raise ValueError(f"y must be 2D or 3D array, got shape {y.shape}")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y have mismatched sample counts: {X.shape[0]} vs {y.shape[0]}"
                )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "model_name": self.config.model_name,
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "device": self.device,
            "is_trained": self.is_trained,
            "config": self.config.__dict__,
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return (
            f"{self.__class__.__name__}("
            f"type={self.model_type}, "
            f"input_dim={self.config.input_dim}, "
            f"output_dim={self.config.output_dim}, "
            f"device={self.device}, "
            f"{status})"
        )


# =============================================================================
# Model Capabilities Mixins
# =============================================================================

class TorchModelMixin:
    """Mixin for PyTorch-based models."""
    
    @requires_torch
    def to_device(self, device: Optional[str] = None):
        """Move PyTorch model to specified device."""
        if hasattr(self, '_torch_model') and self._torch_model is not None:
            target_device = device or self.device
            self._torch_model.to(target_device)
            return self
        raise NotImplementedError("Model does not have _torch_model attribute")
    
    @requires_torch 
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters in PyTorch model."""
        if hasattr(self, '_torch_model') and self._torch_model is not None:
            return sum(p.numel() for p in self._torch_model.parameters() if p.requires_grad)
        return 0


class PhysicsAwareMixin:
    """Mixin for physics-aware models."""
    
    def compute_physics_loss(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray,
        basis: Dict[str, np.ndarray]
    ) -> float:
        """Compute physics-aware loss including power conservation."""
        # This will be implemented by models that use physics-aware training
        raise NotImplementedError("Physics loss computation not implemented")
    
    def validate_physics_constraints(
        self, 
        predictions: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Validate physics constraints on model predictions."""
        # This will be implemented by physics-aware models
        return {}


# =============================================================================
# Utility Functions
# =============================================================================

def create_model_config(
    model_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> ModelConfig:
    """
    Create a model configuration with validation.
    
    Args:
        model_type: Type of model
        input_dim: Input feature dimension
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Model configuration instance
    """
    return ModelConfig(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )


def get_output_dim_for_maxorder(maxorder: int) -> int:
    """
    Calculate output dimension for a given maxorder.
    
    Args:
        maxorder: Maximum multipole order
        
    Returns:
        Output dimension (4 * n_modes)
    """
    n_modes = maxorder * (maxorder + 2)
    return 4 * n_modes  # Real/imag for both E and M coefficients