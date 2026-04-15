"""
Baseline Model Implementation for Multipole Analysis.

This module provides Ridge regression and other linear baseline models
for comparison with neural network approaches.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.core.config import MLConfig
from src.models.base import BaseModel, ModelConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class BaselineConfig(ModelConfig):
    """Configuration for baseline models."""
    
    # Model type
    model_type: str = "baseline"
    baseline_type: str = "ridge"  # "ridge", "linear", "lasso"
    
    # Ridge regression parameters
    ridge_alpha: float = 1.0
    
    # Lasso regression parameters  
    lasso_alpha: float = 1.0
    
    # Training parameters
    max_iter: int = 1000
    tolerance: float = 1e-4
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.baseline_type not in ("ridge", "linear", "lasso"):
            raise ValueError(f"Invalid baseline_type: {self.baseline_type}")
        
        if self.ridge_alpha <= 0:
            raise ValueError(f"ridge_alpha must be positive, got {self.ridge_alpha}")


# =============================================================================
# Baseline Model Implementation
# =============================================================================

class BaselineModel(BaseModel):
    """
    Baseline linear models for multipole coefficient prediction.
    
    Supports Ridge regression, OLS linear regression, and Lasso regression
    as baseline comparisons for neural network models.
    """
    
    def __init__(
        self,
        config: BaselineConfig,
        ml_config: Optional[MLConfig] = None
    ):
        """
        Initialize baseline model.
        
        Args:
            config: Baseline-specific configuration
            ml_config: General ML configuration
        """
        super().__init__(config, ml_config)
        
        # Type hint for config
        self.config: BaselineConfig = config
        
        # Initialize sklearn components
        self._model = None
        self._feature_scaler = None
        self._target_scaler = None
        
        # Training state
        self._training_stats = {}
        
        # Build model
        self._build_model()
    
    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        return "baseline"
    
    def _build_model(self) -> None:
        """Build the sklearn model."""
        try:
            from sklearn.linear_model import Ridge, LinearRegression, Lasso
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise RuntimeError(
                "scikit-learn is required for baseline models. "
                "Install with: pip install scikit-learn"
            )
        
        # Create model based on type
        if self.config.baseline_type == "ridge":
            self._model = Ridge(
                alpha=self.config.ridge_alpha,
                max_iter=self.config.max_iter,
                tol=self.config.tolerance,
                random_state=self.config.seed
            )
        elif self.config.baseline_type == "linear":
            self._model = LinearRegression()
        elif self.config.baseline_type == "lasso":
            self._model = Lasso(
                alpha=self.config.lasso_alpha,
                max_iter=self.config.max_iter,
                tol=self.config.tolerance,
                random_state=self.config.seed
            )
        else:
            raise ValueError(f"Unsupported baseline type: {self.config.baseline_type}")
        
        # Create scalers
        self._feature_scaler = StandardScaler()
        self._target_scaler = StandardScaler()
        
        self.logger.info(f"Built {self.config.baseline_type} baseline model")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the baseline model.
        
        Args:
            X_train: Training features (n_samples, input_dim)
            y_train: Training targets (n_samples, output_dim)
            X_val: Validation features (ignored)
            y_val: Validation targets (ignored)
            **kwargs: Additional parameters
            
        Returns:
            Training results
        """
        # Validate inputs
        self.validate_inputs(X_train, y_train)
        
        self.logger.info(f"Training {self.config.baseline_type} model on {X_train.shape[0]} samples")
        
        import time
        start_time = time.time()
        
        # Scale features and targets
        X_scaled = self._feature_scaler.fit_transform(X_train)
        y_scaled = self._target_scaler.fit_transform(y_train)
        
        # Fit model
        self._model.fit(X_scaled, y_scaled)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Compute training metrics
        train_predictions_scaled = self._model.predict(X_scaled)
        train_predictions = self._target_scaler.inverse_transform(train_predictions_scaled)
        train_mse = float(np.mean((y_train - train_predictions) ** 2))
        
        # Store training stats
        self._training_stats = {
            "training_time": training_time,
            "train_mse": train_mse,
            "n_features": X_train.shape[1],
            "n_targets": y_train.shape[1],
            "n_samples": X_train.shape[0]
        }
        
        # Add model-specific stats
        if hasattr(self._model, 'coef_'):
            self._training_stats["n_parameters"] = np.prod(self._model.coef_.shape)
            if hasattr(self._model, 'intercept_'):
                self._training_stats["n_parameters"] += np.prod(self._model.intercept_.shape)
        
        self.logger.info(f"Baseline training completed in {training_time:.2f}s (MSE: {train_mse:.6f})")
        
        return self._training_stats.copy()
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the baseline model.
        
        Args:
            X: Input features (n_samples, input_dim)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Validate inputs
        self.validate_inputs(X)
        
        # Scale features
        X_scaled = self._feature_scaler.transform(X)
        
        # Make predictions
        predictions_scaled = self._model.predict(X_scaled)
        
        # Inverse scale predictions
        predictions = self._target_scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the baseline model to disk.
        
        Args:
            path: Path to save directory
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn model
        if self._model is not None:
            with open(save_path / "model.pkl", "wb") as f:
                pickle.dump(self._model, f)
        
        # Save scalers
        if self._feature_scaler is not None:
            with open(save_path / "feature_scaler.pkl", "wb") as f:
                pickle.dump(self._feature_scaler, f)
        
        if self._target_scaler is not None:
            with open(save_path / "target_scaler.pkl", "wb") as f:
                pickle.dump(self._target_scaler, f)
        
        # Save configuration and metadata
        model_info = {
            "config": self.config.__dict__,
            "ml_config": self.ml_config.__dict__,
            "is_trained": self.is_trained,
            "training_stats": self._training_stats,
            "model_type": self.model_type,
        }
        
        with open(save_path / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Baseline model saved to {save_path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the baseline model from disk.
        
        Args:
            path: Path to saved model directory
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path not found: {load_path}")
        
        # Load model info
        info_path = load_path / "model_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Model info not found: {info_path}")
        
        with open(info_path) as f:
            model_info = json.load(f)
        
        # Update configuration
        self.config = BaselineConfig(**model_info["config"])
        self.is_trained = model_info["is_trained"]
        self._training_stats = model_info.get("training_stats", {})
        
        # Load sklearn model
        model_path = load_path / "model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
        
        # Load scalers
        scaler_path = load_path / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._feature_scaler = pickle.load(f)
        
        scaler_path = load_path / "target_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._target_scaler = pickle.load(f)
        
        self.logger.info(f"Baseline model loaded from {load_path}")
    
    def get_model_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients if available.
        
        Returns:
            Model coefficients or None if not applicable
        """
        if self._model is not None and hasattr(self._model, 'coef_'):
            return self._model.coef_
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance scores or None if not applicable
        """
        coeffs = self.get_model_coefficients()
        if coeffs is not None:
            # Use mean absolute coefficient value as importance
            return np.mean(np.abs(coeffs), axis=0)
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ridge_model(
    input_dim: int,
    output_dim: int,
    alpha: float = 1.0,
    **kwargs
) -> BaselineModel:
    """
    Create a Ridge regression baseline model.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        alpha: Ridge regularization parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Ridge baseline model
    """
    config = BaselineConfig(
        baseline_type="ridge",
        input_dim=input_dim,
        output_dim=output_dim,
        ridge_alpha=alpha,
        **kwargs
    )
    
    return BaselineModel(config)


def create_linear_model(
    input_dim: int,
    output_dim: int,
    **kwargs
) -> BaselineModel:
    """
    Create a linear regression baseline model.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured linear baseline model
    """
    config = BaselineConfig(
        baseline_type="linear",
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )
    
    return BaselineModel(config)