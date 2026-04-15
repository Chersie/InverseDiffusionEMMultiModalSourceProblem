"""
MLP Model Implementation for Multipole Analysis.

This module provides a modular implementation of the MLP architecture
extracted from the reference mlp_pipeline.py, with enhanced configurability
and consistent interfaces for the model registry.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Union

import numpy as np

from src.core.config import MLConfig
from src.core.dependencies import TORCH, requires_torch
from src.models.base import BaseModel, ModelConfig, TorchModelMixin, PhysicsAwareMixin

# Conditional torch imports
with TORCH as torch_module:
    torch = torch_module
    if torch_module is not None:
        nn = torch_module.nn
    else:
        nn = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class MLPConfig(ModelConfig):
    """Configuration for MLP models."""
    
    # Model architecture
    model_type: str = "mlp"
    hidden_size: int = 512
    n_hidden_layers: int = 2
    dropout_rate: float = 0.1
    activation: str = "gelu"  # "gelu", "relu", "tanh", "leaky_relu", "silu", "elu"
    
    # Training parameters
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    
    # Loss function configuration
    loss_type: str = "coefficient"  # "coefficient" or "physics"
    amplitude_loss_weight: float = 1.0  # For coefficient loss only
    
    # Physics loss parameters (when loss_type="physics")
    physics_grid_type: str = "equiangular"  # "equiangular" or "legendre-gauss"
    physics_grid_resolution_factor: float = 1.0  # Scale factor for grid resolution
    physics_field_weight: float = 0.1  # Weight for E_theta/E_phi component losses
    
    # Regularization
    weight_decay: float = 0.0
    
    # Validation and logging
    val_log_frequency: int = 10
    detailed_metrics_frequency: int = 50
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate MLP-specific parameters
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {self.n_hidden_layers}")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0,1], got {self.dropout_rate}")
        if self.activation not in ("gelu", "relu", "tanh", "leaky_relu", "silu", "elu"):
            raise ValueError(f"Invalid activation: {self.activation}")
        
        # Validate physics loss parameters
        if self.loss_type not in ("coefficient", "physics"):
            raise ValueError(f"Invalid loss_type: {self.loss_type}")
        if self.physics_grid_type not in ("equiangular", "legendre-gauss"):
            raise ValueError(f"Invalid physics_grid_type: {self.physics_grid_type}")
        if self.physics_grid_resolution_factor <= 0:
            raise ValueError(f"physics_grid_resolution_factor must be positive, got {self.physics_grid_resolution_factor}")
        if self.physics_field_weight < 0:
            raise ValueError(f"physics_field_weight must be non-negative, got {self.physics_field_weight}")


# =============================================================================
# PyTorch MLP Architecture
# =============================================================================

class PhysicsMLP:
    """
    PyTorch MLP implementation for multipole coefficient prediction.
    
    This is a modular extraction of the _PhysicsMLP from mlp_pipeline.py
    with enhanced configurability and cleaner interfaces.
    """
    
    def __init__(self, config: MLPConfig):
        """Initialize MLP with configuration."""
        global torch, nn
        
        # Import torch when actually needed
        with TORCH as torch_module:
            if torch_module is None:
                raise RuntimeError(
                    "PyTorch is required for MLP models. "
                    "Install with: pip install -r requirements-ml.txt"
                )
            
            torch = torch_module
            nn = torch_module.nn
        
        self.config = config
        self._model = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the PyTorch model."""
        # Get activation function
        if self.config.activation == "gelu":
            activation_fn = nn.GELU
        elif self.config.activation == "relu":
            activation_fn = nn.ReLU
        elif self.config.activation == "tanh":
            activation_fn = nn.Tanh
        elif self.config.activation == "leaky_relu":
            activation_fn = lambda: nn.LeakyReLU(0.01)
        elif self.config.activation == "silu":
            activation_fn = nn.SiLU
        elif self.config.activation == "elu":
            activation_fn = nn.ELU
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")
        
        # Build layers
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(self.config.input_dim, self.config.hidden_size),
            activation_fn(),
            nn.Dropout(self.config.dropout_rate),
        ])
        
        # Hidden layers
        for _ in range(self.config.n_hidden_layers - 1):
            layers.extend([
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                activation_fn(),
                nn.Dropout(self.config.dropout_rate),
            ])
        
        # Output layer (no activation - linear output)
        layers.append(nn.Linear(self.config.hidden_size, self.config.output_dim))
        
        # Create sequential model
        self._model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        if self._model is None:
            raise RuntimeError("Model not built")
        return self._model(x)
    
    def __call__(self, x):
        """Make model callable."""
        return self.forward(x)
    
    @property
    def model(self):
        """Get the underlying PyTorch model."""
        return self._model
    
    def parameters(self):
        """Get model parameters."""
        if self._model is None:
            return []
        return self._model.parameters()
    
    def state_dict(self):
        """Get model state dictionary."""
        if self._model is None:
            return {}
        return self._model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dictionary."""
        if self._model is None:
            self._build_model()
        self._model.load_state_dict(state_dict)
    
    def train(self, mode=True):
        """Set training mode."""
        if self._model is not None:
            self._model.train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        if self._model is not None:
            self._model.eval()
    
    def to(self, device):
        """Move model to device."""
        if self._model is not None:
            self._model.to(device)
        return self


# =============================================================================
# Main MLP Model Class
# =============================================================================

class MLPModel(BaseModel, TorchModelMixin, PhysicsAwareMixin):
    """
    MLP model for multipole coefficient prediction.
    
    This is the main model class that integrates with the model registry
    and provides consistent interfaces for training and inference.
    """
    
    def __init__(
        self,
        config: MLPConfig,
        ml_config: Optional[MLConfig] = None
    ):
        """
        Initialize MLP model.
        
        Args:
            config: MLP-specific configuration
            ml_config: General ML configuration
        """
        super().__init__(config, ml_config)
        
        # Type hint for config
        self.config: MLPConfig = config
        
        # Initialize PyTorch components
        self._torch_model = None
        self._optimizer = None
        self._loss_fn = None
        
        # Training state
        self._training_history = []
        self._preprocessor_state = None
        
        # Build model
        self._build_model()
    
    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        return "mlp"
    
    def state_dict(self):
        """Get model state dictionary."""
        if self._torch_model is None:
            return {}
        return self._torch_model.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load model state dictionary."""
        if self._torch_model is None:
            self._build_model()
        self._torch_model.load_state_dict(state_dict)
    
    def _build_model(self) -> None:
        """Build the PyTorch model and components."""
        # Build MLP architecture
        self.physics_mlp = PhysicsMLP(self.config)
        self._torch_model = self.physics_mlp.model
        
        # Move to device
        device = self.device
        if device != "cpu":
            self.to_device(device)
        
        self.logger.info(
            f"Built MLP model: {self.config.input_dim} -> "
            f"{self.config.hidden_size}x{self.config.n_hidden_layers} -> "
            f"{self.config.output_dim} (device={device})"
        )
    
    @requires_torch
    def fit(
        self,
        X_train: Union[np.ndarray, str],
        y_train: Union[np.ndarray, str],
        X_val: Optional[Union[np.ndarray, str]] = None,
        y_val: Optional[Union[np.ndarray, str]] = None,
        use_streaming: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the MLP model with support for streaming data.
        
        Args:
            X_train: Training features (array or path to memory-mapped file)
            y_train: Training targets (array or path to memory-mapped file)
            X_val: Optional validation features (array or path to memory-mapped file)
            y_val: Optional validation targets (array or path to memory-mapped file)
            use_streaming: Whether to use streaming DataLoaders
            **kwargs: Additional training arguments
            
        Returns:
            Training history dictionary
        """
        
        # Determine if we're using streaming based on input types
        if isinstance(X_train, str) or use_streaming:
            return self._fit_streaming(X_train, y_train, X_val, y_val, **kwargs)
        else:
            return self._fit_memory(X_train, y_train, X_val, y_val, **kwargs)
    
    def _fit_memory(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train using in-memory arrays (original implementation).
        
        Args:
            X_train: Training features (n_samples, input_dim)
            y_train: Training targets (n_samples, output_dim)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Validate inputs
        self.validate_inputs(X_train, y_train)
        if X_val is not None:
            self.validate_inputs(X_val, y_val)
        
        self.logger.info(f"Starting MLP training (memory mode): {X_train.shape[0]} samples, {self.config.epochs} epochs")
        
        # Setup training components
        self._setup_training()
        
        # Convert to PyTorch tensors
        device = self.device
        X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        
        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
            y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
        else:
            X_val_t = y_val_t = None
        
        # Training loop
        self._training_history = []
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(X_train_t, y_train_t)
            
            # Validation phase
            val_loss = None
            if X_val_t is not None and y_val_t is not None:
                val_loss = self._validate_epoch(X_val_t, y_val_t)
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Log progress
            epoch_info = {"epoch": epoch + 1, "train_loss": train_loss}
            if val_loss is not None:
                epoch_info["val_loss"] = val_loss
            
            self._training_history.append(epoch_info)
            
            # Periodic logging
            if (epoch + 1) % self.config.val_log_frequency == 0:
                log_msg = f"Epoch {epoch+1:3d}/{self.config.epochs}: train_loss={train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", val_loss={val_loss:.4f}"
                self.logger.info(log_msg)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        self.logger.info(f"MLP training completed in {training_time:.2f}s")
        
        # Return training results
        result = {
            "training_time": training_time,
            "final_train_loss": train_loss,
            "training_history": self._training_history.copy(),
            "parameter_count": self.get_parameter_count(),
        }
        
        if val_loss is not None:
            result["final_val_loss"] = val_loss
            result["best_val_loss"] = best_val_loss
        
        return result
    
    def _fit_streaming(
        self,
        X_train: Union[str, np.ndarray],
        y_train: Union[str, np.ndarray],
        X_val: Optional[Union[str, np.ndarray]] = None,
        y_val: Optional[Union[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train using streaming DataLoaders with memory-mapped datasets.
        
        Args:
            X_train: Training features (path to .npy file or array)
            y_train: Training targets (path to .npy file or array)
            X_val: Optional validation features (path to .npy file or array)
            y_val: Optional validation targets (path to .npy file or array)
            **kwargs: Additional training arguments
            
        Returns:
            Training history dictionary
        """
        from src.data.streaming_dataset import MemmapDataset, StreamingDataLoader
        from src.data.memory_monitor import monitor_memory, get_memory_usage
        
        self.logger.info("Starting MLP training in streaming mode...")
        
        with monitor_memory("streaming_training", log_before_after=True):
            
            # Create datasets
            if isinstance(X_train, str) and isinstance(y_train, str):
                train_dataset = MemmapDataset(X_train, y_train)
            else:
                # Convert arrays to temporary memory-mapped files
                import tempfile
                from pathlib import Path
                temp_dir = Path(tempfile.mkdtemp())
                
                X_train_path = temp_dir / "X_train.npy"
                y_train_path = temp_dir / "y_train.npy"
                
                np.save(X_train_path, X_train.astype(np.float32))
                np.save(y_train_path, y_train.astype(np.float32))
                
                train_dataset = MemmapDataset(str(X_train_path), str(y_train_path))
            
            val_dataset = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, str) and isinstance(y_val, str):
                    val_dataset = MemmapDataset(X_val, y_val)
                else:
                    X_val_path = temp_dir / "X_val.npy"
                    y_val_path = temp_dir / "y_val.npy"
                    
                    np.save(X_val_path, X_val.astype(np.float32))
                    np.save(y_val_path, y_val.astype(np.float32))
                    
                    val_dataset = MemmapDataset(str(X_val_path), str(y_val_path))
            
            # Get dataset info
            n_samples = len(train_dataset)
            sample_features, sample_targets = train_dataset[0]
            self.input_dim = sample_features.shape[0]
            self.output_dim = sample_targets.shape[0] if len(sample_targets.shape) > 0 else 1
            
            self.logger.info(f"Streaming training: {n_samples} samples, input_dim={self.input_dim}, output_dim={self.output_dim}")
            
            # Setup training components
            self._setup_training()
            
            # Create DataLoaders
            train_loader = StreamingDataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # Keep 0 for memory-mapped data
                pin_memory=self.device != 'cpu'
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = StreamingDataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=self.device != 'cpu'
                )
            
            # Training loop
            self._training_history = []
            best_val_loss = float('inf')
            
            start_time = time.time()
            
            for epoch in range(self.config.epochs):
                # Training phase
                train_loss = self._train_epoch_streaming(train_loader)
                
                # Validation phase
                val_loss = None
                if val_loader is not None:
                    val_loss = self._validate_epoch_streaming(val_loader)
                    
                    # Track best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
                # Log progress
                epoch_info = {"epoch": epoch + 1, "train_loss": train_loss}
                if val_loss is not None:
                    epoch_info["val_loss"] = val_loss
                
                self._training_history.append(epoch_info)
                
                # Periodic logging
                if (epoch + 1) % self.config.val_log_frequency == 0:
                    current_memory = get_memory_usage()
                    log_msg = f"Epoch {epoch+1:3d}/{self.config.epochs}: train_loss={train_loss:.4f}"
                    if val_loss is not None:
                        log_msg += f", val_loss={val_loss:.4f}"
                    log_msg += f" (Memory: {current_memory:.1f}MB)"
                    self.logger.info(log_msg)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        self.logger.info(f"Streaming MLP training completed in {training_time:.2f}s")
        
        # Return training results
        result = {
            "training_time": training_time,
            "final_train_loss": train_loss,
            "training_history": self._training_history.copy(),
            "parameter_count": self.get_parameter_count(),
            "mode": "streaming"
        }
        
        if val_loss is not None:
            result["final_val_loss"] = val_loss
            result["best_val_loss"] = best_val_loss
        
        return result
    
    def _train_epoch_streaming(self, train_loader) -> float:
        """Train one epoch using streaming DataLoader."""
        self._torch_model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            predictions = self._torch_model(batch_features)
            
            # Compute loss
            if self.config.loss_type == "physics":
                loss = self._loss_fn(predictions, batch_targets)
            else:
                loss = self._loss_fn(predictions, batch_targets)
            
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _validate_epoch_streaming(self, val_loader) -> float:
        """Validate one epoch using streaming DataLoader."""
        self._torch_model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                predictions = self._torch_model(batch_features)
                
                # Compute loss
                if self.config.loss_type == "physics":
                    loss = self._loss_fn(predictions, batch_targets)
                else:
                    loss = self._loss_fn(predictions, batch_targets)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0

    @requires_torch 
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the MLP model (with automatic batching for large datasets).
        
        Args:
            X: Input features (n_samples, input_dim)
            **kwargs: Additional parameters
            
        Returns:
            Predictions (n_samples, output_dim)
        """
        n_samples = X.shape[0]
        
        # Use automatic batching for large datasets
        if n_samples > getattr(self.config, 'auto_batch_threshold', 5000):
            return self.predict_safe(X, **kwargs)
        else:
            return self._predict_direct(X, **kwargs)
    
    @requires_torch
    def _predict_direct(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Direct prediction implementation (no batching).
        
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
        
        # Convert to PyTorch tensor and ensure device consistency
        # Get the actual device of the model weights (not the config device)
        if self._torch_model is not None:
            # Get device from the first parameter of the model
            model_device = next(self._torch_model.parameters()).device
        else:
            model_device = torch.device('cpu')  # fallback
            
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(model_device)
        
        # Make predictions
        self._torch_model.eval()
        with torch.no_grad():
            predictions = self._torch_model(X_tensor)
            predictions_np = predictions.cpu().numpy()
        
        return predictions_np
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the MLP model to disk.
        
        Args:
            path: Path to save directory
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model state
        if self._torch_model is not None:
            torch.save(self._torch_model.state_dict(), save_path / "model.pt")
        
        # Save configuration and metadata
        model_info = {
            "config": self.config.__dict__,
            "ml_config": self.ml_config.__dict__,
            "is_trained": self.is_trained,
            "training_history": self._training_history,
            "model_type": self.model_type,
        }
        
        with open(save_path / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Save preprocessor state if available
        if self._preprocessor_state is not None:
            np.savez_compressed(
                save_path / "preprocessor.npz",
                **self._preprocessor_state
            )
        
        self.logger.info(f"MLP model saved to {save_path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the MLP model from disk.
        
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
        self.config = MLPConfig(**model_info["config"])
        self.is_trained = model_info["is_trained"]
        self._training_history = model_info.get("training_history", [])
        
        # Rebuild model with new config
        self._build_model()
        
        # Load PyTorch state
        model_path = load_path / "model.pt"
        if model_path.exists() and self._torch_model is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self._torch_model.load_state_dict(state_dict)
        
        # Load preprocessor state
        preprocessor_path = load_path / "preprocessor.npz"
        if preprocessor_path.exists():
            self._preprocessor_state = dict(np.load(preprocessor_path))
        
        self.logger.info(f"MLP model loaded from {load_path}")
    
    def _setup_training(self) -> None:
        """Setup training components (optimizer, loss function)."""
        # Setup optimizer
        if self._torch_model is not None:
            self._optimizer = torch.optim.Adam(
                self._torch_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Setup loss function based on configuration
        if self.config.loss_type == "physics":
            # Physics-informed loss
            from src.models.physics_layers import PhysicsPowerLoss
            
            # Calculate grid shape with resolution factor
            grid_n_theta = int(self.config.grid_n_theta * self.config.physics_grid_resolution_factor)
            grid_n_phi = int(self.config.grid_n_phi * self.config.physics_grid_resolution_factor)
            
            self._loss_fn = PhysicsPowerLoss(
                maxorder=self.config.maxorder,
                grid_shape=(grid_n_phi, grid_n_theta),
                grid_type=self.config.physics_grid_type,
                device=self.device,
                field_weight=self.config.physics_field_weight
            )
            
            self.logger.info(f"Using physics loss with grid ({grid_n_phi}, {grid_n_theta}), field_weight={self.config.physics_field_weight}")
        else:
            # Coefficient MSE loss (default)
            self._loss_fn = torch.nn.MSELoss()
            self.logger.info("Using coefficient MSE loss")
        
        self.logger.debug("Training components initialized")
    
    def _train_epoch(self, X_train: "torch.Tensor", y_train: "torch.Tensor") -> float:
        """Train for one epoch."""
        self._torch_model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Create batches
        n_samples = X_train.shape[0]
        batch_size = self.config.batch_size
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Get batch
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            predictions = self._torch_model(X_batch)
            
            # Compute loss based on loss type
            if self.config.loss_type == "physics":
                # y_batch contains P field targets for physics loss
                loss = self._loss_fn(predictions, y_batch)
            else:
                # y_batch contains coefficient targets for MSE loss
                loss = self._loss_fn(predictions, y_batch)
            
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _validate_epoch(self, X_val: "torch.Tensor", y_val: "torch.Tensor") -> float:
        """Validate for one epoch."""
        self._torch_model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            # Create batches
            n_samples = X_val.shape[0]
            batch_size = self.config.batch_size
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                
                # Get batch
                X_batch = X_val[start_idx:end_idx]
                y_batch = y_val[start_idx:end_idx]
                
                # Forward pass
                predictions = self._torch_model(X_batch)
                
                # Compute loss based on loss type
                if self.config.loss_type == "physics":
                    # y_batch contains P field targets for physics loss
                    loss = self._loss_fn(predictions, y_batch)
                else:
                    # y_batch contains coefficient targets for MSE loss
                    loss = self._loss_fn(predictions, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_mlp_model(
    input_dim: int,
    output_dim: int,
    hidden_size: int = 512,
    n_hidden_layers: int = 2,
    **kwargs
) -> MLPModel:
    """
    Create an MLP model with specified architecture.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        hidden_size: Hidden layer size
        n_hidden_layers: Number of hidden layers
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MLP model
    """
    config = MLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
        n_hidden_layers=n_hidden_layers,
        **kwargs
    )
    
    return MLPModel(config)