"""
Unit tests for model components.

Tests the model registry, base model classes, and specific implementations.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.base import BaseModel, ModelConfig, create_model_config, get_output_dim_for_maxorder
from src.models.baseline import BaselineModel, BaselineConfig, create_ridge_model
from src.models.registry import ModelRegistry, get_model_registry, reset_model_registry
from src.core.config import MLConfig


class TestModelConfig:
    """Tests for ModelConfig and related functions."""
    
    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_type == "base"
        assert config.input_dim == 256
        assert config.output_dim == 240
        assert config.device == "auto"
        assert config.validate_inputs is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        ModelConfig(input_dim=100, output_dim=50)
        
        # Invalid dimensions should raise
        with pytest.raises(ValueError, match="input_dim must be positive"):
            ModelConfig(input_dim=0)
        
        with pytest.raises(ValueError, match="output_dim must be positive"):
            ModelConfig(output_dim=-1)
    
    def test_create_model_config(self):
        """Test model config factory function."""
        config = create_model_config(
            model_type="test",
            input_dim=128,
            output_dim=64,
            device="cpu"
        )
        
        assert config.model_type == "test"
        assert config.input_dim == 128
        assert config.output_dim == 64
        assert config.device == "cpu"
    
    def test_get_output_dim_for_maxorder(self):
        """Test output dimension calculation."""
        assert get_output_dim_for_maxorder(1) == 12   # 4 * 3 = 12
        assert get_output_dim_for_maxorder(2) == 32   # 4 * 8 = 32
        assert get_output_dim_for_maxorder(15) == 1020 # 4 * 255 = 1020


class TestBaselineConfig:
    """Tests for BaselineConfig."""
    
    def test_default_config(self):
        """Test default baseline configuration."""
        config = BaselineConfig()
        
        assert config.model_type == "baseline"
        assert config.baseline_type == "ridge"
        assert config.ridge_alpha == 1.0
        assert config.max_iter == 1000
    
    def test_invalid_baseline_type(self):
        """Test error handling for invalid baseline type."""
        with pytest.raises(ValueError, match="Invalid baseline_type"):
            BaselineConfig(baseline_type="invalid")
    
    def test_invalid_alpha(self):
        """Test error handling for invalid alpha."""
        with pytest.raises(ValueError, match="ridge_alpha must be positive"):
            BaselineConfig(ridge_alpha=-1.0)


class MockModel(BaseModel):
    """Mock model for testing base functionality."""
    
    def __init__(self, config: ModelConfig, ml_config=None):
        super().__init__(config, ml_config)
        self._mock_trained = False
        self._mock_predictions = None
    
    @property
    def model_type(self) -> str:
        return "mock"
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.validate_inputs(X_train, y_train)
        if X_val is not None:
            self.validate_inputs(X_val, y_val)
        
        self._mock_trained = True
        self.is_trained = True
        
        return {
            "training_time": 0.1,
            "final_loss": 0.01
        }
    
    def predict(self, X, **kwargs):
        if not self.is_trained:
            raise RuntimeError("Model must be trained")
        
        self.validate_inputs(X)
        
        # Return mock predictions
        return np.random.randn(X.shape[0], self.config.output_dim).astype(np.float32)
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "mock_model.txt").write_text("mock model data")
    
    def load(self, path):
        model_file = Path(path) / "mock_model.txt"
        if model_file.exists():
            self.is_trained = True


class TestBaseModel:
    """Tests for BaseModel functionality."""
    
    def test_initialization(self):
        """Test base model initialization."""
        config = ModelConfig(input_dim=100, output_dim=50)
        ml_config = MLConfig(seed=123)
        
        model = MockModel(config, ml_config)
        
        assert model.config == config
        assert model.ml_config == ml_config
        assert not model.is_trained
        assert model.device in ["cpu", "cuda", "mps"]  # Should be valid device
    
    def test_training_workflow(self, ml_dataset_small):
        """Test training workflow."""
        config = ModelConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1]
        )
        model = MockModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        X_val = ml_dataset_small["X"][ml_dataset_small["val_idx"]]
        y_val = ml_dataset_small["y"][ml_dataset_small["val_idx"]]
        
        result = model.fit(X_train, y_train, X_val, y_val)
        
        assert model.is_trained
        assert "training_time" in result
        assert "final_loss" in result
    
    def test_prediction_workflow(self, ml_dataset_small):
        """Test prediction workflow."""
        config = ModelConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1]
        )
        model = MockModel(config)
        
        # Train first
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        model.fit(X_train, y_train)
        
        # Make predictions
        X_test = ml_dataset_small["X"][ml_dataset_small["test_idx"]]
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(ml_dataset_small["test_idx"]), config.output_dim)
        assert predictions.dtype == np.float32
    
    def test_batch_prediction(self, ml_dataset_small):
        """Test batch prediction functionality."""
        config = ModelConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1]
        )
        model = MockModel(config)
        
        # Train first
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        model.fit(X_train, y_train)
        
        # Test batch prediction with small batch size
        X_test = ml_dataset_small["X"][ml_dataset_small["test_idx"]]
        predictions = model.predict_batch(X_test, batch_size=2)
        
        assert predictions.shape == (len(ml_dataset_small["test_idx"]), config.output_dim)
    
    def test_input_validation(self):
        """Test input validation."""
        config = ModelConfig(input_dim=10, output_dim=5)
        model = MockModel(config)
        
        # Valid inputs should not raise
        X = np.random.randn(20, 10)
        y = np.random.randn(20, 5)
        model.validate_inputs(X, y)
        
        # Invalid input dimension should raise
        X_bad = np.random.randn(20, 15)  # Wrong feature dimension
        with pytest.raises(ValueError, match="wrong feature dimension"):
            model.validate_inputs(X_bad)
        
        # Invalid output dimension should raise
        y_bad = np.random.randn(20, 8)  # Wrong output dimension
        with pytest.raises(ValueError, match="wrong output dimension"):
            model.validate_inputs(X, y_bad)
        
        # Mismatched sample counts should raise
        y_mismatch = np.random.randn(15, 5)  # Different sample count
        with pytest.raises(ValueError, match="mismatched sample counts"):
            model.validate_inputs(X, y_mismatch)
    
    def test_save_load(self, temp_model_dir):
        """Test model save/load functionality."""
        config = ModelConfig(input_dim=10, output_dim=5)
        model = MockModel(config)
        
        # Save model
        save_path = temp_model_dir / "test_model"
        model.save(save_path)
        
        assert (save_path / "mock_model.txt").exists()
        
        # Load model
        new_model = MockModel(config)
        assert not new_model.is_trained
        
        new_model.load(save_path)
        assert new_model.is_trained
    
    def test_model_info(self):
        """Test model information reporting."""
        config = ModelConfig(input_dim=100, output_dim=50, model_name="test_model")
        model = MockModel(config)
        
        info = model.get_model_info()
        
        assert info["model_type"] == "mock"
        assert info["model_name"] == "test_model"
        assert info["input_dim"] == 100
        assert info["output_dim"] == 50
        assert info["is_trained"] is False
    
    def test_prediction_before_training(self):
        """Test error when predicting before training."""
        config = ModelConfig(input_dim=10, output_dim=5)
        model = MockModel(config)
        
        X = np.random.randn(5, 10)
        
        with pytest.raises(RuntimeError, match="Model must be trained"):
            model.predict(X)


@pytest.mark.requires_sklearn
class TestBaselineModel:
    """Tests for BaselineModel implementation."""
    
    def test_initialization(self):
        """Test baseline model initialization."""
        config = BaselineConfig(input_dim=10, output_dim=5, baseline_type="ridge")
        model = BaselineModel(config)
        
        assert model.model_type == "baseline"
        assert not model.is_trained
    
    def test_ridge_training(self, ml_dataset_small):
        """Test Ridge regression training."""
        config = BaselineConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1],
            baseline_type="ridge",
            ridge_alpha=0.1
        )
        model = BaselineModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        
        result = model.fit(X_train, y_train)
        
        assert model.is_trained
        assert "training_time" in result
        assert "train_mse" in result
        assert result["train_mse"] >= 0
    
    def test_linear_training(self, ml_dataset_small):
        """Test linear regression training."""
        config = BaselineConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1],
            baseline_type="linear"
        )
        model = BaselineModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        
        result = model.fit(X_train, y_train)
        
        assert model.is_trained
        assert "training_time" in result
    
    def test_prediction(self, ml_dataset_small):
        """Test baseline model prediction."""
        config = BaselineConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1],
            baseline_type="ridge"
        )
        model = BaselineModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        model.fit(X_train, y_train)
        
        # Make predictions
        X_test = ml_dataset_small["X"][ml_dataset_small["test_idx"]]
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(ml_dataset_small["test_idx"]), config.output_dim)
    
    def test_save_load(self, temp_model_dir, ml_dataset_small):
        """Test baseline model save/load."""
        config = BaselineConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1],
            baseline_type="ridge"
        )
        model = BaselineModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        model.fit(X_train, y_train)
        
        # Save model
        save_path = temp_model_dir / "baseline_model"
        model.save(save_path)
        
        # Check files exist
        assert (save_path / "model.pkl").exists()
        assert (save_path / "model_info.json").exists()
        
        # Load model
        new_model = BaselineModel(config)
        new_model.load(save_path)
        
        assert new_model.is_trained
        
        # Test that loaded model can make predictions
        X_test = ml_dataset_small["X"][ml_dataset_small["test_idx"]]
        predictions = new_model.predict(X_test)
        assert predictions.shape[0] == len(ml_dataset_small["test_idx"])
    
    def test_model_coefficients(self, ml_dataset_small):
        """Test accessing model coefficients."""
        config = BaselineConfig(
            input_dim=ml_dataset_small["X"].shape[1],
            output_dim=ml_dataset_small["y"].shape[1],
            baseline_type="ridge"
        )
        model = BaselineModel(config)
        
        # Train model
        X_train = ml_dataset_small["X"][ml_dataset_small["train_idx"]]
        y_train = ml_dataset_small["y"][ml_dataset_small["train_idx"]]
        model.fit(X_train, y_train)
        
        # Get coefficients
        coeffs = model.get_model_coefficients()
        assert coeffs is not None
        assert coeffs.shape[1] == config.input_dim
        assert coeffs.shape[0] == config.output_dim
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert importance.shape == (config.input_dim,)
    
    def test_create_ridge_model_factory(self):
        """Test ridge model factory function."""
        model = create_ridge_model(input_dim=50, output_dim=20, alpha=0.5)
        
        assert isinstance(model, BaselineModel)
        assert model.config.baseline_type == "ridge"
        assert model.config.ridge_alpha == 0.5
        assert model.config.input_dim == 50
        assert model.config.output_dim == 20


class TestModelRegistry:
    """Tests for ModelRegistry system."""
    
    def setup_method(self):
        """Reset registry before each test."""
        reset_model_registry()
    
    def test_registry_initialization(self):
        """Test registry initialization with built-in models."""
        registry = ModelRegistry()
        
        models = registry.list_models()
        assert "mlp" in models
        assert "baseline" in models
    
    def test_model_registration(self):
        """Test registering new model types."""
        registry = ModelRegistry()
        
        # Register mock model
        registry.register_model(
            model_type="mock",
            model_class=MockModel,
            config_class=ModelConfig,
            description="Mock model for testing"
        )
        
        assert "mock" in registry.list_models()
        
        info = registry.get_model_info("mock")
        assert info["model_type"] == "mock"
        assert info["description"] == "Mock model for testing"
    
    def test_model_creation(self):
        """Test model creation through registry."""
        registry = ModelRegistry()
        
        # Create baseline model
        config = registry.create_config("baseline", input_dim=10, output_dim=5)
        model = registry.create_model("baseline", config)
        
        assert isinstance(model, BaselineModel)
        assert model.config.input_dim == 10
        assert model.config.output_dim == 5
    
    def test_create_model_for_maxorder(self):
        """Test model creation for specific maxorder."""
        registry = ModelRegistry()
        
        model = registry.create_model_for_maxorder("baseline", maxorder=3)
        
        expected_n_modes = 3 * (3 + 2)  # 15
        expected_output_dim = 4 * expected_n_modes  # 60
        
        assert model.config.output_dim == expected_output_dim
        assert model.config.input_dim == 256  # Default PCA size
    
    def test_unregister_model(self):
        """Test unregistering model types."""
        registry = ModelRegistry()
        
        # Register then unregister mock model
        registry.register_model("mock", MockModel, ModelConfig)
        assert "mock" in registry.list_models()
        
        registry.unregister_model("mock")
        assert "mock" not in registry.list_models()
        
        # Should raise error for nonexistent model
        with pytest.raises(ValueError, match="not registered"):
            registry.unregister_model("nonexistent")
    
    def test_global_registry(self):
        """Test global registry functions."""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        
        # Should return same instance
        assert registry1 is registry2
        
        # Reset and get new instance
        reset_model_registry()
        registry3 = get_model_registry()
        
        assert registry1 is not registry3