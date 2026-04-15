"""
Unit tests for configuration system.

Tests the unified configuration classes and environment variable handling.
"""
import os
import tempfile
from pathlib import Path

import pytest

from src.core.config import (
    Config, PathConfig, PipelineConfig, DataConfig, MLConfig,
    get_config, as_bool_env, get_device
)


class TestPathConfig:
    """Tests for PathConfig."""
    
    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()
        
        # Check that paths are Path objects
        assert isinstance(config.project_root, Path)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.models_dir, Path)
        
        # Check relative path structure
        assert config.data_dir == config.project_root / "data"
        assert config.models_dir == config.project_root / "models"
        assert config.docs_dir == config.project_root / "docs"
    
    def test_data_subdirectories(self):
        """Test data subdirectory paths."""
        config = PathConfig()
        
        expected_subdirs = [
            "raw", "interim", "processed", "external", "ml"
        ]
        
        for subdir in expected_subdirs:
            path = getattr(config, f"data_{subdir}_dir")
            assert isinstance(path, Path)
            assert path == config.data_dir / subdir
    
    def test_ml_subdirectories(self):
        """Test ML data subdirectory paths."""
        config = PathConfig()
        
        ml_subdirs = ["datasets", "features", "splits"]
        for subdir in ml_subdirs:
            path = getattr(config, f"data_ml_{subdir}_dir")
            assert isinstance(path, Path)
            assert path == config.data_ml_dir / subdir
    
    def test_ensure_scaffold_dirs(self, tmp_path):
        """Test directory creation."""
        # Create config with temporary root
        config = PathConfig()
        original_root = config.project_root
        
        # Patch project root to temporary directory
        config.__dict__["project_root"] = tmp_path
        
        # Ensure directories are created
        config.ensure_scaffold_dirs()
        
        # Check that directories exist
        assert (tmp_path / "data").exists()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "data" / "ml" / "datasets").exists()


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.default_maxorder == 15
        assert config.angle_step_deg == 1
        assert config.library_header_lines == 43
        assert config.field_file == "Fields.txt"
        assert config.library_type == "fast"
    
    def test_default_library_dir(self):
        """Test default library directory selection."""
        config = PipelineConfig()
        
        # Test fast library
        config = PipelineConfig(library_type="fast")
        assert "FieldsFast" in str(config.default_library_dir)
        
        # Test slow library  
        config = PipelineConfig(library_type="slow")
        assert "Fields0.5" in str(config.default_library_dir)
    
    def test_valid_pipeline_steps(self):
        """Test pipeline step validation."""
        config = PipelineConfig()
        
        valid_steps = config.valid_pipeline_test_steps
        assert "1" in valid_steps
        assert "2" in valid_steps
        assert "3" in valid_steps
        assert "4a" in valid_steps
        assert "4b" in valid_steps


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_grid_properties(self):
        """Test grid configuration properties."""
        config = DataConfig()
        
        assert config.n_theta == 179
        assert config.n_phi == 360
        assert config.n_points == 179 * 360
    
    def test_latin_square_settings(self):
        """Test Latin square configuration."""
        config = DataConfig()
        
        assert len(config.latin_square_seeds) == 4
        assert config.latin_square_seeds == (0, 1, 2, 3)
    
    def test_field_generation_settings(self):
        """Test field generation parameters."""
        config = DataConfig()
        
        assert config.z0_impedance == 377.0  # Free space impedance
        assert config.threshold == 0.01


class TestMLConfig:
    """Tests for MLConfig."""
    
    def test_default_values(self):
        """Test default ML configuration."""
        config = MLConfig()
        
        assert config.n_samples == 10_000
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.seed == 42
        
        # Check derived property
        assert config.test_ratio == 0.1  # 1.0 - 0.8 - 0.1
    
    def test_pca_settings(self):
        """Test PCA configuration."""
        config = MLConfig()
        
        assert config.pca_components == 256
        assert config.pca_oversample == 16
        assert config.pca_iterations == 0
    
    def test_training_settings(self):
        """Test training configuration."""
        config = MLConfig()
        
        assert config.batch_size == 256
        assert config.epochs == 200
        assert config.learning_rate == 1e-3
        assert config.device == "auto"


class TestUnifiedConfig:
    """Tests for the unified Config class."""
    
    def test_default_creation(self):
        """Test default config creation."""
        config = Config()
        
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.ml, MLConfig)
    
    def test_config_factory_methods(self):
        """Test config factory methods."""
        # Test MLP config creation
        base_config, mlp_config = Config.for_mlp(hidden_size=1024)
        
        assert isinstance(base_config, Config)
        assert mlp_config.hidden_size == 1024
        assert mlp_config.n_samples == base_config.ml.n_samples  # Inherited
    
    def test_environment_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        # Set environment variables
        monkeypatch.setenv("MAXORDER", "10")
        monkeypatch.setenv("N_SAMPLES", "5000")
        monkeypatch.setenv("BATCH_SIZE", "128")
        monkeypatch.setenv("LEARNING_RATE", "0.01")
        
        # Create config with environment overrides
        config = Config.from_env()
        
        assert config.pipeline.default_maxorder == 10
        assert config.ml.n_samples == 5000
        assert config.ml.batch_size == 128
        assert config.ml.learning_rate == 0.01


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_as_bool_env(self, monkeypatch):
        """Test boolean environment variable parsing."""
        # Test default value
        assert as_bool_env("NONEXISTENT") is False
        assert as_bool_env("NONEXISTENT", default=True) is True
        
        # Test true values
        for true_val in ["1", "true", "True", "TRUE", "yes", "YES", "y", "Y", "on", "ON"]:
            monkeypatch.setenv("TEST_BOOL", true_val)
            assert as_bool_env("TEST_BOOL") is True
        
        # Test false values
        for false_val in ["0", "false", "False", "FALSE", "no", "NO", "n", "N", "off", "OFF"]:
            monkeypatch.setenv("TEST_BOOL", false_val)
            assert as_bool_env("TEST_BOOL") is False
        
        # Test with whitespace
        monkeypatch.setenv("TEST_BOOL", "  true  ")
        assert as_bool_env("TEST_BOOL") is True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("cpu")
        assert device == "cpu"
        
        device = get_device("auto")
        assert device in ["cpu", "cuda", "mps"]  # Should be one of these
    
    def test_get_config_function(self):
        """Test global config function."""
        config = get_config()
        assert isinstance(config, Config)


class TestConfigIntegration:
    """Integration tests for config system."""
    
    def test_maxorder_consistency(self):
        """Test maxorder consistency across components."""
        config = Config()
        
        # Pipeline and ML configs should use same maxorder by default
        assert config.pipeline.default_maxorder == 15
        
        # ML output dimension should match maxorder
        n_modes = config.pipeline.default_maxorder * (config.pipeline.default_maxorder + 2)
        expected_output_dim = 4 * n_modes  # Real/imag for E and M
        
        # This would be used when creating models
        assert n_modes == 255  # For maxorder=15
        assert expected_output_dim == 1020
    
    def test_path_consistency(self):
        """Test path configuration consistency."""
        config = Config()
        
        # All paths should be under project root
        assert config.paths.data_dir.is_relative_to(config.paths.project_root)
        assert config.paths.models_dir.is_relative_to(config.paths.project_root)
        
        # Library paths should be under chersie dir
        assert config.paths.library_fast_dir.is_relative_to(config.paths.chersie_dir)
        assert config.paths.library_slow_dir.is_relative_to(config.paths.chersie_dir)
    
    def test_grid_dimension_consistency(self):
        """Test grid dimension consistency."""
        config = Config()
        
        # Data config should match pipeline expectations
        assert config.data.n_phi == int(360 / config.pipeline.angle_step_deg)
        assert config.data.n_theta == int(180 / config.pipeline.angle_step_deg) - 1