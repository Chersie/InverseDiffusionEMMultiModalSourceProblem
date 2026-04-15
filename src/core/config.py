"""
Unified Configuration System for ML Pipeline Project.

This module provides a hierarchical configuration system that consolidates all
scattered configuration throughout the project into a single, type-safe system.
Supports environment variable overrides and config file loading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


# =============================================================================
# Base Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class PathConfig:
    """Central path configuration for the project."""
    
    # Root directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    
    @property
    def naive_dir(self) -> Path:
        return self.project_root / "NaiveSolution"
    
    @property 
    def chersie_dir(self) -> Path:
        return self.project_root / "Chersie"
    
    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    # Data directories
    @property
    def test_features_dir(self) -> Path:
        # Check environment variable first
        import os
        env_path = os.getenv('DIPLOM_TEST_FEATURES_DIR')
        if env_path:
            return Path(env_path)
        
        # Check if data exists in project directory
        local_path = self.project_root / "data" / "external" / "test" / "E_in_plane"
        if local_path.exists():
            return local_path
        
        # Fall back to original path (with warning)
        default_path = self.project_root.parent / "diplom_dump" / "E+multip" / "E_in_plane"
        if not default_path.exists():
            print(f"WARNING: Test features directory not found at {default_path}")
            print("Consider setting DIPLOM_TEST_FEATURES_DIR environment variable")
        return default_path
        
    @property
    def test_targets_dir(self) -> Path:
        # Check environment variable first
        import os
        env_path = os.getenv('DIPLOM_TEST_TARGETS_DIR')
        if env_path:
            return Path(env_path)
        
        # Check if data exists in project directory
        local_path = self.project_root / "data" / "external" / "test" / "Multipoles_in_plane"
        if local_path.exists():
            return local_path
        
        # Fall back to original path (with warning)
        default_path = self.project_root.parent / "diplom_dump" / "E+multip" / "Multipoles_in_plane"
        if not default_path.exists():
            print(f"WARNING: Test targets directory not found at {default_path}")
            print("Consider setting DIPLOM_TEST_TARGETS_DIR environment variable")
        return default_path
        
    @property
    def data_raw_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def data_interim_dir(self) -> Path:
        return self.data_dir / "interim"
    
    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def data_external_dir(self) -> Path:
        return self.data_dir / "external"
    
    @property
    def data_ml_dir(self) -> Path:
        return self.data_dir / "ml"
    
    @property
    def data_ml_datasets_dir(self) -> Path:
        return self.data_ml_dir / "datasets"
    
    @property
    def data_ml_features_dir(self) -> Path:
        return self.data_ml_dir / "features"
    
    @property
    def data_ml_splits_dir(self) -> Path:
        return self.data_ml_dir / "splits"
    
    # Model directories
    @property
    def models_tracking_dir(self) -> Path:
        return self.models_dir / "tracking"
    
    @property
    def models_training_dir(self) -> Path:
        return self.models_dir / "training"
    
    @property
    def models_artifacts_dir(self) -> Path:
        return self.models_dir / "artifacts"
    
    # Library directories
    @property
    def library_fast_dir(self) -> Path:
        return self.chersie_dir / "FieldsFast0.5"
    
    @property
    def library_slow_dir(self) -> Path:
        return self.chersie_dir / "Fields0.5"
    
    def ensure_scaffold_dirs(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.docs_dir,
            self.data_dir,
            self.data_raw_dir,
            self.data_interim_dir,
            self.data_processed_dir,
            self.data_external_dir,
            self.data_ml_dir,
            self.data_ml_datasets_dir,
            self.data_ml_features_dir,
            self.data_ml_splits_dir,
            self.models_dir,
            self.models_tracking_dir,
            self.models_training_dir,
            self.models_artifacts_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def ensure_streaming_dirs(self, streaming_config: 'StreamingConfig') -> None:
        """Ensure streaming-related directories exist."""
        streaming_dirs = [
            streaming_config.cache_path,
            streaming_config.temp_path,
        ]
        
        for directory in streaming_dirs:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for core pipeline operations."""
    
    # Fundamental constants
    default_maxorder: int = 15
    angle_step_deg: int = 1
    library_header_lines: int = 43
    
    # File naming defaults
    field_file: str = "Fields.txt"
    result_file: str = "Results_Fields.txt"
    
    # Environment variable names
    field_file_stem_env: str = "FIELD_FILE"
    library_env: str = "MULTIPOLAR_LIBRARY"
    
    # Pipeline step validation
    valid_pipeline_test_steps: tuple[str, ...] = ("1", "2", "3", "4a", "4b")
    
    # Library settings
    library_type: str = "fast"  # "fast" or "slow"
    
    @property
    def default_library_dir(self) -> Path:
        """Get the default library directory based on library_type."""
        paths = PathConfig()
        return paths.library_fast_dir if self.library_type == "fast" else paths.library_slow_dir


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data processing and generation."""
    
    # Field generation settings
    z0_impedance: float = 377.0  # Free space impedance
    threshold: float = 0.01
    
    # Grid settings  
    n_theta: int = 179
    n_phi: int = 360
    
    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        return self.n_theta * self.n_phi
    
    # Latin square settings
    latin_square_seeds: tuple[int, ...] = (0, 1, 2, 3)
    
    # Dataset versioning
    dataset_version: str = "v1"


@dataclass(frozen=True)  
class MLConfig:
    """Configuration for machine learning components."""
    
    # Dataset settings
    n_samples: int = 10_000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    
    # Feature preprocessing
    pca_components: int = 256
    pca_oversample: int = 16
    pca_iterations: int = 0
    
    # Training settings
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-3
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Loss function weights
    amplitude_loss_weight: float = 1.0
    
    # Logging and validation
    val_log_frequency: int = 10
    detailed_metrics_frequency: int = 50
    
    @property
    def test_ratio(self) -> float:
        """Calculate test ratio from train and validation ratios."""
        return 1.0 - self.train_ratio - self.val_ratio


@dataclass(frozen=True)
class MLPConfig(MLConfig):
    """Configuration specific to MLP models."""
    
    # MLP architecture
    hidden_size: int = 512
    n_hidden_layers: int = 2
    dropout_rate: float = 0.1
    activation: str = "relu"  # "relu", "tanh", "gelu"


@dataclass(frozen=True)
class PhysicsAwareConfig(MLConfig):
    """Configuration specific to physics-aware models."""
    
    # Physics-aware architecture settings
    use_multipole_encoding: bool = True
    multipole_embedding_dim: int = 64
    attention_heads: int = 8
    transformer_layers: int = 4


@dataclass(frozen=True)
class BaselineConfig(MLConfig):
    """Configuration for baseline models (Ridge, Linear)."""
    
    # Baseline model settings
    trainer_type: str = "ridge"  # "ridge", "physics"
    ridge_alpha: float = 1.0
    
    # Physics trainer specific
    physics_epochs: int = 40


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for experiment tracking and management."""
    
    # MLflow settings
    experiment_name: str = "default"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    
    # Experiment metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    
    # Artifact settings
    log_models: bool = True
    log_metrics_frequency: int = 10
    log_artifacts: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory management and monitoring."""
    
    # Memory limits
    max_memory_gb: float = 8.0
    warning_threshold_gb: float = 6.0
    critical_threshold_gb: float = 7.5
    
    # Chunk sizes
    chunk_size_mb: int = 512
    max_chunk_size_mb: int = 1024
    min_chunk_size_mb: int = 64
    
    # Monitoring
    enable_monitoring: bool = True
    log_interval_seconds: float = 30.0
    gc_frequency: int = 10  # Force GC every N operations
    
    # Adaptive behavior
    adaptive_batching: bool = True
    safety_factor: float = 0.8  # Use 80% of available memory
    
    @property
    def warning_threshold_mb(self) -> float:
        return self.warning_threshold_gb * 1024
    
    @property 
    def critical_threshold_mb(self) -> float:
        return self.critical_threshold_gb * 1024
    
    @property
    def max_memory_mb(self) -> float:
        return self.max_memory_gb * 1024


@dataclass(frozen=True)
class StreamingConfig:
    """Configuration for streaming data processing."""
    
    # Streaming behavior
    enable_streaming: bool = True
    force_streaming_above_samples: int = 1000
    
    # Cache and storage
    cache_dir: str = "data/cache"
    temp_dir: str = "data/temp"
    use_compression: bool = True
    
    # Memory mapping
    mmap_mode: str = 'r'  # 'r', 'r+', 'w+', 'c'
    
    # Batch processing
    default_batch_size: int = 1000
    eval_batch_size: int = 2000
    preprocessing_batch_size: int = 5000
    
    # Data loader settings
    num_workers: int = 0  # 0 for single process, >0 for multiprocessing
    pin_memory: bool = False
    prefetch_factor: int = 2
    
    # File management
    cleanup_temp_files: bool = True
    keep_cache_files: bool = False
    
    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)
    
    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir)


# =============================================================================
# Unified Configuration System
# =============================================================================

@dataclass(frozen=True)
class Config:
    """
    Unified configuration system for the entire project.
    
    This is the main configuration class that combines all subsystem configs
    and provides environment variable override capabilities.
    """
    
    paths: PathConfig = field(default_factory=PathConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    
    @classmethod
    def from_env(cls) -> Config:
        """Create configuration with environment variable overrides."""
        # Get base config
        config = cls()
        
        # Apply environment overrides
        env_overrides = {}
        
        # Pipeline overrides
        if maxorder := os.getenv("MAXORDER"):
            env_overrides["pipeline"] = dataclass_replace(
                config.pipeline, 
                default_maxorder=int(maxorder)
            )
        
        # ML overrides
        ml_overrides = {}
        if n_samples := os.getenv("N_SAMPLES"):
            ml_overrides["n_samples"] = int(n_samples)
        if seed := os.getenv("SEED"):
            ml_overrides["seed"] = int(seed)
        if batch_size := os.getenv("BATCH_SIZE"):
            ml_overrides["batch_size"] = int(batch_size)
        if epochs := os.getenv("EPOCHS"):
            ml_overrides["epochs"] = int(epochs)
        if learning_rate := os.getenv("LEARNING_RATE"):
            ml_overrides["learning_rate"] = float(learning_rate)
        if device := os.getenv("DEVICE"):
            ml_overrides["device"] = device
            
        if ml_overrides:
            env_overrides["ml"] = dataclass_replace(config.ml, **ml_overrides)
        
        # Experiment overrides
        exp_overrides = {}
        if experiment_name := os.getenv("EXPERIMENT_NAME"):
            exp_overrides["experiment_name"] = experiment_name
        if run_name := os.getenv("RUN_NAME"):
            exp_overrides["run_name"] = run_name
        if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
            exp_overrides["tracking_uri"] = tracking_uri
            
        if exp_overrides:
            env_overrides["experiment"] = dataclass_replace(config.experiment, **exp_overrides)
        
        # Memory overrides
        memory_overrides = {}
        if max_memory_gb := os.getenv("MAX_MEMORY_GB"):
            memory_overrides["max_memory_gb"] = float(max_memory_gb)
        if chunk_size_mb := os.getenv("CHUNK_SIZE_MB"):
            memory_overrides["chunk_size_mb"] = int(chunk_size_mb)
        if enable_monitoring := os.getenv("ENABLE_MEMORY_MONITORING"):
            memory_overrides["enable_monitoring"] = as_bool_env("ENABLE_MEMORY_MONITORING", True)
        if adaptive_batching := os.getenv("ADAPTIVE_BATCHING"):
            memory_overrides["adaptive_batching"] = as_bool_env("ADAPTIVE_BATCHING", True)
            
        if memory_overrides:
            env_overrides["memory"] = dataclass_replace(config.memory, **memory_overrides)
        
        # Streaming overrides
        streaming_overrides = {}
        if enable_streaming := os.getenv("ENABLE_STREAMING"):
            streaming_overrides["enable_streaming"] = as_bool_env("ENABLE_STREAMING", True)
        if cache_dir := os.getenv("CACHE_DIR"):
            streaming_overrides["cache_dir"] = cache_dir
        if force_streaming := os.getenv("FORCE_STREAMING_ABOVE"):
            streaming_overrides["force_streaming_above_samples"] = int(force_streaming)
        if streaming_batch_size := os.getenv("STREAMING_BATCH_SIZE"):
            streaming_overrides["default_batch_size"] = int(streaming_batch_size)
        if num_workers := os.getenv("NUM_WORKERS"):
            streaming_overrides["num_workers"] = int(num_workers)
            
        if streaming_overrides:
            env_overrides["streaming"] = dataclass_replace(config.streaming, **streaming_overrides)
        
        # Return updated config
        if env_overrides:
            return dataclass_replace(config, **env_overrides)
        return config
    
    @classmethod
    def for_mlp(cls, **kwargs) -> tuple[Config, MLPConfig]:
        """Create config with MLP-specific settings."""
        base_config = cls.from_env()
        
        # Create MLP config inheriting from base ML config
        mlp_config = MLPConfig(
            # Inherit from base ML config
            n_samples=base_config.ml.n_samples,
            train_ratio=base_config.ml.train_ratio,
            val_ratio=base_config.ml.val_ratio,
            seed=base_config.ml.seed,
            pca_components=base_config.ml.pca_components,
            pca_oversample=base_config.ml.pca_oversample,
            pca_iterations=base_config.ml.pca_iterations,
            batch_size=base_config.ml.batch_size,
            epochs=base_config.ml.epochs,
            learning_rate=base_config.ml.learning_rate,
            device=base_config.ml.device,
            amplitude_loss_weight=base_config.ml.amplitude_loss_weight,
            val_log_frequency=base_config.ml.val_log_frequency,
            detailed_metrics_frequency=base_config.ml.detailed_metrics_frequency,
            # MLP-specific defaults (can be overridden by kwargs)
            **kwargs
        )
        
        return base_config, mlp_config
    
    @classmethod
    def for_physics_aware(cls, **kwargs) -> tuple[Config, PhysicsAwareConfig]:
        """Create config with physics-aware model settings."""
        base_config = cls.from_env()
        
        physics_config = PhysicsAwareConfig(
            # Inherit from base ML config
            n_samples=base_config.ml.n_samples,
            train_ratio=base_config.ml.train_ratio,
            val_ratio=base_config.ml.val_ratio,
            seed=base_config.ml.seed,
            pca_components=base_config.ml.pca_components,
            pca_oversample=base_config.ml.pca_oversample,
            pca_iterations=base_config.ml.pca_iterations,
            batch_size=base_config.ml.batch_size,
            epochs=base_config.ml.epochs,
            learning_rate=base_config.ml.learning_rate,
            device=base_config.ml.device,
            amplitude_loss_weight=base_config.ml.amplitude_loss_weight,
            val_log_frequency=base_config.ml.val_log_frequency,
            detailed_metrics_frequency=base_config.ml.detailed_metrics_frequency,
            # Physics-aware specific defaults
            **kwargs
        )
        
        return base_config, physics_config
    
    @classmethod 
    def for_baseline(cls, **kwargs) -> tuple[Config, BaselineConfig]:
        """Create config with baseline model settings."""
        base_config = cls.from_env()
        
        baseline_config = BaselineConfig(
            # Inherit from base ML config
            n_samples=base_config.ml.n_samples,
            train_ratio=base_config.ml.train_ratio,
            val_ratio=base_config.ml.val_ratio,
            seed=base_config.ml.seed,
            pca_components=base_config.ml.pca_components,
            pca_oversample=base_config.ml.pca_oversample,
            pca_iterations=base_config.ml.pca_iterations,
            batch_size=base_config.ml.batch_size,
            epochs=base_config.ml.epochs,
            learning_rate=base_config.ml.learning_rate,
            device=base_config.ml.device,
            amplitude_loss_weight=base_config.ml.amplitude_loss_weight,
            val_log_frequency=base_config.ml.val_log_frequency,
            detailed_metrics_frequency=base_config.ml.detailed_metrics_frequency,
            # Baseline specific defaults
            **kwargs
        )
        
        return base_config, baseline_config
    
    def ensure_directories(self) -> None:
        """Ensure all configuration directories exist."""
        self.paths.ensure_scaffold_dirs()
        self.paths.ensure_streaming_dirs(self.streaming)


# =============================================================================
# Utility Functions
# =============================================================================

def dataclass_replace(obj, **kwargs):
    """Replace fields in a dataclass, creating a new instance."""
    # Get the dataclass fields
    import dataclasses
    field_dict = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    
    # Update with kwargs
    field_dict.update(kwargs)
    
    # Create new instance
    return type(obj)(**field_dict)


def as_bool_env(name: str, default: bool = False) -> bool:
    """Parse environment variable as boolean."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_device(device: str = "auto") -> str:
    """Get the appropriate device for training."""
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    return device


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Global default configuration instance
DEFAULT_CONFIG = Config()

# Convenience function to get current config
def get_config() -> Config:
    """Get the current configuration with environment overrides."""
    return Config.from_env()