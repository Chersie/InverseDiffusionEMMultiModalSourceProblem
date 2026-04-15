"""
MLFlow Configuration Management

Centralized configuration system for MLFlow tracking, model registry,
and deployment infrastructure integrated with the main pipeline configuration.
"""

from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import contextlib

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

from .config import Config


@dataclass(frozen=True)
class MLFlowServerConfig:
    """MLFlow server configuration."""
    host: str = "127.0.0.1"
    port: int = 5000
    backend_store_uri: str = "sqlite:///mlflow.db"
    default_artifact_root: str = "mlartifacts"
    serve_artifacts: bool = True
    workers: int = 1
    
    @property
    def tracking_uri(self) -> str:
        """Get the tracking URI for this server configuration."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def server_url(self) -> str:
        """Get the web UI URL."""
        return self.tracking_uri


@dataclass(frozen=True)
class MLFlowTrackingConfig:
    """Experiment tracking configuration."""
    # Experiment settings
    experiment_name: str = "electromagnetic_multipole_analysis"
    auto_log: bool = True
    log_system_metrics: bool = True
    
    # Artifact logging
    log_models: bool = True
    log_plots: bool = True
    log_datasets: bool = False  # Can be large
    log_code: bool = True
    
    # Metric logging
    log_every_n_steps: int = 10
    log_loss_curves: bool = True
    log_gradients: bool = False
    
    # Tags
    default_tags: Dict[str, str] = field(default_factory=lambda: {
        "project": "electromagnetic_multipole_ml",
        "framework": "pytorch",
        "domain": "physics"
    })
    
    # Run naming
    run_name_template: str = "{model_type}_{timestamp}"
    include_git_info: bool = True
    include_system_info: bool = True


@dataclass(frozen=True)
class MLFlowRegistryConfig:
    """Model registry configuration."""
    # Registry settings
    register_models_automatically: bool = True
    default_stage: str = "None"
    
    # Versioning
    version_strategy: str = "auto"  # "auto", "semantic", "timestamp"
    include_metrics_in_description: bool = True
    
    # Model naming
    model_name_template: str = "{experiment_name}_{model_type}"
    
    # Promotion rules
    auto_promote_threshold: Optional[float] = None  # Auto-promote if metric > threshold
    promotion_metric: str = "test_r2"
    
    # Model validation
    require_signature: bool = True
    require_input_example: bool = True


@dataclass(frozen=True)
class MLFlowDeploymentConfig:
    """Model deployment configuration."""
    # Serving settings
    enable_model_serving: bool = True
    serving_port: int = 5001
    serving_host: str = "127.0.0.1"
    
    # Environment
    conda_env_path: Optional[str] = None
    docker_image: Optional[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    log_predictions: bool = False  # Privacy consideration
    
    # Performance
    max_batch_size: int = 32
    timeout_seconds: int = 30


@dataclass(frozen=True)
class MLFlowConfig:
    """Complete MLFlow configuration."""
    server: MLFlowServerConfig = field(default_factory=MLFlowServerConfig)
    tracking: MLFlowTrackingConfig = field(default_factory=MLFlowTrackingConfig)
    registry: MLFlowRegistryConfig = field(default_factory=MLFlowRegistryConfig)
    deployment: MLFlowDeploymentConfig = field(default_factory=MLFlowDeploymentConfig)
    
    # Global settings
    enabled: bool = True
    offline_mode: bool = False
    
    @classmethod
    def from_env(cls) -> MLFlowConfig:
        """Create configuration from environment variables."""
        return cls(
            server=MLFlowServerConfig(
                host=os.getenv("MLFLOW_SERVER_HOST", "127.0.0.1"),
                port=int(os.getenv("MLFLOW_SERVER_PORT", "5000")),
                backend_store_uri=os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db"),
                default_artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", "mlartifacts"),
                serve_artifacts=os.getenv("MLFLOW_SERVE_ARTIFACTS", "true").lower() == "true",
                workers=int(os.getenv("MLFLOW_WORKERS", "1"))
            ),
            tracking=MLFlowTrackingConfig(
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "electromagnetic_multipole_analysis"),
                auto_log=os.getenv("MLFLOW_AUTO_LOG", "true").lower() == "true",
                log_system_metrics=os.getenv("MLFLOW_LOG_SYSTEM_METRICS", "true").lower() == "true"
            ),
            enabled=os.getenv("MLFLOW_ENABLED", "true").lower() == "true",
            offline_mode=os.getenv("MLFLOW_OFFLINE", "false").lower() == "true"
        )
    
    @property
    def is_available(self) -> bool:
        """Check if MLFlow is available and enabled."""
        return MLFLOW_AVAILABLE and self.enabled
    
    def get_tracking_uri(self) -> str:
        """Get the tracking URI, with fallbacks."""
        if self.offline_mode:
            return str(Path.cwd() / "mlruns")
        
        # Check environment variable first
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri
        
        # Use server configuration
        return self.server.tracking_uri


class MLFlowManager:
    """Manages MLFlow configuration and client lifecycle."""
    
    def __init__(self, config: Optional[MLFlowConfig] = None):
        self.config = config or MLFlowConfig.from_env()
        self._client: Optional[MlflowClient] = None
        self._experiment_id: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        """Check if MLFlow is available."""
        return self.config.is_available
    
    @property
    def client(self) -> Optional[MlflowClient]:
        """Get MLFlow client, creating if needed."""
        if not self.is_available:
            return None
        
        if self._client is None:
            tracking_uri = self.config.get_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            self._client = MlflowClient(tracking_uri)
        
        return self._client
    
    def setup_experiment(self, experiment_name: Optional[str] = None) -> Optional[str]:
        """Set up MLFlow experiment and return experiment ID."""
        if not self.is_available:
            return None
        
        experiment_name = experiment_name or self.config.tracking.experiment_name
        
        try:
            # Set tracking URI
            tracking_uri = self.config.get_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=str(Path(self.config.server.default_artifact_root) / experiment_name)
                )
            else:
                experiment_id = experiment.experiment_id
            
            # Set as active experiment
            mlflow.set_experiment(experiment_name)
            self._experiment_id = experiment_id
            
            return experiment_id
            
        except Exception as e:
            print(f"Warning: Failed to setup MLFlow experiment: {e}")
            return None
    
    def get_experiment_id(self) -> Optional[str]:
        """Get current experiment ID."""
        if self._experiment_id is None:
            self.setup_experiment()
        return self._experiment_id
    
    @contextlib.contextmanager
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  nested: bool = False):
        """Context manager for MLFlow runs."""
        if not self.is_available:
            # Provide a no-op context for graceful degradation
            class NoOpRun:
                def log_param(self, *args, **kwargs): pass
                def log_metric(self, *args, **kwargs): pass
                def log_artifact(self, *args, **kwargs): pass
                def set_tag(self, *args, **kwargs): pass
            
            yield NoOpRun()
            return
        
        # Ensure experiment is set up
        self.setup_experiment()
        
        # Prepare tags
        run_tags = self.config.tracking.default_tags.copy()
        if tags:
            run_tags.update(tags)
        
        # Add system information if enabled
        if self.config.tracking.include_system_info:
            import platform
            import getpass
            run_tags.update({
                "system.platform": platform.system(),
                "system.python_version": platform.python_version(),
                "system.user": getpass.getuser()
            })
        
        # Add git information if enabled
        if self.config.tracking.include_git_info:
            try:
                import subprocess
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], 
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                run_tags["git.commit"] = git_commit
            except:
                pass
        
        # Start run
        with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
            yield run
    
    def log_config(self, config: Config):
        """Log pipeline configuration to MLFlow."""
        if not self.is_available:
            return
        
        # Log basic parameters
        mlflow.log_params({
            "maxorder": config.data.maxorder if hasattr(config.data, 'maxorder') else config.pipeline.maxorder,
            "n_phi": config.data.n_phi,
            "n_theta": config.data.n_theta,
            "scale": config.data.scale
        })
        
        # Log paths as tags
        mlflow.set_tags({
            "paths.project_root": str(config.paths.project_root),
            "paths.data_dir": str(config.paths.data_dir),
            "paths.output_dir": str(config.paths.output_dir)
        })
    
    def is_server_running(self) -> bool:
        """Check if MLFlow server is accessible."""
        if not self.is_available:
            return False
        
        try:
            client = self.client
            if client:
                # Try to list experiments as a health check
                client.search_experiments()
                return True
        except Exception:
            pass
        
        return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MLFlow server."""
        info = {
            "available": self.is_available,
            "tracking_uri": self.config.get_tracking_uri() if self.is_available else None,
            "server_running": False,
            "experiment_count": 0,
            "run_count": 0
        }
        
        if self.is_available:
            try:
                client = self.client
                if client:
                    info["server_running"] = True
                    experiments = client.search_experiments()
                    info["experiment_count"] = len(experiments)
                    
                    # Count runs across all experiments
                    total_runs = 0
                    for exp in experiments:
                        runs = client.search_runs([exp.experiment_id])
                        total_runs += len(runs)
                    info["run_count"] = total_runs
                    
            except Exception as e:
                info["error"] = str(e)
        
        return info


# Global MLFlow manager instance
_mlflow_manager: Optional[MLFlowManager] = None

def get_mlflow_manager() -> MLFlowManager:
    """Get global MLFlow manager instance."""
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLFlowManager()
    return _mlflow_manager

def configure_mlflow(config: MLFlowConfig):
    """Configure global MLFlow manager."""
    global _mlflow_manager
    _mlflow_manager = MLFlowManager(config)

# Convenience functions
def is_mlflow_available() -> bool:
    """Check if MLFlow is available and configured."""
    return get_mlflow_manager().is_available

def mlflow_context(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Create MLFlow run context."""
    return get_mlflow_manager().start_run(run_name=run_name, tags=tags)

def log_config_to_mlflow(config: Config):
    """Log pipeline configuration to MLFlow."""
    get_mlflow_manager().log_config(config)