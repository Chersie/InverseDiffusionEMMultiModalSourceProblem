"""
MLFlow Experiment Management Utilities

High-level utilities for managing MLFlow experiments, runs, and model lifecycle
in the electromagnetic multipole ML pipeline.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.experiment_tracker import ExperimentTracker, track_experiment
from src.core.mlflow_config import get_mlflow_manager, MLFlowConfig
from src.models.mlflow_integration import get_model_registry
from src.core.config import Config


class MLFlowExperimentManager:
    """High-level experiment management with MLFlow integration."""
    
    def __init__(self, 
                 experiment_name: str,
                 config: Optional[Union[Config, Dict[str, Any]]] = None,
                 mlflow_config: Optional[MLFlowConfig] = None):
        self.experiment_name = experiment_name
        self.config = config
        self.mlflow_config = mlflow_config or MLFlowConfig.from_env()
        
        # Initialize components
        self.mlflow_manager = get_mlflow_manager()
        self.model_registry = get_model_registry()
        self.tracker = ExperimentTracker(self.mlflow_config)
        
        # Experiment state
        self._experiment_started = False
        self._run_name = None
        
    def start_experiment(self, 
                        run_name: Optional[str] = None,
                        tags: Optional[Dict[str, str]] = None) -> bool:
        """Start the experiment with automatic configuration logging."""
        if not self.mlflow_manager.is_available:
            print("⚠️ MLFlow not available - running without experiment tracking")
            return False
        
        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.experiment_name}_{timestamp}"
        
        self._run_name = run_name
        
        # Prepare tags
        experiment_tags = {
            "experiment.type": "electromagnetic_multipole",
            "experiment.framework": "pytorch",
            "experiment.pipeline_version": "2.0"
        }
        if tags:
            # Handle both list and dictionary formats for tags
            if isinstance(tags, list):
                # Convert list of tags to dictionary format
                for i, tag in enumerate(tags):
                    experiment_tags[f"tag.{i}"] = str(tag)
            elif isinstance(tags, dict):
                experiment_tags.update(tags)
            else:
                # Handle other iterable formats
                try:
                    experiment_tags.update(tags)
                except (TypeError, ValueError):
                    experiment_tags["tags"] = str(tags)
        
        # Start tracking
        success = self.tracker.start_experiment(
            experiment_name=self.experiment_name,
            run_name=run_name,
            tags=experiment_tags
        )
        
        if success:
            self._experiment_started = True
            
            # Log configuration if provided
            if self.config:
                self.tracker.log_config(self.config)
            
            print(f"🚀 Started MLFlow experiment: {self.experiment_name}")
            print(f"   Run: {run_name}")
            
        return success
    
    def log_training_metrics(self, 
                           epoch: int,
                           metrics: Dict[str, float],
                           prefix: str = ""):
        """Log training metrics for an epoch."""
        if not self._experiment_started:
            return
            
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}.{k}": v for k, v in metrics.items()}
        
        self.tracker.log_training_step(
            epoch=epoch,
            step=epoch,  # Use epoch as step for simplicity
            metrics=metrics,
            log_system=True  # Log system metrics every epoch
        )
    
    def log_model_performance(self, 
                            performance_metrics: Dict[str, float],
                            model_info: Optional[Dict[str, Any]] = None):
        """Log final model performance metrics."""
        if not self._experiment_started:
            return
            
        # Log performance metrics
        final_metrics = {}
        for key, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                final_metrics[f"final.{key}"] = float(value)
        
        self.tracker.log_metrics(final_metrics)
        
        # Log model information as parameters
        if model_info:
            model_params = {}
            for key, value in model_info.items():
                if not isinstance(value, (dict, list)):
                    model_params[f"model.{key}"] = value
            
            self.tracker.log_params(model_params)
    
    def register_model(self, 
                      model: Any,
                      model_name: Optional[str] = None,
                      input_example: Optional[Any] = None,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      auto_promote: bool = True) -> Optional[str]:
        """Register trained model with automatic promotion logic."""
        if not self._experiment_started or not self.model_registry.is_available:
            print("⚠️ Model registration not available")
            return None
        
        # Generate model name if not provided
        if not model_name:
            model_name = f"{self.experiment_name}_model"
        
        # Prepare model description
        description = f"Model trained in experiment '{self.experiment_name}'"
        if self._run_name:
            description += f", run '{self._run_name}'"
        
        if performance_metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in performance_metrics.items()])
            description += f". Metrics: {metrics_str}"
        
        # Register model
        version = self.model_registry.register_model(
            model=model,
            model_name=model_name,
            input_example=input_example,
            description=description,
            tags={
                "experiment": self.experiment_name,
                "run": self._run_name or "unknown",
                "registration_time": datetime.now().isoformat()
            }
        )
        
        if version:
            print(f"✅ Registered model '{model_name}' version {version}")
            
            # Attempt automatic promotion if metrics provided
            if auto_promote and performance_metrics:
                promoted = self.model_registry.auto_promote_model(
                    model_name=model_name,
                    version=version,
                    metrics=performance_metrics
                )
                if promoted:
                    print(f"🚀 Auto-promoted model to Production stage")
        
        return version
    
    def log_artifacts_directory(self, 
                              artifacts_dir: Path,
                              artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts."""
        if not self._experiment_started or not artifacts_dir.exists():
            return
            
        self.tracker.log_artifacts(artifacts_dir, artifact_path)
        print(f"📁 Logged artifacts from {artifacts_dir}")
    
    def log_plots(self, plots_dir: Path):
        """Log all plots from a directory.""" 
        if not plots_dir.exists():
            return
            
        plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.jpg"))
        for plot_file in plot_files:
            self.tracker.log_artifact(plot_file, "plots")
        
        if plot_files:
            print(f"📊 Logged {len(plot_files)} plots")
    
    def finish_experiment(self, 
                         summary_metrics: Optional[Dict[str, Any]] = None):
        """Finish the experiment and log summary."""
        if not self._experiment_started:
            return
            
        # Log summary metrics if provided
        if summary_metrics:
            summary_params = {}
            summary_metrics_numeric = {}
            
            for key, value in summary_metrics.items():
                if isinstance(value, (int, float)):
                    summary_metrics_numeric[f"summary.{key}"] = float(value)
                else:
                    summary_params[f"summary.{key}"] = str(value)
            
            if summary_metrics_numeric:
                self.tracker.log_metrics(summary_metrics_numeric)
            if summary_params:
                self.tracker.log_params(summary_params)
        
        # Stop tracking
        self.tracker.stop_experiment()
        self._experiment_started = False
        
        print("✅ Experiment completed and logged to MLFlow")
    
    def get_experiment_url(self) -> Optional[str]:
        """Get URL to view this experiment in MLFlow UI."""
        if not self.mlflow_manager.is_available:
            return None
            
        server_config = self.mlflow_config.server
        base_url = f"http://{server_config.host}:{server_config.port}"
        
        try:
            experiment_id = self.mlflow_manager.get_experiment_id()
            if experiment_id:
                return f"{base_url}/#/experiments/{experiment_id}"
        except:
            pass
            
        return base_url


def create_experiment_manager(experiment_name: str, 
                            config: Optional[Config] = None,
                            **kwargs) -> MLFlowExperimentManager:
    """Create and configure an experiment manager."""
    return MLFlowExperimentManager(
        experiment_name=experiment_name,
        config=config,
        **kwargs
    )


def mlflow_training_session(experiment_name: str,
                           run_name: Optional[str] = None,
                           config: Optional[Config] = None,
                           tags: Optional[Dict[str, str]] = None):
    """
    Context manager for MLFlow training sessions.
    
    Usage:
        with mlflow_training_session("my_experiment") as session:
            # Training code here
            session.log_training_metrics(epoch, metrics)
            session.register_model(model)
    """
    class MLFlowSession:
        def __init__(self, manager: MLFlowExperimentManager):
            self.manager = manager
            
        def __enter__(self):
            self.manager.start_experiment(run_name, tags)
            return self.manager
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Log any exception that occurred
            if exc_type:
                self.manager.tracker.log_param("training.error", str(exc_val))
                print(f"❌ Training failed: {exc_val}")
            
            self.manager.finish_experiment()
    
    manager = create_experiment_manager(experiment_name, config)
    return MLFlowSession(manager)


# Training function decorator with automatic MLFlow integration
def mlflow_experiment(experiment_name: str,
                     auto_register_model: bool = True,
                     log_plots: bool = True,
                     log_config: bool = True):
    """
    Decorator to automatically add MLFlow tracking to training functions.
    
    Args:
        experiment_name: Name of the MLFlow experiment
        auto_register_model: Whether to automatically register the trained model
        log_plots: Whether to automatically log generated plots
        log_config: Whether to log the configuration
    """
    def decorator(training_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract config if available
            config = None
            for arg in args:
                if hasattr(arg, '__dataclass_fields__') or isinstance(arg, Config):
                    config = arg
                    break
            
            if not config:
                config = kwargs.get('config')
            
            # Create experiment manager
            manager = create_experiment_manager(experiment_name, config)
            
            # Generate run name
            run_name = f"{training_func.__name__}_{int(time.time())}"
            
            # Start experiment
            if not manager.start_experiment(run_name=run_name):
                # Run without tracking if MLFlow unavailable
                return training_func(*args, **kwargs)
            
            try:
                # Execute training function
                result = training_func(*args, **kwargs)
                
                # Auto-log results if they're in expected format
                if isinstance(result, dict):
                    # Extract and log metrics
                    metrics = {k: v for k, v in result.items() 
                             if isinstance(v, (int, float))}
                    if metrics:
                        manager.log_model_performance(metrics)
                    
                    # Auto-register model if present
                    if auto_register_model and 'model' in result:
                        input_example = result.get('input_example')
                        manager.register_model(
                            model=result['model'],
                            input_example=input_example,
                            performance_metrics=metrics
                        )
                    
                    # Auto-log plots if directory provided
                    if log_plots and 'plots_dir' in result:
                        manager.log_plots(Path(result['plots_dir']))
                
                return result
                
            except Exception as e:
                # Log the error
                manager.tracker.log_param("error.message", str(e))
                raise
                
            finally:
                # Always finish experiment
                manager.finish_experiment()
        
        return wrapper
    return decorator


# Global experiment manager for simple usage
_current_experiment: Optional[MLFlowExperimentManager] = None

def start_global_experiment(experiment_name: str, **kwargs) -> bool:
    """Start a global experiment for simple usage."""
    global _current_experiment
    _current_experiment = create_experiment_manager(experiment_name)
    return _current_experiment.start_experiment(**kwargs)

def log_global_metric(key: str, value: float, step: Optional[int] = None):
    """Log metric to global experiment."""
    if _current_experiment and _current_experiment._experiment_started:
        _current_experiment.tracker.log_metric(key, value, step)

def finish_global_experiment():
    """Finish the global experiment."""
    global _current_experiment
    if _current_experiment:
        _current_experiment.finish_experiment()
        _current_experiment = None