"""
Comprehensive Experiment Tracking System

Advanced experiment tracking decorators and utilities that automatically
log parameters, metrics, artifacts, and system information to MLFlow.
"""

from __future__ import annotations
import time
import functools
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import json
import pickle
import traceback
from dataclasses import asdict

try:
    import mlflow
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    mlflow = None
    psutil = None  
    GPUtil = None
    MONITORING_AVAILABLE = False

import numpy as np
import torch

from .mlflow_config import get_mlflow_manager, MLFlowConfig
from .config import Config


class SystemMetricsCollector:
    """Collects system metrics during training."""
    
    def __init__(self):
        self.enabled = MONITORING_AVAILABLE
        
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        if not self.enabled:
            return metrics
            
        try:
            # CPU metrics
            metrics['system.cpu_percent'] = psutil.cpu_percent()
            metrics['system.cpu_count'] = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['system.memory_percent'] = memory.percent
            metrics['system.memory_available_gb'] = memory.available / (1024**3)
            metrics['system.memory_used_gb'] = memory.used / (1024**3)
            
            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics[f'gpu_{i}.utilization'] = gpu.load * 100
                    metrics[f'gpu_{i}.memory_percent'] = gpu.memoryUtil * 100
                    metrics[f'gpu_{i}.temperature'] = gpu.temperature
            except:
                pass
                
            # PyTorch GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    metrics[f'torch_gpu_{i}.memory_allocated_gb'] = allocated
                    metrics[f'torch_gpu_{i}.memory_reserved_gb'] = reserved
                    
        except Exception as e:
            print(f"Warning: Failed to collect system metrics: {e}")
            
        return metrics


class ExperimentTracker:
    """Advanced experiment tracker with automatic logging capabilities."""
    
    def __init__(self, 
                 config: Optional[MLFlowConfig] = None,
                 auto_log_system_metrics: bool = True):
        self.config = config or MLFlowConfig.from_env()
        self.mlflow_manager = get_mlflow_manager()
        self.system_collector = SystemMetricsCollector() if auto_log_system_metrics else None
        
        # Tracking state
        self._active_run = None
        self._start_time = None
        self._logged_params = set()
        self._logged_artifacts = set()
        
    @property 
    def is_tracking(self) -> bool:
        """Check if currently tracking an experiment."""
        return self._active_run is not None
    
    def start_experiment(self, 
                        experiment_name: Optional[str] = None,
                        run_name: Optional[str] = None,
                        tags: Optional[Dict[str, str]] = None) -> bool:
        """Start experiment tracking."""
        if not self.mlflow_manager.is_available:
            print("MLFlow not available - experiment tracking disabled")
            return False
        
        try:
            # Setup experiment
            self.mlflow_manager.setup_experiment(experiment_name)
            
            # Start run
            run_tags = tags or {}
            run_tags.update({
                "tracker.version": "2.0",
                "tracker.auto_generated": "true"
            })
            
            self._active_run = self.mlflow_manager.start_run(
                run_name=run_name, 
                tags=run_tags
            )
            self._start_time = time.time()
            
            # Log system information
            self._log_system_info()
            
            print(f"✅ Started experiment tracking: {experiment_name or 'default'}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start experiment tracking: {e}")
            return False
    
    def stop_experiment(self):
        """Stop experiment tracking."""
        if self._active_run:
            try:
                # Log final metrics
                if self._start_time:
                    total_time = time.time() - self._start_time
                    self.log_metric("experiment.total_time_seconds", total_time)
                
                self._active_run.__exit__(None, None, None)
                self._active_run = None
                print("✅ Experiment tracking stopped")
                
            except Exception as e:
                print(f"Warning: Error stopping experiment: {e}")
    
    def _log_system_info(self):
        """Log system information at experiment start."""
        if not self.is_tracking:
            return
            
        try:
            import platform
            import getpass
            
            system_info = {
                "system.platform": platform.system(),
                "system.platform_version": platform.version(),
                "system.python_version": platform.python_version(),
                "system.user": getpass.getuser(),
                "system.hostname": platform.node(),
            }
            
            # Log as tags
            for key, value in system_info.items():
                mlflow.set_tag(key, str(value))
            
            # Log system metrics
            if self.system_collector:
                metrics = self.system_collector.collect_metrics()
                for key, value in metrics.items():
                    mlflow.log_metric(f"initial.{key}", value)
                    
        except Exception as e:
            print(f"Warning: Failed to log system info: {e}")
    
    def log_param(self, key: str, value: Any):
        """Log a parameter, avoiding duplicates."""
        if not self.is_tracking:
            return
            
        # Convert complex types to string
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        elif not isinstance(value, (str, int, float, bool)):
            value = str(value)
            
        param_key = f"{key}={value}"
        if param_key not in self._logged_params:
            try:
                mlflow.log_param(key, value)
                self._logged_params.add(param_key)
            except Exception as e:
                print(f"Warning: Failed to log param {key}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: Union[float, int], step: Optional[int] = None):
        """Log a metric."""
        if not self.is_tracking:
            return
            
        try:
            mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            print(f"Warning: Failed to log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log multiple metrics."""
        if not self.is_tracking:
            return
            
        try:
            # Filter out non-numeric values
            numeric_metrics = {}
            for key, value in metrics.items():
                try:
                    numeric_metrics[key] = float(value)
                except (ValueError, TypeError):
                    # Log as tag instead
                    mlflow.set_tag(key, str(value))
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                
        except Exception as e:
            print(f"Warning: Failed to log metrics: {e}")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if not self.is_tracking:
            return
            
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Warning: Artifact file not found: {local_path}")
            return
            
        artifact_key = str(local_path)
        if artifact_key not in self._logged_artifacts:
            try:
                mlflow.log_artifact(str(local_path), artifact_path)
                self._logged_artifacts.add(artifact_key)
            except Exception as e:
                print(f"Warning: Failed to log artifact {local_path}: {e}")
    
    def log_artifacts(self, local_dir: Union[str, Path], artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts."""
        if not self.is_tracking:
            return
            
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            print(f"Warning: Artifact directory not found: {local_dir}")
            return
            
        try:
            mlflow.log_artifacts(str(local_dir), artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifacts from {local_dir}: {e}")
    
    def log_model(self, model: Any, artifact_path: str = "model", **kwargs):
        """Log a model artifact."""
        if not self.is_tracking:
            return
            
        try:
            if hasattr(model, 'state_dict'):  # PyTorch model
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            else:
                # Generic model - pickle it
                temp_path = Path(f"/tmp/model_{int(time.time())}.pkl")
                with open(temp_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log_artifact(temp_path, artifact_path)
                temp_path.unlink()  # Clean up
                
        except Exception as e:
            print(f"Warning: Failed to log model: {e}")
    
    def log_config(self, config: Union[Config, Dict[str, Any]]):
        """Log configuration object."""
        if not self.is_tracking:
            return
            
        try:
            if hasattr(config, '__dict__'):
                # Dataclass or similar
                if hasattr(config, '__dataclass_fields__'):
                    config_dict = asdict(config)
                else:
                    config_dict = config.__dict__
            else:
                config_dict = dict(config)
            
            # Flatten nested configuration
            def flatten_dict(d, parent_key='', sep='.'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            flat_config = flatten_dict(config_dict)
            self.log_params(flat_config)
            
            # Also save full config as artifact
            temp_config_path = Path(f"/tmp/config_{int(time.time())}.json")
            with open(temp_config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            self.log_artifact(temp_config_path, "config")
            temp_config_path.unlink()  # Clean up
            
        except Exception as e:
            print(f"Warning: Failed to log config: {e}")
    
    def log_system_metrics(self, step: Optional[int] = None):
        """Log current system metrics."""
        if not self.is_tracking or not self.system_collector:
            return
            
        metrics = self.system_collector.collect_metrics()
        if metrics:
            self.log_metrics(metrics, step=step)
    
    def log_training_step(self, 
                         epoch: int, 
                         step: int, 
                         metrics: Dict[str, float],
                         log_system: bool = False):
        """Log a training step with metrics."""
        if not self.is_tracking:
            return
            
        # Add step info to metrics
        step_metrics = {
            "training.epoch": epoch,
            "training.step": step,
            **metrics
        }
        
        self.log_metrics(step_metrics, step=step)
        
        # Optionally log system metrics
        if log_system and step % 10 == 0:  # Every 10 steps to avoid spam
            self.log_system_metrics(step=step)


def track_experiment(experiment_name: Optional[str] = None,
                    run_name: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None,
                    log_params: bool = True,
                    log_return: bool = True,
                    log_system_metrics: bool = False):
    """
    Decorator for automatic experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run (auto-generated if None)
        tags: Additional tags to attach to the run
        log_params: Whether to log function parameters
        log_return: Whether to log return values as metrics
        log_system_metrics: Whether to log system metrics during execution
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker()
            
            # Generate run name if not provided
            actual_run_name = run_name or f"{func.__name__}_{int(time.time())}"
            
            # Start tracking
            if not tracker.start_experiment(experiment_name, actual_run_name, tags):
                # If tracking fails, run function without tracking
                return func(*args, **kwargs)
            
            try:
                # Log function parameters if enabled
                if log_params:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    params = {}
                    for name, value in bound_args.arguments.items():
                        if hasattr(value, '__dict__') and hasattr(value, '__dataclass_fields__'):
                            # Handle dataclass parameters
                            tracker.log_config(value)
                        else:
                            params[f"param.{name}"] = value
                    
                    if params:
                        tracker.log_params(params)
                
                # Log initial system metrics
                if log_system_metrics:
                    tracker.log_system_metrics()
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log execution time
                tracker.log_metric("execution_time_seconds", execution_time)
                
                # Log return values if enabled and they're numeric
                if log_return and result is not None:
                    if isinstance(result, dict):
                        # Try to log metrics from dict return
                        tracker.log_metrics({f"result.{k}": v for k, v in result.items() 
                                           if isinstance(v, (int, float))})
                    elif isinstance(result, (int, float)):
                        tracker.log_metric("result.value", result)
                
                # Log final system metrics
                if log_system_metrics:
                    tracker.log_system_metrics()
                
                return result
                
            except Exception as e:
                # Log the error
                tracker.log_param("error.message", str(e))
                tracker.log_param("error.traceback", traceback.format_exc())
                raise
                
            finally:
                # Always stop tracking
                tracker.stop_experiment()
        
        return wrapper
    return decorator


def log_training_progress(tracker: ExperimentTracker):
    """
    Decorator for automatic training progress logging.
    
    Use this on training functions that yield epoch results.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # If function returns a generator (for epoch-by-epoch training)
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                def logged_generator():
                    for epoch, epoch_data in enumerate(result):
                        if isinstance(epoch_data, dict):
                            tracker.log_metrics({
                                f"epoch_{epoch}.{k}": v 
                                for k, v in epoch_data.items() 
                                if isinstance(v, (int, float))
                            }, step=epoch)
                        
                        yield epoch_data
                        
                return logged_generator()
            else:
                return result
        
        return wrapper
    return decorator


# Global tracker instance
_global_tracker: Optional[ExperimentTracker] = None

def get_global_tracker() -> ExperimentTracker:
    """Get the global experiment tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker()
    return _global_tracker

def start_global_experiment(experiment_name: str, **kwargs) -> bool:
    """Start global experiment tracking."""
    return get_global_tracker().start_experiment(experiment_name, **kwargs)

def stop_global_experiment():
    """Stop global experiment tracking."""
    get_global_tracker().stop_experiment()

# Convenience functions for global tracker
def log_param(key: str, value: Any):
    """Log parameter to global tracker."""
    get_global_tracker().log_param(key, value)

def log_metric(key: str, value: Union[float, int], step: Optional[int] = None):
    """Log metric to global tracker."""
    get_global_tracker().log_metric(key, value, step)

def log_artifact(local_path: Union[str, Path], artifact_path: Optional[str] = None):
    """Log artifact to global tracker.""" 
    get_global_tracker().log_artifact(local_path, artifact_path)