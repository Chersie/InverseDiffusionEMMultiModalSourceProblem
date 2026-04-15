"""
MLFlow Model Registry Integration

Automatic model registration, versioning, and lifecycle management
integrated with the electromagnetic multipole ML pipeline.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import tempfile
import shutil

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models import Model, ModelSignature, infer_signature
    from mlflow.types import DataType, Schema, ColSpec
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    Model = None
    ModelSignature = None
    infer_signature = None
    MLFLOW_AVAILABLE = False

import numpy as np
import torch

from ..core.mlflow_config import get_mlflow_manager, MLFlowConfig
from .base import BaseModel


class ModelVersionInfo:
    """Information about a model version in the registry."""
    
    def __init__(self, 
                 name: str, 
                 version: str, 
                 stage: str,
                 run_id: str,
                 creation_timestamp: int,
                 description: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.version = version
        self.stage = stage
        self.run_id = run_id
        self.creation_timestamp = creation_timestamp
        self.description = description or ""
        self.tags = tags or {}
        
    @property
    def creation_date(self) -> datetime:
        """Get creation date as datetime object."""
        return datetime.fromtimestamp(self.creation_timestamp / 1000)
    
    def __str__(self) -> str:
        return f"ModelVersion(name={self.name}, version={self.version}, stage={self.stage})"


class ModelRegistry:
    """Manages model registration, versioning, and lifecycle in MLFlow."""
    
    def __init__(self, config: Optional[MLFlowConfig] = None):
        self.config = config or MLFlowConfig.from_env()
        self.mlflow_manager = get_mlflow_manager()
        self._client: Optional[MlflowClient] = None
        
    @property
    def is_available(self) -> bool:
        """Check if model registry is available."""
        return MLFLOW_AVAILABLE and self.config.enabled
        
    @property
    def client(self) -> Optional[MlflowClient]:
        """Get MLFlow client for registry operations."""
        if not self.is_available:
            return None
            
        if self._client is None:
            self._client = self.mlflow_manager.client
            
        return self._client
    
    def create_model_signature(self, 
                              input_example: np.ndarray,
                              output_example: np.ndarray) -> Optional[ModelSignature]:
        """Create MLFlow model signature from input/output examples."""
        if not self.is_available:
            return None
            
        try:
            return infer_signature(input_example, output_example)
        except Exception as e:
            print(f"Warning: Could not infer model signature: {e}")
            return None
    
    def register_model(self,
                      model: BaseModel,
                      model_name: str,
                      input_example: Optional[np.ndarray] = None,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None,
                      await_registration_for: int = 30) -> Optional[str]:
        """
        Register a trained model in MLFlow registry.
        
        Args:
            model: The trained model to register
            model_name: Name for the model in registry
            input_example: Example input for signature inference
            description: Model description
            tags: Additional tags for the model version
            await_registration_for: Seconds to wait for registration
            
        Returns:
            Model version string if successful, None otherwise
        """
        if not self.is_available:
            print("Model registry not available - skipping registration")
            return None
            
        client = self.client
        if not client:
            return None
            
        try:
            # Ensure we have an active run
            if not mlflow.active_run():
                print("Warning: No active MLFlow run - creating temporary run for registration")
                mlflow.start_run()
                temporary_run = True
            else:
                temporary_run = False
            
            # Create model signature if input example provided
            signature = None
            if input_example is not None:
                try:
                    # Generate output example for signature
                    with torch.no_grad():
                        if hasattr(model, 'predict'):
                            output_example = model.predict(input_example[:1])
                        else:
                            # Fallback for models without predict method
                            output_example = np.random.randn(1, 10)  # Placeholder
                    signature = self.create_model_signature(input_example, output_example)
                except Exception as e:
                    print(f"Warning: Could not create signature: {e}")
            
            # Save model to MLFlow
            pytorch_model = None
            
            # Check if it's a direct PyTorch model
            if isinstance(model, torch.nn.Module):
                pytorch_model = model
            # Check if it's a wrapper with internal PyTorch model  
            elif hasattr(model, '_torch_model') and isinstance(getattr(model, '_torch_model'), torch.nn.Module):
                pytorch_model = model._torch_model
            # Check for other common PyTorch model attributes
            elif hasattr(model, 'model') and isinstance(getattr(model, 'model'), torch.nn.Module):
                pytorch_model = model.model
            
            if pytorch_model is not None:
                # Log as PyTorch model
                print(f"🔧 Logging PyTorch model: {type(pytorch_model)}")
                mlflow.pytorch.log_model(
                    pytorch_model=pytorch_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=model_name
                )
            else:
                # Log as generic model using pickle
                print(f"🔧 Logging generic model via pickle: {type(model)}")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model", 
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=model_name
                )
            
            # Get the registered model version
            run = mlflow.active_run()
            if run:
                # Wait for model to be registered
                for _ in range(await_registration_for):
                    try:
                        versions = client.search_model_versions(f"name='{model_name}'")
                        matching_versions = [v for v in versions if v.run_id == run.info.run_id]
                        if matching_versions:
                            version = matching_versions[0].version
                            break
                    except:
                        pass
                    time.sleep(1)
                else:
                    print("Warning: Timed out waiting for model registration")
                    version = "unknown"
                
                # Update version with additional metadata
                if version != "unknown":
                    self.update_model_version(
                        model_name=model_name,
                        version=version,
                        description=description,
                        tags=tags
                    )
                
                print(f"✅ Registered model '{model_name}' version {version}")
                
                # Close temporary run if created
                if temporary_run:
                    mlflow.end_run()
                    
                return version
                
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return None
    
    def update_model_version(self,
                           model_name: str,
                           version: str,
                           description: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None):
        """Update model version metadata."""
        if not self.is_available:
            return
            
        client = self.client
        if not client:
            return
            
        try:
            # Update description
            if description:
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
            
            # Set tags
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=str(value)
                    )
                    
        except Exception as e:
            print(f"Warning: Failed to update model version metadata: {e}")
    
    def get_model_version(self, 
                         model_name: str, 
                         version: Optional[str] = None,
                         stage: Optional[str] = None) -> Optional[ModelVersionInfo]:
        """
        Get model version information.
        
        Args:
            model_name: Name of the model
            version: Specific version (latest if None)
            stage: Model stage filter
            
        Returns:
            ModelVersionInfo if found, None otherwise
        """
        if not self.is_available:
            return None
            
        client = self.client
        if not client:
            return None
            
        try:
            if version:
                # Get specific version
                mv = client.get_model_version(model_name, version)
            else:
                # Get latest version, optionally filtered by stage
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    return None
                    
                if stage:
                    versions = [v for v in versions if v.current_stage.lower() == stage.lower()]
                    if not versions:
                        return None
                
                # Sort by version number (assuming semantic versioning)
                versions.sort(key=lambda x: int(x.version), reverse=True)
                mv = versions[0]
            
            return ModelVersionInfo(
                name=mv.name,
                version=mv.version,
                stage=mv.current_stage,
                run_id=mv.run_id,
                creation_timestamp=mv.creation_timestamp,
                description=mv.description,
                tags=dict(mv.tags) if mv.tags else {}
            )
            
        except Exception as e:
            print(f"Warning: Failed to get model version: {e}")
            return None
    
    def load_model(self, 
                  model_name: str,
                  version: Optional[str] = None,
                  stage: Optional[str] = None) -> Optional[Any]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Specific version (latest if None)  
            stage: Model stage ("Production", "Staging", etc.)
            
        Returns:
            Loaded model if successful, None otherwise
        """
        if not self.is_available:
            return None
            
        # Ensure client is initialized (which sets the tracking URI)
        client = self.client
        if not client:
            return None
            
        try:
            # Build model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Load model
            try:
                model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
                print(f"✅ Loaded model '{model_name}' from registry (pytorch)")
                return model
            except Exception as pytorch_e:
                try:
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✅ Loaded model '{model_name}' from registry (sklearn)")
                    return model
                except Exception as sklearn_e:
                    try:
                        import pickle
                        import os
                        import torch
                        import io
                        
                        # Set environment variable to disable MPS completely
                        original_pytorch_enable_mps = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', None)
                        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                        
                        # Also disable via environment
                        original_disable_mps = os.environ.get('PYTORCH_DISABLE_MPS', None) 
                        os.environ['PYTORCH_DISABLE_MPS'] = '1'
                        
                        # Disable MPS completely during loading to prevent crashes
                        original_mps_available = torch.backends.mps.is_available if hasattr(torch.backends, 'mps') else lambda: False
                        original_mps_built = torch.backends.mps.is_built if hasattr(torch.backends, 'mps') else lambda: False
                        
                        # Patch MPS availability to False
                        if hasattr(torch.backends, 'mps'):
                            torch.backends.mps.is_available = lambda: False
                            torch.backends.mps.is_built = lambda: False
                        
                        # Patch torch.load to use map_location="cpu"
                        original_load = torch.load
                        def patched_load(*args, **kwargs):
                            kwargs["map_location"] = "cpu"
                            return original_load(*args, **kwargs)
                        torch.load = patched_load
                        
                        # Try to download the artifact and load with pickle
                        local_path = mlflow.artifacts.download_artifacts(model_uri)
                        model_path = os.path.join(local_path, "model.pkl")
                        if os.path.exists(model_path):
                            with open(model_path, "rb") as f:
                                model = pickle.load(f)
                            print(f"✅ Loaded model '{model_name}' from registry (pickle)")
                            
                            # Restore original functions and environment
                            torch.load = original_load
                            if hasattr(torch.backends, 'mps'):
                                torch.backends.mps.is_available = original_mps_available
                                torch.backends.mps.is_built = original_mps_built
                            
                            # Restore environment variables
                            if original_pytorch_enable_mps is None:
                                os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
                            else:
                                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = original_pytorch_enable_mps
                                
                            if original_disable_mps is None:
                                os.environ.pop('PYTORCH_DISABLE_MPS', None)
                            else:
                                os.environ['PYTORCH_DISABLE_MPS'] = original_disable_mps
                            
                            return model
                        else:
                            # Restore original functions and environment
                            torch.load = original_load
                            if hasattr(torch.backends, 'mps'):
                                torch.backends.mps.is_available = original_mps_available
                                torch.backends.mps.is_built = original_mps_built
                            
                            # Restore environment variables
                            if original_pytorch_enable_mps is None:
                                os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
                            else:
                                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = original_pytorch_enable_mps
                                
                            if original_disable_mps is None:
                                os.environ.pop('PYTORCH_DISABLE_MPS', None)
                            else:
                                os.environ['PYTORCH_DISABLE_MPS'] = original_disable_mps
                            
                            raise Exception("model.pkl not found")
                    except Exception as pickle_e:
                        # Restore original functions and environment on error
                        try:
                            torch.load = original_load
                            if hasattr(torch.backends, 'mps'):
                                torch.backends.mps.is_available = original_mps_available
                                torch.backends.mps.is_built = original_mps_built
                            
                            # Restore environment variables
                            if original_pytorch_enable_mps is None:
                                os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
                            else:
                                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = original_pytorch_enable_mps
                                
                            if original_disable_mps is None:
                                os.environ.pop('PYTORCH_DISABLE_MPS', None)
                            else:
                                os.environ['PYTORCH_DISABLE_MPS'] = original_disable_mps
                        except:
                            pass  # Ignore restoration errors
                        
                        print(f"❌ Failed to load model '{model_name}':")
                        print(f"  PyTorch error: {pytorch_e}")
                        print(f"  Sklearn error: {sklearn_e}")
                        print(f"  Pickle error: {pickle_e}")
                        return None
        except Exception as outer_e:
            print(f"❌ Failed to load model '{model_name}': {outer_e}")
            return None
    
    def promote_model(self, 
                     model_name: str,
                     version: str,
                     stage: str,
                     archive_existing: bool = True) -> bool:
        """
        Promote model to a new stage.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage ("Staging", "Production", etc.)
            archive_existing: Whether to archive existing models in target stage
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            return False
            
        client = self.client
        if not client:
            return False
            
        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage.lower() in ['production', 'staging']:
                existing_versions = client.search_model_versions(
                    f"name='{model_name}' and current_stage='{stage}'"
                )
                for existing in existing_versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=existing.version,
                        stage="Archived"
                    )
                    print(f"📦 Archived existing {stage} model version {existing.version}")
            
            # Promote new version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            print(f"🚀 Promoted model '{model_name}' version {version} to {stage}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to promote model: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        if not self.is_available:
            return []
            
        client = self.client
        if not client:
            return []
            
        try:
            models = []
            for model in client.search_registered_models():
                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'tags': dict(model.tags) if model.tags else {},
                    'latest_versions': []
                }
                
                # Get version information
                for version in model.latest_versions:
                    version_info = {
                        'version': version.version,
                        'stage': version.current_stage,
                        'run_id': version.run_id,
                        'creation_timestamp': version.creation_timestamp
                    }
                    model_info['latest_versions'].append(version_info)
                
                models.append(model_info)
            
            return models
            
        except Exception as e:
            print(f"Warning: Failed to list models: {e}")
            return []
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version."""
        if not self.is_available:
            return False
            
        client = self.client
        if not client:
            return False
            
        try:
            client.delete_model_version(model_name, version)
            print(f"🗑️ Deleted model '{model_name}' version {version}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete model version: {e}")
            return False
    
    def auto_promote_model(self, 
                          model_name: str, 
                          version: str,
                          metrics: Dict[str, float]) -> bool:
        """
        Automatically promote model based on performance metrics.
        
        Args:
            model_name: Name of the model
            version: Version to evaluate
            metrics: Performance metrics
            
        Returns:
            True if promoted, False otherwise
        """
        if not self.config.registry.auto_promote_threshold:
            return False
            
        metric_name = self.config.registry.promotion_metric
        if metric_name not in metrics:
            return False
            
        metric_value = metrics[metric_name]
        threshold = self.config.registry.auto_promote_threshold
        
        if metric_value >= threshold:
            return self.promote_model(
                model_name=model_name,
                version=version,
                stage="Production",
                archive_existing=True
            )
        
        return False


# Integration with BaseModel
class MLFlowModelMixin:
    """Mixin to add MLFlow registry integration to models."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._registry = ModelRegistry()
        
    def register_model(self, 
                      model_name: Optional[str] = None,
                      input_example: Optional[np.ndarray] = None,
                      **kwargs) -> Optional[str]:
        """Register this model in MLFlow registry."""
        if not model_name:
            # Auto-generate model name
            model_name = f"{self.__class__.__name__}_{int(time.time())}"
            
        return self._registry.register_model(
            model=self,
            model_name=model_name,
            input_example=input_example,
            **kwargs
        )
    
    def save_with_registry(self, 
                          save_path: Path,
                          model_name: Optional[str] = None,
                          input_example: Optional[np.ndarray] = None,
                          register: bool = True) -> Optional[str]:
        """Save model and optionally register in MLFlow."""
        # Save normally first
        self.save(save_path)
        
        # Register if requested
        if register and self._registry.is_available:
            return self.register_model(model_name, input_example)
        
        return None


# Global registry instance
_model_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry

# Convenience functions
def register_model(model: BaseModel, 
                  model_name: str, 
                  input_example: Optional[np.ndarray] = None,
                  **kwargs) -> Optional[str]:
    """Register a model in the global registry."""
    return get_model_registry().register_model(model, model_name, input_example, **kwargs)

def load_model_from_registry(model_name: str, 
                           version: Optional[str] = None,
                           stage: Optional[str] = None) -> Optional[Any]:
    """Load a model from the global registry."""
    return get_model_registry().load_model(model_name, version, stage)

def promote_model(model_name: str, version: str, stage: str) -> bool:
    """Promote a model in the global registry."""
    return get_model_registry().promote_model(model_name, version, stage)