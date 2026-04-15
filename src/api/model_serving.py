"""
MLFlow Model Serving and Deployment System

Production-ready model serving with REST API, batch inference,
A/B testing, and automated deployment capabilities.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass
import asyncio
import threading
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.pyfunc import load_model as mlflow_load_model
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    mlflow_load_model = None
    MLFLOW_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = None
    Field = None
    uvicorn = None
    FASTAPI_AVAILABLE = False

import numpy as np

from ..core.mlflow_config import get_mlflow_manager
from ..models.mlflow_integration import get_model_registry
from ..api.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


# Request/Response models for API
if FASTAPI_AVAILABLE:
    class PredictionRequest(BaseModel):
        """Request model for predictions."""
        features: List[List[float]] = Field(..., description="Input features for prediction")
        model_name: Optional[str] = Field(None, description="Specific model name to use")
        model_version: Optional[str] = Field(None, description="Specific model version to use")
        model_stage: Optional[str] = Field("Production", description="Model stage to use")
        preprocessing: bool = Field(True, description="Whether to apply preprocessing")
        
    class PredictionResponse(BaseModel):
        """Response model for predictions."""
        predictions: List[List[float]] = Field(..., description="Model predictions")
        model_info: Dict[str, Any] = Field(..., description="Information about the model used")
        processing_time: float = Field(..., description="Processing time in seconds")
        timestamp: str = Field(..., description="Prediction timestamp")
    
    class ModelInfo(BaseModel):
        """Model information response."""
        name: str
        version: str
        stage: str
        description: Optional[str] = None
        metrics: Optional[Dict[str, float]] = None
        signature: Optional[Dict[str, Any]] = None
        
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        timestamp: str
        loaded_models: List[str]
        mlflow_connection: bool


@dataclass
class ModelServingConfig:
    """Configuration for model serving."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model loading
    default_model_name: str = "electromagnetic_multipole_model"
    default_model_stage: str = "Production"
    auto_reload_models: bool = True
    model_refresh_interval: int = 300  # seconds
    
    # Performance
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout: float = 0.1  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    log_predictions: bool = False  # Privacy consideration
    
    # A/B Testing
    enable_ab_testing: bool = False
    ab_test_models: List[Dict[str, str]] = None  # [{"name": "model_a", "version": "1", "traffic": 0.5}]


class ModelCache:
    """Manages loaded models in memory."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._last_refresh: float = 0
        self._lock = threading.RLock()
    
    def get_model_key(self, name: str, version: Optional[str] = None, stage: Optional[str] = None) -> str:
        """Generate cache key for model."""
        if version:
            return f"{name}:v{version}"
        elif stage:
            return f"{name}:{stage}"
        else:
            return f"{name}:latest"
    
    def load_model(self, name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Load model from MLFlow registry."""
        model_key = self.get_model_key(name, version, stage)
        
        with self._lock:
            # Check if model is already cached
            if model_key in self._models:
                return self._models[model_key], self._model_info[model_key]
            
            # Load model from registry
            try:
                registry = get_model_registry()
                
                # Load model
                model = registry.load_model(name, version, stage)
                if model is None:
                    raise ValueError(f"Model not found: {name}")
                
                # Get model metadata
                model_version_info = registry.get_model_version(name, version, stage)
                model_info = {
                    "name": name,
                    "version": model_version_info.version if model_version_info else "unknown",
                    "stage": model_version_info.stage if model_version_info else stage or "unknown",
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Cache model
                self._models[model_key] = model
                self._model_info[model_key] = model_info
                
                logger.info(f"Loaded model: {model_key}")
                return model, model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                raise
    
    def refresh_models(self, force: bool = False):
        """Refresh all cached models."""
        current_time = time.time()
        
        with self._lock:
            if not force and current_time - self._last_refresh < 300:  # 5 minutes
                return
            
            # Clear cache to force reload
            self._models.clear()
            self._model_info.clear()
            self._last_refresh = current_time
            
            logger.info("Model cache refreshed")
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models."""
        with self._lock:
            return list(self._models.keys())


class ModelServer:
    """High-performance model serving server."""
    
    def __init__(self, config: ModelServingConfig):
        self.config = config
        self.model_cache = ModelCache()
        self.app: Optional[FastAPI] = None
        
        # Batch processing
        self._batch_queue: List[Tuple[np.ndarray, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "requests_served": 0,
            "predictions_made": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLFlow not available for model serving")
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Electromagnetic Multipole Model Serving API",
            description="Production model serving for electromagnetic multipole analysis",
            version="1.0.0"
        )
        
        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            mlflow_manager = get_mlflow_manager()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                loaded_models=self.model_cache.list_loaded_models(),
                mlflow_connection=mlflow_manager.is_server_running()
            )
        
        # Model info endpoint
        @app.get("/models/{model_name}/info", response_model=ModelInfo)
        async def get_model_info(model_name: str, 
                                version: Optional[str] = None,
                                stage: Optional[str] = "Production"):
            """Get information about a model."""
            try:
                _, model_info = self.model_cache.load_model(model_name, version, stage)
                
                registry = get_model_registry()
                model_version_info = registry.get_model_version(model_name, version, stage)
                
                return ModelInfo(
                    name=model_name,
                    version=model_version_info.version if model_version_info else "unknown",
                    stage=model_version_info.stage if model_version_info else stage,
                    description=model_version_info.description if model_version_info else None
                )
                
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
        
        # Prediction endpoint
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Make predictions using the specified model."""
            start_time = time.time()
            
            try:
                # Validate input
                if not request.features:
                    raise HTTPException(status_code=400, detail="No features provided")
                
                # Convert to numpy array
                X = np.array(request.features, dtype=np.float32)
                
                # Determine model to use
                model_name = request.model_name or self.config.default_model_name
                model_stage = request.model_stage or self.config.default_model_stage
                
                # Load model
                model, model_info = self.model_cache.load_model(
                    model_name, request.model_version, model_stage
                )
                
                # Make predictions
                if self.config.enable_batching and len(X) > 1:
                    predictions = await self._batch_predict(model, X)
                else:
                    predictions = self._predict_single(model, X)
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats["requests_served"] += 1
                self.stats["predictions_made"] += len(predictions)
                
                # Log prediction if enabled
                if self.config.enable_logging:
                    background_tasks.add_task(
                        self._log_prediction,
                        model_info,
                        len(X),
                        processing_time
                    )
                
                return PredictionResponse(
                    predictions=predictions.tolist(),
                    model_info=model_info,
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Prediction failed: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Batch prediction endpoint
        @app.post("/predict/batch")
        async def predict_batch(request: PredictionRequest):
            """Batch prediction endpoint optimized for large inputs."""
            return await predict(request, BackgroundTasks())
        
        # Model management endpoints
        @app.post("/models/refresh")
        async def refresh_models():
            """Refresh all loaded models."""
            self.model_cache.refresh_models(force=True)
            return {"status": "success", "message": "Models refreshed"}
        
        # Statistics endpoint
        @app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            uptime = time.time() - self.stats["start_time"]
            return {
                **self.stats,
                "uptime_seconds": uptime,
                "requests_per_second": self.stats["requests_served"] / uptime if uptime > 0 else 0
            }
        
        self.app = app
        return app
    
    def _predict_single(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make prediction with a single model."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(X)
            else:
                # Fallback for MLFlow models
                return model.predict(X)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise
    
    async def _batch_predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using batching for better performance."""
        # For now, use simple batching - could be enhanced with queue processing
        batch_size = min(self.config.max_batch_size, len(X))
        
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            batch_pred = self._predict_single(model, batch)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)
    
    async def _log_prediction(self, model_info: Dict[str, Any], n_samples: int, processing_time: float):
        """Log prediction for monitoring (background task)."""
        if not self.config.enable_logging:
            return
        
        # Log to MLFlow if available
        try:
            mlflow_manager = get_mlflow_manager()
            if mlflow_manager.is_available:
                # Could log serving metrics here
                pass
        except Exception as e:
            logger.warning(f"Failed to log prediction metrics: {e}")
    
    def run(self):
        """Run the model serving server."""
        if not self.app:
            self.create_app()
        
        logger.info(f"Starting model server on {self.config.host}:{self.config.port}")
        
        # Pre-load default model
        try:
            self.model_cache.load_model(
                self.config.default_model_name,
                stage=self.config.default_model_stage
            )
            logger.info(f"Pre-loaded default model: {self.config.default_model_name}")
        except Exception as e:
            logger.warning(f"Failed to pre-load default model: {e}")
        
        # Start server
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level="info"
        )


class DeploymentManager:
    """Manages model deployment lifecycle."""
    
    def __init__(self):
        self.mlflow_manager = get_mlflow_manager()
        self.model_registry = get_model_registry()
    
    def deploy_model(self, 
                    model_name: str,
                    version: Optional[str] = None,
                    stage: Optional[str] = None,
                    deployment_target: str = "local",
                    config: Optional[ModelServingConfig] = None) -> Dict[str, Any]:
        """
        Deploy a model to the specified target.
        
        Args:
            model_name: Name of the model to deploy
            version: Specific version to deploy
            stage: Model stage to deploy  
            deployment_target: Target environment ("local", "docker", "kubernetes")
            config: Deployment configuration
            
        Returns:
            Deployment information
        """
        
        logger.info(f"Deploying model {model_name} to {deployment_target}")
        
        config = config or ModelServingConfig()
        
        if deployment_target == "local":
            return self._deploy_local(model_name, version, stage, config)
        elif deployment_target == "docker":
            return self._deploy_docker(model_name, version, stage, config)
        elif deployment_target == "kubernetes":
            return self._deploy_kubernetes(model_name, version, stage, config)
        else:
            raise ValueError(f"Unsupported deployment target: {deployment_target}")
    
    def _deploy_local(self, model_name: str, version: Optional[str], stage: Optional[str], config: ModelServingConfig) -> Dict[str, Any]:
        """Deploy model locally."""
        server = ModelServer(config)
        
        # Start server in a separate thread for non-blocking deployment
        def run_server():
            server.run()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return {
            "status": "deployed",
            "target": "local",
            "endpoint": f"http://{config.host}:{config.port}",
            "model": {"name": model_name, "version": version, "stage": stage},
            "deployment_time": datetime.now().isoformat()
        }
    
    def _deploy_docker(self, model_name: str, version: Optional[str], stage: Optional[str], config: ModelServingConfig) -> Dict[str, Any]:
        """Deploy model using Docker."""
        # This would create a Docker container with the model
        # For now, return placeholder
        return {
            "status": "not_implemented",
            "target": "docker",
            "message": "Docker deployment not yet implemented"
        }
    
    def _deploy_kubernetes(self, model_name: str, version: Optional[str], stage: Optional[str], config: ModelServingConfig) -> Dict[str, Any]:
        """Deploy model to Kubernetes."""
        # This would create Kubernetes manifests and deploy
        # For now, return placeholder
        return {
            "status": "not_implemented", 
            "target": "kubernetes",
            "message": "Kubernetes deployment not yet implemented"
        }
    
    def promote_and_deploy(self, 
                          model_name: str,
                          version: str,
                          target_stage: str = "Production",
                          deployment_target: str = "local",
                          config: Optional[ModelServingConfig] = None) -> Dict[str, Any]:
        """Promote model to stage and deploy automatically."""
        
        # Promote model
        promoted = self.model_registry.promote_model(model_name, version, target_stage)
        
        if promoted:
            # Deploy promoted model
            deployment_info = self.deploy_model(
                model_name=model_name,
                stage=target_stage,
                deployment_target=deployment_target,
                config=config
            )
            
            return {
                "promotion": {"status": "success", "stage": target_stage},
                "deployment": deployment_info
            }
        else:
            return {
                "promotion": {"status": "failed"},
                "deployment": {"status": "skipped"}
            }


# Convenience functions
def create_model_server(config: Optional[ModelServingConfig] = None) -> ModelServer:
    """Create a model server with default configuration."""
    config = config or ModelServingConfig()
    return ModelServer(config)

def serve_model(model_name: str, 
               version: Optional[str] = None,
               stage: str = "Production",
               host: str = "0.0.0.0",
               port: int = 8000):
    """Convenience function to serve a model."""
    config = ModelServingConfig(
        host=host,
        port=port,
        default_model_name=model_name,
        default_model_stage=stage
    )
    
    server = create_model_server(config)
    server.run()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLFlow Model Serving Server")
    parser.add_argument("--model-name", type=str, default="electromagnetic_multipole_model", help="Model name to serve")
    parser.add_argument("--model-stage", type=str, default="Production", help="Model stage to serve")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    serve_model(
        model_name=args.model_name,
        stage=args.model_stage,
        host=args.host,
        port=args.port
    )