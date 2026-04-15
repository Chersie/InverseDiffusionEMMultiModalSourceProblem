"""
Inference API for Model Serving and Batch Processing.

This module provides production-ready inference capabilities including
model serving, batch processing, and result formatting for different use cases.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.core.config import Config
from src.core.data_generator import unpack_coefficients
from src.models.base import BaseModel
from src.models.registry import get_model_registry

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    
    # Batch processing
    batch_size: int = 256
    max_batch_size: int = 1024
    
    # Performance settings
    enable_preprocessing_cache: bool = True
    enable_result_validation: bool = True
    
    # Output formatting
    output_format: str = "dict"  # "dict", "array", "packed"
    include_metadata: bool = True
    precision: int = 6
    
    # Memory management
    memory_limit_mb: float = 2048.0
    gc_frequency: int = 100  # Run garbage collection every N batches
    
    # Validation thresholds
    coefficient_magnitude_threshold: float = 1e6
    power_conservation_tolerance: float = 0.1


@dataclass
class InferenceResult:
    """Result from model inference."""
    
    # Predictions
    coefficients_e: np.ndarray
    coefficients_m: np.ndarray
    
    # Metadata
    n_samples: int
    inference_time: float
    model_type: str
    
    # Optional additional data
    raw_predictions: Optional[np.ndarray] = None
    preprocessing_time: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            "coefficients": {
                "electric": self.coefficients_e.tolist(),
                "magnetic": self.coefficients_m.tolist()
            },
            "metadata": {
                "n_samples": self.n_samples,
                "inference_time": self.inference_time,
                "model_type": self.model_type,
            }
        }
        
        if self.preprocessing_time is not None:
            result["metadata"]["preprocessing_time"] = self.preprocessing_time
        
        if self.validation_results is not None:
            result["validation"] = self.validation_results
        
        return result


# =============================================================================
# Inference Engine
# =============================================================================

class InferenceEngine:
    """
    Core inference engine for model predictions.
    
    Provides high-level interface for making predictions with trained models,
    including preprocessing, batch processing, and result formatting.
    """
    
    def __init__(
        self,
        model: BaseModel,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model for inference
            preprocessing_pipeline: Preprocessing pipeline (creates default if None)
            config: Inference configuration (creates default if None)
        """
        self.model = model
        self.config = config or InferenceConfig()
        
        # Initialize preprocessing
        if preprocessing_pipeline is not None:
            self.preprocessing = preprocessing_pipeline
        else:
            preproc_config = PreprocessingConfig(
                pca_components=model.config.input_dim,
                validate_inputs=self.config.enable_result_validation
            )
            self.preprocessing = PreprocessingPipeline(preproc_config)
        
        # State
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        # Validate model
        if not model.is_trained:
            raise ValueError("Model must be trained before inference")
        
        logger.info(
            f"Initialized InferenceEngine with {model.model_type} model "
            f"(batch_size={self.config.batch_size})"
        )
    
    def predict_from_fields(
        self,
        E_theta: np.ndarray,
        E_phi: np.ndarray,
        **kwargs
    ) -> InferenceResult:
        """
        Make predictions from electromagnetic field data.
        
        Args:
            E_theta: E_theta component (n_samples, n_phi, n_theta) or (n_phi, n_theta)
            E_phi: E_phi component (n_samples, n_phi, n_theta) or (n_phi, n_theta)
            **kwargs: Additional inference parameters
            
        Returns:
            Inference result with coefficients and metadata
        """
        start_time = time.time()
        
        # Handle single sample (add batch dimension)
        if E_theta.ndim == 2:
            E_theta = E_theta[np.newaxis, ...]
            E_phi = E_phi[np.newaxis, ...]
            single_sample = True
        else:
            single_sample = False
        
        n_samples = E_theta.shape[0]
        logger.debug(f"Running inference on {n_samples} field samples")
        
        # Preprocessing
        preproc_start = time.time()
        if not self.preprocessing.is_fitted:
            raise RuntimeError("Preprocessing pipeline must be fitted before inference")
        
        X = self.preprocessing.transform_features(E_theta, E_phi)
        preprocessing_time = time.time() - preproc_start
        
        # Model inference
        inference_start = time.time()
        if n_samples <= self.config.batch_size:
            # Single batch
            y_pred = self.model.predict(X, **kwargs)
        else:
            # Multiple batches
            y_pred = self.model.predict_batch(X, batch_size=self.config.batch_size, **kwargs)
        
        inference_time = time.time() - inference_start
        
        # Post-process predictions
        y_pred_denorm = self.preprocessing.inverse_transform_targets(y_pred)
        coeffs_e, coeffs_m = self.preprocessing.unprocess_coefficients(y_pred_denorm)
        
        # Remove batch dimension for single samples
        if single_sample:
            coeffs_e = coeffs_e[0]
            coeffs_m = coeffs_m[0]
        
        # Validation if enabled
        validation_results = None
        if self.config.enable_result_validation:
            validation_results = self._validate_predictions(coeffs_e, coeffs_m)
        
        # Update statistics
        self._inference_count += n_samples
        self._total_inference_time += inference_time
        
        total_time = time.time() - start_time
        
        logger.debug(f"Inference completed in {total_time:.4f}s ({inference_time:.4f}s model)")
        
        return InferenceResult(
            coefficients_e=coeffs_e,
            coefficients_m=coeffs_m,
            n_samples=n_samples,
            inference_time=inference_time,
            model_type=self.model.model_type,
            raw_predictions=y_pred if self.config.include_metadata else None,
            preprocessing_time=preprocessing_time,
            validation_results=validation_results
        )
    
    def predict_from_power(
        self,
        power: np.ndarray,
        **kwargs
    ) -> InferenceResult:
        """
        Make predictions from power patterns.
        
        Args:
            power: Power patterns (n_samples, n_phi, n_theta) or (n_phi, n_theta)
            **kwargs: Additional inference parameters
            
        Returns:
            Inference result with coefficients
        """
        # Convert power to field amplitudes (assuming uniform phase)
        # This is a simplified approach - in practice you might have more sophisticated methods
        amplitude = np.sqrt(power)
        E_theta = amplitude.astype(complex)
        E_phi = np.zeros_like(E_theta)  # Assume theta-polarized
        
        return self.predict_from_fields(E_theta, E_phi, **kwargs)
    
    def predict_batch_from_files(
        self,
        field_files: List[Union[str, Path]],
        **kwargs
    ) -> List[InferenceResult]:
        """
        Make predictions from field data files.
        
        Args:
            field_files: List of field file paths
            **kwargs: Additional inference parameters
            
        Returns:
            List of inference results
        """
        results = []
        
        for field_file in field_files:
            try:
                # Load field data (this would need implementation based on your file format)
                E_theta, E_phi = self._load_field_file(Path(field_file))
                
                # Make prediction
                result = self.predict_from_fields(E_theta, E_phi, **kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {field_file}: {e}")
                # Add error result
                results.append(InferenceResult(
                    coefficients_e=np.array([]),
                    coefficients_m=np.array([]),
                    n_samples=0,
                    inference_time=0.0,
                    model_type=self.model.model_type,
                    validation_results={"error": str(e)}
                ))
        
        return results
    
    def _validate_predictions(
        self, 
        coeffs_e: np.ndarray, 
        coeffs_m: np.ndarray
    ) -> Dict[str, Any]:
        """Validate prediction results."""
        validation = {}
        
        try:
            # Check for NaN or infinite values
            validation["has_nan"] = bool(np.any(np.isnan(coeffs_e)) or np.any(np.isnan(coeffs_m)))
            validation["has_inf"] = bool(np.any(np.isinf(coeffs_e)) or np.any(np.isinf(coeffs_m)))
            
            # Check magnitude thresholds
            max_mag_e = float(np.max(np.abs(coeffs_e)))
            max_mag_m = float(np.max(np.abs(coeffs_m)))
            validation["max_magnitude_e"] = max_mag_e
            validation["max_magnitude_m"] = max_mag_m
            validation["magnitude_ok"] = (
                max_mag_e < self.config.coefficient_magnitude_threshold and
                max_mag_m < self.config.coefficient_magnitude_threshold
            )
            
            # Compute coefficient statistics
            validation["total_power_e"] = float(np.sum(np.abs(coeffs_e) ** 2))
            validation["total_power_m"] = float(np.sum(np.abs(coeffs_m) ** 2))
            validation["total_power"] = validation["total_power_e"] + validation["total_power_m"]
            
        except Exception as e:
            validation["validation_error"] = str(e)
        
        return validation
    
    def _load_field_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load field data from file (placeholder implementation)."""
        # This would implement your specific field file format
        # For now, return dummy data
        logger.warning(f"Field file loading not implemented: {file_path}")
        n_phi, n_theta = 360, 179
        E_theta = np.random.random((n_phi, n_theta)) + 1j * np.random.random((n_phi, n_theta))
        E_phi = np.random.random((n_phi, n_theta)) + 1j * np.random.random((n_phi, n_theta))
        return E_theta, E_phi
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        avg_inference_time = (
            self._total_inference_time / self._inference_count 
            if self._inference_count > 0 else 0.0
        )
        
        return {
            "model_info": self.model.get_model_info(),
            "preprocessing_stats": self.preprocessing.get_stats(),
            "inference_count": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_inference_time,
            "config": self.config.__dict__,
        }


# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor:
    """
    Optimized batch processor for large-scale inference.
    
    Handles memory management, parallel processing, and progress tracking
    for processing large datasets efficiently.
    """
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            inference_engine: Configured inference engine
            config: Processing configuration
        """
        self.engine = inference_engine
        self.config = config or InferenceConfig()
        
        # Processing state
        self._processed_count = 0
        self._failed_count = 0
        self._total_processing_time = 0.0
        
        logger.info(f"Initialized BatchProcessor (batch_size={self.config.batch_size})")
    
    def process_field_batch(
        self,
        E_theta_batch: np.ndarray,
        E_phi_batch: np.ndarray,
        progress_callback: Optional[callable] = None
    ) -> List[InferenceResult]:
        """
        Process a large batch of field data.
        
        Args:
            E_theta_batch: E_theta components (n_samples, n_phi, n_theta)
            E_phi_batch: E_phi components (n_samples, n_phi, n_theta)  
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of inference results
        """
        n_samples = E_theta_batch.shape[0]
        batch_size = min(self.config.batch_size, self.config.max_batch_size)
        
        logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
        
        results = []
        start_time = time.time()
        
        for i in range(0, n_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, n_samples)
            batch_num = i // batch_size + 1
            total_batches = (n_samples + batch_size - 1) // batch_size
            
            try:
                # Extract batch
                E_theta_batch_slice = E_theta_batch[batch_start:batch_end]
                E_phi_batch_slice = E_phi_batch[batch_start:batch_end]
                
                # Process batch
                batch_result = self.engine.predict_from_fields(
                    E_theta_batch_slice, 
                    E_phi_batch_slice
                )
                results.append(batch_result)
                
                self._processed_count += batch_result.n_samples
                
                # Progress callback
                if progress_callback is not None:
                    progress = {
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "samples_processed": batch_end,
                        "total_samples": n_samples,
                        "elapsed_time": time.time() - start_time,
                    }
                    progress_callback(progress)
                
                # Garbage collection
                if batch_num % self.config.gc_frequency == 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                self._failed_count += batch_end - batch_start
                
                # Add error result
                results.append(InferenceResult(
                    coefficients_e=np.array([]),
                    coefficients_m=np.array([]),
                    n_samples=0,
                    inference_time=0.0,
                    model_type=self.engine.model.model_type,
                    validation_results={"error": str(e)}
                ))
        
        total_time = time.time() - start_time
        self._total_processing_time += total_time
        
        logger.info(
            f"Batch processing completed: {self._processed_count} processed, "
            f"{self._failed_count} failed, {total_time:.2f}s total"
        )
        
        return results
    
    def process_files(
        self,
        file_list: List[Union[str, Path]], 
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple field files with results saving.
        
        Args:
            file_list: List of field file paths
            output_dir: Directory to save results (optional)
            progress_callback: Optional progress callback
            
        Returns:
            Processing summary
        """
        logger.info(f"Processing {len(file_list)} files")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        failed_files = []
        start_time = time.time()
        
        for i, file_path in enumerate(file_list):
            try:
                # Process single file
                file_results = self.engine.predict_batch_from_files([file_path])
                results.extend(file_results)
                
                # Save result if output directory provided
                if output_dir is not None and file_results:
                    result_file = output_dir / f"{Path(file_path).stem}_result.json"
                    with open(result_file, 'w') as f:
                        json.dump(file_results[0].to_dict(), f, indent=2)
                
                # Progress callback
                if progress_callback is not None:
                    progress = {
                        "file": i + 1,
                        "total_files": len(file_list),
                        "current_file": str(file_path),
                        "elapsed_time": time.time() - start_time,
                    }
                    progress_callback(progress)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                failed_files.append(str(file_path))
        
        total_time = time.time() - start_time
        
        summary = {
            "total_files": len(file_list),
            "successful_files": len(file_list) - len(failed_files),
            "failed_files": failed_files,
            "total_results": len(results),
            "processing_time": total_time,
            "output_directory": str(output_dir) if output_dir else None,
        }
        
        logger.info(f"File processing completed: {summary}")
        return summary


# =============================================================================
# Model Server
# =============================================================================

class ModelServer:
    """
    High-level model server for production inference.
    
    Provides a simple interface for loading models and serving predictions
    with automatic preprocessing and result formatting.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model server.
        
        Args:
            config: Global configuration
        """
        self.config = config or Config()
        
        # Server state
        self._loaded_models: Dict[str, InferenceEngine] = {}
        self._model_registry = get_model_registry()
        
        logger.info("Initialized ModelServer")
    
    def load_model(
        self,
        model_name: str,
        model_path: Union[str, Path],
        model_type: Optional[str] = None,
        preprocessing_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Load a trained model for serving.
        
        Args:
            model_name: Name to identify the model
            model_path: Path to saved model
            model_type: Model type (auto-detected if None)
            preprocessing_path: Path to preprocessing pipeline (optional)
        """
        logger.info(f"Loading model '{model_name}' from {model_path}")
        
        try:
            # Auto-detect model type if not provided
            if model_type is None:
                model_info_path = Path(model_path) / "model_info.json"
                if model_info_path.exists():
                    with open(model_info_path) as f:
                        model_info = json.load(f)
                    model_type = model_info.get("model_type", "mlp")
                else:
                    model_type = "mlp"  # Default
            
            # Create and load model
            # This is simplified - in practice you'd need to determine config from saved model
            model = self._model_registry.create_model(
                model_type=model_type,
                input_dim=256,  # This should come from saved model info
                output_dim=240  # This should come from saved model info
            )
            model.load(model_path)
            
            # Load preprocessing pipeline
            preprocessing = None
            if preprocessing_path is not None:
                preprocessing = PreprocessingPipeline()
                preprocessing.load(preprocessing_path)
            
            # Create inference engine
            engine = InferenceEngine(model, preprocessing)
            self._loaded_models[model_name] = engine
            
            logger.info(f"Model '{model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def predict(
        self,
        model_name: str,
        E_theta: np.ndarray,
        E_phi: np.ndarray,
        **kwargs
    ) -> InferenceResult:
        """
        Make prediction with a loaded model.
        
        Args:
            model_name: Name of loaded model
            E_theta: E_theta component
            E_phi: E_phi component
            **kwargs: Additional inference parameters
            
        Returns:
            Inference result
        """
        if model_name not in self._loaded_models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        engine = self._loaded_models[model_name]
        return engine.predict_from_fields(E_theta, E_phi, **kwargs)
    
    def list_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._loaded_models.keys())
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from server."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info(f"Model '{model_name}' unloaded")
        else:
            logger.warning(f"Model '{model_name}' not found")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name not in self._loaded_models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        engine = self._loaded_models[model_name]
        return engine.get_stats()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_inference_engine(
    model_path: Union[str, Path],
    model_type: str = "mlp",
    **config_kwargs
) -> InferenceEngine:
    """
    Create an inference engine from a saved model.
    
    Args:
        model_path: Path to saved model
        model_type: Type of model
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured inference engine
    """
    # Load model
    registry = get_model_registry()
    model = registry.create_model(model_type, input_dim=256, output_dim=240)
    model.load(model_path)
    
    # Create inference engine
    config = InferenceConfig(**config_kwargs)
    return InferenceEngine(model, config=config)