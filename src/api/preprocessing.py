"""
Preprocessing Pipeline for Model Inference.

This module provides preprocessing components for transforming raw field data
into model-ready features, including PCA, normalization, and validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.config import Config, DataConfig
from src.core.data_generator import get_mode_list, pack_coefficients, unpack_coefficients

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # PCA settings
    pca_components: int = 256
    pca_oversample: int = 16
    pca_iterations: int = 0
    
    # Normalization
    normalize_features: bool = True
    normalize_targets: bool = True
    
    # Validation
    validate_inputs: bool = True
    clip_outliers: bool = True
    outlier_std_threshold: float = 5.0
    
    # Grid settings
    expected_n_phi: int = 360
    expected_n_theta: int = 179


# =============================================================================
# PCA Transformer
# =============================================================================

class PCATransformer:
    """
    PCA transformer for field data preprocessing.
    
    Handles dimensionality reduction of concatenated E_theta and E_phi fields
    using randomized PCA for computational efficiency.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize PCA transformer.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.is_fitted = False
        
        # PCA components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        
        logger.debug(f"Initialized PCATransformer (n_components={config.pca_components})")
    
    def fit(self, X: np.ndarray, use_incremental: Optional[bool] = None) -> PCATransformer:
        """
        Fit PCA on training data with automatic strategy selection.
        
        Args:
            X: Training data (n_samples, n_features)
            use_incremental: Force incremental PCA (None = auto-select)
            
        Returns:
            Self for method chaining
        """
        n_samples, n_features = X.shape
        logger.info(f"Fitting PCA: {X.shape} -> {self.config.pca_components} components")
        
        # Automatic strategy selection based on data size
        memory_size_gb = (n_samples * n_features * 8) / (1024**3)  # Estimate for float64
        
        if use_incremental is True or (use_incremental is None and memory_size_gb > 2.0):
            # Use incremental PCA for large datasets (>2GB estimated memory)
            logger.info(f"Using IncrementalPCA (estimated memory: {memory_size_gb:.1f}GB)")
            batch_size = min(1000, max(100, n_samples // 20))  # Adaptive batch size
            self._fit_incremental_pca(X, batch_size=batch_size)
            
        elif self.config.pca_components >= min(X.shape):
            # Use standard PCA if components >= min dimension
            logger.info("Using standard PCA (full SVD)")
            self._fit_standard_pca(X)
            
        else:
            # Use randomized PCA for large datasets
            logger.info("Using randomized PCA")
            self._fit_randomized_pca(X)
        
        self.is_fitted = True
        logger.info(f"✅ PCA fitted, explained variance ratio: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Data to transform (n_samples, n_features)
            
        Returns:
            Transformed data (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted before transform")
        
        # Center data
        X_centered = X - self.mean_
        
        # Apply PCA transformation
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed.astype(np.float32)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform PCA-compressed data.
        
        Args:
            X_transformed: PCA-transformed data (n_samples, n_components)
            
        Returns:
            Reconstructed data (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted before inverse_transform")
        
        # Apply inverse PCA transformation
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        return X_reconstructed
    
    def _fit_standard_pca(self, X: np.ndarray) -> None:
        """Fit using standard PCA (via SVD)."""
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Extract components and explained variance
        self.components_ = Vt[:self.config.pca_components]
        
        # Compute explained variance ratio
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        self.explained_variance_ratio_ = explained_variance[:self.config.pca_components] / total_variance
    
    def _fit_randomized_pca(self, X: np.ndarray) -> None:
        """Fit using randomized PCA for efficiency."""
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA(
                n_components=self.config.pca_components,
                svd_solver='randomized',
                random_state=42
            )
            
            pca.fit(X)
            
            self.components_ = pca.components_
            self.mean_ = pca.mean_
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            
        except ImportError:
            logger.warning("sklearn not available, falling back to standard PCA")
            self._fit_standard_pca(X)
    
    def _fit_incremental_pca(self, X: np.ndarray, batch_size: int = 1000) -> None:
        """
        Fit using IncrementalPCA for memory-efficient processing of large datasets.
        
        Args:
            X: Input data matrix (n_samples, n_features)
            batch_size: Batch size for incremental fitting
        """
        try:
            from sklearn.decomposition import IncrementalPCA
            
            pca = IncrementalPCA(
                n_components=self.config.pca_components,
                batch_size=batch_size
            )
            
            n_samples = X.shape[0]
            logger.info(f"Fitting IncrementalPCA on {n_samples} samples with batch_size={batch_size}")
            
            # Fit in batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X[start_idx:end_idx]
                pca.partial_fit(X_batch)
                
                if batch_idx % 10 == 0:
                    logger.info(f"  IncrementalPCA batch {batch_idx + 1}/{n_batches}")
            
            self.components_ = pca.components_
            self.mean_ = pca.mean_
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            
            # Store the fitted PCA object for streaming transforms
            self._incremental_pca = pca
            
            logger.info("✅ IncrementalPCA fitting completed")
            
        except ImportError:
            logger.warning("sklearn not available, falling back to standard PCA")
            self._fit_standard_pca(X)
    
    def fit_streaming(
        self, 
        data_generator,
        total_samples: int,
        batch_size: int = 1000,
        n_fit_samples: Optional[int] = None
    ) -> 'PCATransformer':
        """
        Fit PCA using streaming data generator for very large datasets.
        
        Args:
            data_generator: Generator that yields (X_batch,) tuples
            total_samples: Total number of samples in dataset
            batch_size: Batch size for incremental fitting
            n_fit_samples: Optional limit on number of samples to use for fitting
            
        Returns:
            Self for method chaining
        """
        try:
            from sklearn.decomposition import IncrementalPCA
            
            pca = IncrementalPCA(
                n_components=self.config.pca_components,
                batch_size=batch_size
            )
            
            samples_processed = 0
            max_samples = n_fit_samples or total_samples
            
            logger.info(f"Fitting streaming PCA on up to {max_samples} samples")
            
            for batch_idx, X_batch in enumerate(data_generator):
                if samples_processed >= max_samples:
                    break
                
                # Limit batch size if needed
                remaining_samples = max_samples - samples_processed
                if X_batch.shape[0] > remaining_samples:
                    X_batch = X_batch[:remaining_samples]
                
                pca.partial_fit(X_batch)
                samples_processed += X_batch.shape[0]
                
                if batch_idx % 10 == 0:
                    logger.info(f"  Streaming PCA: processed {samples_processed}/{max_samples} samples")
            
            self.components_ = pca.components_
            self.mean_ = pca.mean_
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            self._incremental_pca = pca
            self.is_fitted = True
            
            logger.info(f"✅ Streaming PCA completed: {samples_processed} samples processed")
            
            return self
            
        except ImportError:
            logger.error("sklearn not available, cannot perform streaming PCA")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get PCA statistics."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "n_components": self.config.pca_components,
            "input_dim": self.mean_.shape[0] if self.mean_ is not None else None,
            "explained_variance_ratio": float(np.sum(self.explained_variance_ratio_)),
            "components_shape": self.components_.shape if self.components_ is not None else None,
        }


# =============================================================================
# Feature Normalizer
# =============================================================================

class FeatureNormalizer:
    """
    Feature normalizer with support for different normalization strategies.
    """
    
    def __init__(self, strategy: str = "standard"):
        """
        Initialize feature normalizer.
        
        Args:
            strategy: Normalization strategy ("standard", "minmax", "robust")
        """
        self.strategy = strategy
        self.is_fitted = False
        
        # Normalization parameters
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        
        logger.debug(f"Initialized FeatureNormalizer (strategy={strategy})")
    
    def fit(self, X: np.ndarray) -> FeatureNormalizer:
        """
        Fit normalizer on training data.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.strategy == "standard":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)  # Avoid division by zero
            
        elif self.strategy == "minmax":
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
            range_val = self.max_ - self.min_
            self.max_ = np.where(range_val == 0, self.min_ + 1.0, self.max_)  # Avoid division by zero
            
        elif self.strategy == "robust":
            self.mean_ = np.median(X, axis=0)
            self.std_ = np.median(np.abs(X - self.mean_), axis=0) * 1.4826  # MAD * constant
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
            
        else:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalization.
        
        Args:
            X: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if self.strategy == "standard" or self.strategy == "robust":
            return (X - self.mean_) / self.std_
        elif self.strategy == "minmax":
            return (X - self.min_) / (self.max_ - self.min_)
    
    def inverse_transform(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            X_normalized: Normalized data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        if self.strategy == "standard" or self.strategy == "robust":
            return X_normalized * self.std_ + self.mean_
        elif self.strategy == "minmax":
            return X_normalized * (self.max_ - self.min_) + self.min_


# =============================================================================
# Field Preprocessor
# =============================================================================

class FieldPreprocessor:
    """
    Preprocessor for electromagnetic field data.
    
    Handles conversion from field amplitudes to model-ready features,
    including concatenation, PCA transformation, and normalization.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize field preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.is_fitted = False
        
        # Components
        self.pca_transformer = PCATransformer(config)
        self.feature_normalizer = FeatureNormalizer("standard") if config.normalize_features else None
        
        logger.info(f"Initialized FieldPreprocessor (PCA: {config.pca_components}, normalize: {config.normalize_features})")
    
    def fit(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> FieldPreprocessor:
        """
        Fit preprocessor on training field data.
        
        Args:
            E_theta: E_theta component (n_samples, n_phi, n_theta)
            E_phi: E_phi component (n_samples, n_phi, n_theta)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting field preprocessor on {E_theta.shape[0]} samples")
        
        # Validate inputs
        self._validate_field_shapes(E_theta, E_phi)
        
        # Concatenate and flatten fields
        X = self._prepare_features(E_theta, E_phi)
        
        # Fit PCA
        X_pca = self.pca_transformer.fit_transform(X)
        
        # Fit feature normalizer if enabled
        if self.feature_normalizer is not None:
            self.feature_normalizer.fit(X_pca)
        
        self.is_fitted = True
        logger.info("Field preprocessor fitted successfully")
        
        return self
    
    def transform(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> np.ndarray:
        """
        Transform field data to model features.
        
        Args:
            E_theta: E_theta component (n_samples, n_phi, n_theta)  
            E_phi: E_phi component (n_samples, n_phi, n_theta)
            
        Returns:
            Model-ready features (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Validate inputs
        self._validate_field_shapes(E_theta, E_phi)
        
        # Prepare features
        X = self._prepare_features(E_theta, E_phi)
        
        # Apply PCA
        X_pca = self.pca_transformer.transform(X)
        
        # Apply normalization if enabled
        if self.feature_normalizer is not None:
            X_pca = self.feature_normalizer.transform(X_pca)
        
        return X_pca
    
    def fit_transform(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            E_theta: E_theta component  
            E_phi: E_phi component
            
        Returns:
            Model-ready features
        """
        return self.fit(E_theta, E_phi).transform(E_theta, E_phi)
    
    def _prepare_features(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> np.ndarray:
        """Prepare features from field components."""
        # Flatten spatial dimensions
        E_theta_flat = E_theta.reshape(E_theta.shape[0], -1)
        E_phi_flat = E_phi.reshape(E_phi.shape[0], -1)
        
        # Concatenate components
        X = np.concatenate([E_theta_flat, E_phi_flat], axis=1)
        
        # Handle complex values (convert to real)
        if np.iscomplexobj(X):
            X = np.concatenate([X.real, X.imag], axis=1)
        
        return X.astype(np.float32)
    
    def _validate_field_shapes(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> None:
        """Validate field data shapes."""
        if not self.config.validate_inputs:
            return
        
        # Check dimensions
        if E_theta.ndim != 3 or E_phi.ndim != 3:
            raise ValueError(f"Expected 3D field arrays, got shapes {E_theta.shape}, {E_phi.shape}")
        
        # Check matching shapes
        if E_theta.shape != E_phi.shape:
            raise ValueError(f"E_theta and E_phi must have same shape, got {E_theta.shape} vs {E_phi.shape}")
        
        # Check expected grid dimensions
        _, n_phi, n_theta = E_theta.shape
        if n_phi != self.config.expected_n_phi or n_theta != self.config.expected_n_theta:
            logger.warning(
                f"Unexpected grid dimensions: {n_phi}x{n_theta}, "
                f"expected {self.config.expected_n_phi}x{self.config.expected_n_theta}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessor statistics."""
        stats = {
            "fitted": self.is_fitted,
            "pca_stats": self.pca_transformer.get_stats(),
        }
        
        if self.feature_normalizer is not None:
            stats["normalization"] = {
                "enabled": True,
                "strategy": self.feature_normalizer.strategy,
                "fitted": self.feature_normalizer.is_fitted
            }
        else:
            stats["normalization"] = {"enabled": False}
        
        return stats


# =============================================================================
# Complete Preprocessing Pipeline
# =============================================================================

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for ML inference.
    
    Integrates field preprocessing, target normalization, and provides
    utilities for batch processing and validation.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration (creates default if None)
        """
        self.config = config or PreprocessingConfig()
        
        # Components
        self.field_preprocessor = FieldPreprocessor(self.config)
        self.target_normalizer = FeatureNormalizer("standard") if self.config.normalize_targets else None
        
        # State
        self.is_fitted = False
        
        logger.info("Initialized complete PreprocessingPipeline")
    
    def fit(
        self,
        E_theta: np.ndarray,
        E_phi: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> PreprocessingPipeline:
        """
        Fit complete preprocessing pipeline.
        
        Args:
            E_theta: Training E_theta data
            E_phi: Training E_phi data  
            targets: Training targets (optional, for target normalization)
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting complete preprocessing pipeline")
        
        # Fit field preprocessor
        self.field_preprocessor.fit(E_theta, E_phi)
        
        # Fit target normalizer if provided
        if targets is not None and self.target_normalizer is not None:
            self.target_normalizer.fit(targets)
            logger.debug("Target normalizer fitted")
        
        self.is_fitted = True
        return self
    
    def transform_features(
        self, 
        E_theta: np.ndarray, 
        E_phi: np.ndarray
    ) -> np.ndarray:
        """
        Transform field data to model features.
        
        Args:
            E_theta: E_theta component
            E_phi: E_phi component
            
        Returns:
            Model-ready features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        return self.field_preprocessor.transform(E_theta, E_phi)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Transform targets using fitted normalization.
        
        Args:
            targets: Target values
            
        Returns:
            Normalized targets
        """
        if self.target_normalizer is None:
            return targets
        
        if not self.target_normalizer.is_fitted:
            raise RuntimeError("Target normalizer not fitted")
        
        return self.target_normalizer.transform(targets)
    
    def inverse_transform_targets(self, targets_normalized: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized targets.
        
        Args:
            targets_normalized: Normalized targets
            
        Returns:
            Original scale targets
        """
        if self.target_normalizer is None:
            return targets_normalized
        
        return self.target_normalizer.inverse_transform(targets_normalized)
    
    def process_coefficients(
        self, 
        a_e: np.ndarray, 
        a_m: np.ndarray
    ) -> np.ndarray:
        """
        Process coefficient arrays into model targets.
        
        Args:
            a_e: Electric coefficients (..., n_modes)
            a_m: Magnetic coefficients (..., n_modes)
            
        Returns:
            Packed and normalized targets
        """
        # Pack coefficients
        targets = pack_coefficients(a_e, a_m)
        
        # Apply target normalization if enabled
        if self.target_normalizer is not None and self.target_normalizer.is_fitted:
            targets = self.target_normalizer.transform(targets)
        
        return targets
    
    def unprocess_coefficients(
        self, 
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unprocess model targets back to coefficient arrays.
        
        Args:
            targets: Model targets
            
        Returns:
            Tuple of (a_e, a_m) coefficient arrays
        """
        # Inverse transform targets if needed
        if self.target_normalizer is not None and self.target_normalizer.is_fitted:
            targets = self.target_normalizer.inverse_transform(targets)
        
        # Unpack coefficients
        return unpack_coefficients(targets)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save preprocessing pipeline to disk."""
        import json
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config as JSON for human readability
        config_path = save_path / "config.json"
        config_dict = {
            "pca": {
                "pca_components": self.config.pca_components,
                "pca_oversample": self.config.pca_oversample,
                "pca_iterations": self.config.pca_iterations
            },
            "normalization": {
                "normalize_features": self.config.normalize_features,
                "normalize_targets": self.config.normalize_targets,
                "validate_inputs": self.config.validate_inputs,
                "clip_outliers": self.config.clip_outliers,
                "outlier_std_threshold": self.config.outlier_std_threshold
            },
            "grid_settings": {
                "expected_n_phi": self.config.expected_n_phi,
                "expected_n_theta": self.config.expected_n_theta
            },
            "is_fitted": self.is_fitted
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Collect fitted state data
        fitted_data = {}
        
        # Save PCA transformer state
        if hasattr(self.field_preprocessor, 'pca_transformer') and self.field_preprocessor.pca_transformer.is_fitted:
            pca = self.field_preprocessor.pca_transformer
            fitted_data.update({
                'pca_components': pca.components_,
                'pca_mean': pca.mean_,
                'pca_explained_variance_ratio': pca.explained_variance_ratio_,
                'pca_is_fitted': pca.is_fitted
            })
        
        # Save feature normalizer state
        if (hasattr(self.field_preprocessor, 'feature_normalizer') and 
            self.field_preprocessor.feature_normalizer is not None and 
            self.field_preprocessor.feature_normalizer.is_fitted):
            
            norm = self.field_preprocessor.feature_normalizer
            fitted_data['feature_normalizer_strategy'] = norm.strategy
            fitted_data['feature_normalizer_is_fitted'] = norm.is_fitted
            
            if norm.strategy in ["standard", "robust"]:
                fitted_data['feature_normalizer_mean'] = norm.mean_
                fitted_data['feature_normalizer_std'] = norm.std_
            elif norm.strategy == "minmax":
                fitted_data['feature_normalizer_min'] = norm.min_
                fitted_data['feature_normalizer_max'] = norm.max_
        
        # Save target normalizer state
        if (hasattr(self, 'target_normalizer') and 
            self.target_normalizer is not None and 
            self.target_normalizer.is_fitted):
            
            norm = self.target_normalizer
            fitted_data['target_normalizer_strategy'] = norm.strategy
            fitted_data['target_normalizer_is_fitted'] = norm.is_fitted
            
            if norm.strategy in ["standard", "robust"]:
                fitted_data['target_normalizer_mean'] = norm.mean_
                fitted_data['target_normalizer_std'] = norm.std_
            elif norm.strategy == "minmax":
                fitted_data['target_normalizer_min'] = norm.min_
                fitted_data['target_normalizer_max'] = norm.max_
        
        # Save fitted state as compressed numpy arrays
        if fitted_data:
            state_path = save_path / "fitted_state.npz"
            np.savez_compressed(str(state_path), **fitted_data)
        
        logger.info(f"✅ Preprocessing pipeline saved to {save_path} (config + fitted state)")
        logger.info(f"   Saved {len(fitted_data)} fitted components to {state_path if fitted_data else 'none'}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load preprocessing pipeline from disk."""
        import json
        
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessing pipeline directory not found: {load_path}")
        
        # Load config from JSON
        config_path = load_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Verify configuration compatibility (just log warnings for now)
        if config_dict.get("pca", {}).get("pca_components") != self.config.pca_components:
            logger.warning(f"PCA components mismatch: saved={config_dict.get('pca', {}).get('pca_components')}, current={self.config.pca_components}")
        
        # Load fitted state from compressed numpy arrays
        state_path = load_path / "fitted_state.npz"
        if not state_path.exists():
            logger.warning(f"No fitted state file found: {state_path}")
            return
        
        fitted_data = np.load(str(state_path))
        components_loaded = 0
        
        # Restore PCA transformer state
        if 'pca_is_fitted' in fitted_data and fitted_data['pca_is_fitted'].item():
            pca = self.field_preprocessor.pca_transformer
            pca.components_ = fitted_data['pca_components']
            pca.mean_ = fitted_data['pca_mean']
            pca.explained_variance_ratio_ = fitted_data['pca_explained_variance_ratio']
            pca.is_fitted = True
            components_loaded += 1
            logger.debug(f"Restored PCA transformer state: {pca.components_.shape}")
        
        # Restore feature normalizer state
        if 'feature_normalizer_is_fitted' in fitted_data and fitted_data['feature_normalizer_is_fitted'].item():
            # Ensure feature normalizer exists
            if self.field_preprocessor.feature_normalizer is None:
                strategy = fitted_data['feature_normalizer_strategy'].item()
                self.field_preprocessor.feature_normalizer = FeatureNormalizer(strategy)
            
            norm = self.field_preprocessor.feature_normalizer
            norm.strategy = fitted_data['feature_normalizer_strategy'].item()
            norm.is_fitted = True
            
            if norm.strategy in ["standard", "robust"]:
                norm.mean_ = fitted_data['feature_normalizer_mean']
                norm.std_ = fitted_data['feature_normalizer_std']
            elif norm.strategy == "minmax":
                norm.min_ = fitted_data['feature_normalizer_min']
                norm.max_ = fitted_data['feature_normalizer_max']
            
            components_loaded += 1
            logger.debug(f"Restored feature normalizer state: strategy={norm.strategy}")
        
        # Restore target normalizer state
        if 'target_normalizer_is_fitted' in fitted_data and fitted_data['target_normalizer_is_fitted'].item():
            # Ensure target normalizer exists
            if self.target_normalizer is None:
                strategy = fitted_data['target_normalizer_strategy'].item()
                self.target_normalizer = FeatureNormalizer(strategy)
            
            norm = self.target_normalizer
            norm.strategy = fitted_data['target_normalizer_strategy'].item()
            norm.is_fitted = True
            
            if norm.strategy in ["standard", "robust"]:
                norm.mean_ = fitted_data['target_normalizer_mean']
                norm.std_ = fitted_data['target_normalizer_std']
            elif norm.strategy == "minmax":
                norm.min_ = fitted_data['target_normalizer_min']
                norm.max_ = fitted_data['target_normalizer_max']
            
            components_loaded += 1
            logger.debug(f"Restored target normalizer state: strategy={norm.strategy}")
        
        # Update pipeline fitted state
        self.is_fitted = config_dict.get('is_fitted', False)
        
        # Mark field preprocessor as fitted if any components were restored
        if components_loaded > 0:
            self.field_preprocessor.is_fitted = True
        
        logger.info(f"✅ Preprocessing pipeline loaded from {load_path}")
        logger.info(f"   Restored {components_loaded} fitted components")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "fitted": self.is_fitted,
            "field_preprocessor": self.field_preprocessor.get_stats(),
            "target_normalizer": {
                "enabled": self.target_normalizer is not None,
                "fitted": self.target_normalizer.is_fitted if self.target_normalizer else False
            }
        }
    
    def fit_streaming(
        self,
        E_theta_path: str,
        E_phi_path: str,
        targets_path: Optional[str] = None,
        max_fit_samples: int = 10000,
        batch_size: int = 1000
    ) -> 'PreprocessingPipeline':
        """
        Fit preprocessing pipeline using streaming data from memory-mapped files.
        
        Args:
            E_theta_path: Path to E_theta memory-mapped file
            E_phi_path: Path to E_phi memory-mapped file  
            targets_path: Optional path to targets memory-mapped file
            max_fit_samples: Maximum samples to use for fitting
            batch_size: Batch size for processing
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessing pipeline in streaming mode...")
        
        # Load memory-mapped arrays
        E_theta_mm = np.load(E_theta_path, mmap_mode='r')
        E_phi_mm = np.load(E_phi_path, mmap_mode='r')
        n_samples = E_theta_mm.shape[0]
        
        # Limit samples for fitting if dataset is very large
        fit_samples = min(max_fit_samples, n_samples)
        if fit_samples < n_samples:
            logger.info(f"Using {fit_samples} samples for fitting (from {n_samples} total)")
            # Use random subset
            np.random.seed(42)
            fit_indices = np.random.choice(n_samples, fit_samples, replace=False)
            fit_indices.sort()  # Sort for better memory access pattern
        else:
            fit_indices = np.arange(n_samples)
        
        # Create data generator for streaming PCA
        def feature_batch_generator():
            for i in range(0, len(fit_indices), batch_size):
                batch_indices = fit_indices[i:i + batch_size]
                E_theta_batch = E_theta_mm[batch_indices]
                E_phi_batch = E_phi_mm[batch_indices]
                
                # Prepare features
                X_batch = self.field_preprocessor._prepare_features(E_theta_batch, E_phi_batch)
                yield X_batch
        
        # Fit PCA using streaming approach
        self.field_preprocessor.pca_transformer.fit_streaming(
            feature_batch_generator(),
            total_samples=fit_samples,
            batch_size=batch_size
        )
        
        # Mark field preprocessor as fitted
        self.field_preprocessor.is_fitted = True
        
        # Fit target normalizer if needed and targets provided
        if self.config.normalize_targets and targets_path:
            targets_mm = np.load(targets_path, mmap_mode='r')
            
            if fit_samples < n_samples:
                targets_subset = targets_mm[fit_indices]
            else:
                targets_subset = np.array(targets_mm)  # Load full array
                
            if self.target_normalizer is not None:
                self.target_normalizer.fit(targets_subset)
            del targets_subset
        
        self.is_fitted = True
        logger.info("✅ Streaming preprocessing pipeline fitted")
        
        return self
    
    def transform_features_streaming(
        self,
        E_theta_path: str,
        E_phi_path: str,
        output_path: str,
        batch_size: int = 2000
    ) -> str:
        """
        Transform features in streaming mode, writing to memory-mapped output.
        
        Args:
            E_theta_path: Path to E_theta memory-mapped file
            E_phi_path: Path to E_phi memory-mapped file
            output_path: Path for output transformed features
            batch_size: Batch size for processing
            
        Returns:
            Path to transformed features file
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming features in streaming mode...")
        
        # Load memory-mapped input arrays
        E_theta_mm = np.load(E_theta_path, mmap_mode='r')
        E_phi_mm = np.load(E_phi_path, mmap_mode='r')
        n_samples = E_theta_mm.shape[0]
        
        # Determine output shape by processing a small batch
        probe_size = min(10, n_samples)
        X_probe = self.transform_features(E_theta_mm[:probe_size], E_phi_mm[:probe_size])
        n_features = X_probe.shape[1]
        del X_probe
        
        # Create output memory-mapped array
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        X_mm = np.lib.format.open_memmap(
            str(output_path), mode='w+', dtype=np.float32,
            shape=(n_samples, n_features)
        )
        
        logger.info(f"Processing {n_samples} samples -> {n_features} features")
        
        # Process in batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            if batch_idx % 5 == 0:
                logger.info(f"  Processing batch {batch_idx + 1}/{n_batches}")
            
            # Load and transform batch
            E_theta_batch = E_theta_mm[start_idx:end_idx]
            E_phi_batch = E_phi_mm[start_idx:end_idx]
            
            X_batch = self.transform_features(E_theta_batch, E_phi_batch)
            X_mm[start_idx:end_idx] = X_batch.astype(np.float32)
            
            # Clear batch data
            del E_theta_batch, E_phi_batch, X_batch
            
            # Periodic garbage collection
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        # Flush to disk
        X_mm.flush()
        logger.info(f"✅ Streaming feature transformation complete: {output_path}")
        
        return str(output_path)