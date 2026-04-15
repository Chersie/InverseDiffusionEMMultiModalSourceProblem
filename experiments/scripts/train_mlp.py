#!/usr/bin/env python3
"""
MLP Model Training Script

This script provides a complete training pipeline for MLP models with
configuration management, experiment tracking, and comprehensive logging.

Usage:
    python train_mlp.py --config configs/mlp_basic.yaml
    python train_mlp.py --config configs/mlp_large.yaml --maxorder 10
    python train_mlp.py --config configs/mlp_basic.yaml --resume results/experiment/
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import yaml
import mlflow

def set_random_seeds(seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducibility."""
    np.random.seed(seed)
    # Set Python's random module seed too
    import random
    random.seed(seed)
    logger.info(f"✅ Set random seeds to {seed} for reproducibility")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import Config
from src.core.data_generator import DataGenerator, pack_coefficients, unpack_coefficients, get_mode_list
from src.core.dependencies import print_environment_info, validate_ml_environment
from src.models.registry import get_model_registry
from src.api.preprocessing import PreprocessingPipeline
from src.api.inference import InferenceEngine

# Add plotting utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.plotting import (
    plot_training_curves,
    plot_prediction_scatter,
    plot_coefficient_comparison,
    plot_field_comparison,
    plot_p_field_comparison,
    create_experiment_summary_plot
)
from utils.mlflow_manager import create_experiment_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def setup_experiment_directory(config: Dict[str, Any]) -> Path:
    """Set up experiment directory with timestamp."""
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = Path("experiments/results") / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    # Save configuration
    config_save_path = experiment_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def generate_training_data_streaming(
    config: Dict[str, Any], 
    cache_dir: Optional[str] = None
) -> Tuple[str, str, str, Optional[str], Dict[str, Any]]:
    """
    Generate synthetic training data using memory-mapped files to avoid memory issues.
    
    Returns:
        Tuple of (E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info)
        where paths point to memory-mapped .npy files on disk.
    """
    from src.data.memory_monitor import monitor_memory, get_memory_usage
    from src.data.streaming_dataset import create_memmap_arrays
    import tempfile
    
    logger.info("Generating synthetic training data using streaming approach...")
    
    # Extract configuration
    maxorder = config['model']['maxorder']
    n_samples = config['training']['n_samples']
    generator_mode = config['data']['generator_mode']
    seed = config['data']['seed']
    loss_type = config['model'].get('loss_type', 'coefficient')
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = config.get('memory', {}).get('cache_dir', 'data/cache')
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Use memory-optimized batch sizes to prevent OOM kills
    if n_samples <= 500:
        batch_size = n_samples
    elif n_samples <= 2000:
        batch_size = 400  # Reduced from 500
    elif n_samples <= 10000:
        batch_size = 600  # Reduced from 800 for better memory management
    else:
        batch_size = 800  # Cap at 800 for very large datasets
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if n_batches > 1:
        logger.info(f"Streaming {n_samples} samples in {n_batches} batches of ~{batch_size}")
    
    # Create data generator
    if generator_mode == "random":
        generator = DataGenerator.for_ml_training()
    else:
        generator = DataGenerator.for_pipeline()
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    with monitor_memory("data_generation", log_before_after=True, warning_threshold_mb=4000):
        
        # Probe shape from single sample
        logger.info("Probing data shapes...")
        sample_dataset = generator.generate_batch(maxorder=maxorder, n_samples=1)
        amplitude_shape = sample_dataset['amplitude'].shape  # (1, n_phi, n_theta, 2)
        E_field_shape = amplitude_shape[1:3]  # (n_phi, n_theta)
        coeffs_shape = pack_coefficients(sample_dataset['coefficients_e'], sample_dataset['coefficients_m']).shape[1:]
        
        logger.info(f"Detected shapes: E_field={E_field_shape}, coeffs={coeffs_shape}")
        
        # Clear probe data
        del sample_dataset
        
        # Create memory-mapped files for output
        timestamp = int(time.time())
        E_theta_path = cache_path / f"E_theta_{timestamp}.npy"
        E_phi_path = cache_path / f"E_phi_{timestamp}.npy"  
        y_coeffs_path = cache_path / f"y_coeffs_{timestamp}.npy"
        y_P_path = None
        
        # Create memory-mapped arrays
        E_theta_mm, E_phi_mm = create_memmap_arrays(
            features_shape=(n_samples,) + E_field_shape,
            targets_shape=(n_samples,) + E_field_shape,
            features_path=E_theta_path,
            targets_path=E_phi_path,
            dtype=np.float32
        )
        
        y_coeffs_mm = np.lib.format.open_memmap(
            str(y_coeffs_path), mode='w+', dtype=np.float32, 
            shape=(n_samples,) + coeffs_shape
        )
        
        y_P_mm = None
        if loss_type == "physics":
            y_P_path = cache_path / f"y_P_{timestamp}.npy"
            # Store as (n_samples, n_theta, n_phi) matching DifferentiableMultipoleField output
            P_field_shape = (E_field_shape[1], E_field_shape[0])  # swap (n_phi,n_theta)->(n_theta,n_phi)
            y_P_mm = np.lib.format.open_memmap(
                str(y_P_path), mode='w+', dtype=np.float32,
                shape=(n_samples,) + P_field_shape
            )
        
        start_time = time.time()
        
        # Process in batches, filling memory-mapped arrays directly
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            if n_batches > 1:
                logger.info(f"Processing batch {batch_idx + 1}/{n_batches}: {current_batch_size} samples")
                if batch_idx % 5 == 0:
                    current_memory = get_memory_usage()
                    logger.info(f"  Current memory usage: {current_memory:.1f}MB")
            
            # Generate batch
            batch_dataset = generator.generate_batch(maxorder=maxorder, n_samples=current_batch_size)
            
            # Extract field components
            E_theta_batch = batch_dataset['amplitude'][..., 0]
            E_phi_batch = batch_dataset['amplitude'][..., 1]
            
            # Fill memory-mapped arrays directly (no concatenation!)
            if np.iscomplexobj(E_theta_batch):
                E_theta_mm[batch_start:batch_end] = np.abs(E_theta_batch).astype(np.float32)
                E_phi_mm[batch_start:batch_end] = np.abs(E_phi_batch).astype(np.float32)
            else:
                E_theta_mm[batch_start:batch_end] = E_theta_batch.astype(np.float32)
                E_phi_mm[batch_start:batch_end] = E_phi_batch.astype(np.float32)
            
            # Pack coefficients directly to memory-mapped array
            y_coeffs_batch = pack_coefficients(
                batch_dataset['coefficients_e'], 
                batch_dataset['coefficients_m']
            )
            y_coeffs_mm[batch_start:batch_end] = y_coeffs_batch.astype(np.float32)
            
            # Compute P field if needed; transpose to (batch, n_theta, n_phi) to match DMF output
            if loss_type == "physics" and y_P_mm is not None:
                if np.iscomplexobj(E_theta_batch):
                    P_batch = (np.abs(E_theta_batch)**2 + np.abs(E_phi_batch)**2).transpose(0, 2, 1).astype(np.float32)
                else:
                    P_batch = (E_theta_batch**2 + E_phi_batch**2).transpose(0, 2, 1).astype(np.float32)
                y_P_mm[batch_start:batch_end] = P_batch
                del P_batch
            
            # Clean up batch data immediately to free memory
            del batch_dataset, E_theta_batch, E_phi_batch, y_coeffs_batch
            
            # Force garbage collection periodically to keep memory usage down
            if batch_idx % 3 == 0:
                gc.collect()
        
        # Flush memory-mapped arrays to disk
        logger.info("Flushing data to disk...")
        E_theta_mm.flush()
        E_phi_mm.flush()
        y_coeffs_mm.flush()
        if y_P_mm is not None:
            y_P_mm.flush()
        
        generation_time = time.time() - start_time
        
    logger.info(f"✅ Generated {n_samples} samples in {generation_time:.2f}s ({n_samples/generation_time:.1f} samples/sec)")
    logger.info(f"✅ Data written to memory-mapped files in {cache_path}")
    
    data_info = {
            'generation_time': generation_time,
            'n_samples': n_samples,
            'E_field_shape': E_field_shape,
            'coeffs_shape': coeffs_shape,
            'n_modes': coeffs_shape[0] // 4,  # Add missing n_modes calculation
            'loss_type': loss_type,
            'has_p_field_targets': y_P_path is not None,
            'batches_used': n_batches,
            'batch_size': batch_size,
            'samples_per_second': n_samples / generation_time,
            'cache_dir': str(cache_path),
            'E_theta_path': str(E_theta_path),
            'E_phi_path': str(E_phi_path),
            'y_coeffs_path': str(y_coeffs_path),
            'y_P_path': str(y_P_path) if y_P_path else None
        }
    
    return str(E_theta_path), str(E_phi_path), str(y_coeffs_path), str(y_P_path) if y_P_path else None, data_info


def generate_training_data(config: Dict[str, Any]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]], 
                                                            Tuple[str, str, str, Optional[str], Dict[str, Any]]]:
    """
    Generate synthetic training data with automatic streaming for large datasets.
    
    Automatically chooses between in-memory and streaming approaches based on
    dataset size and memory configuration.
    
    Returns:
        For small datasets: (E_theta, E_phi, y_coeffs, y_P, data_info) as numpy arrays
        For large datasets: (E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info) as file paths
    """
    n_samples = config['training']['n_samples']
    
    # Check if streaming should be forced
    streaming_config = config.get('streaming', {})
    memory_config = config.get('memory', {})
    
    force_streaming_threshold = streaming_config.get('force_streaming_above_samples', 1000)
    enable_streaming = streaming_config.get('enable_streaming', True)
    
    # Decide between streaming and in-memory based on size and config
    use_streaming = (
        enable_streaming and 
        n_samples > force_streaming_threshold
    )
    
    if use_streaming:
        logger.info(f"Using streaming approach for {n_samples} samples (threshold: {force_streaming_threshold})")
        return generate_training_data_streaming(config)
    else:
        logger.info(f"Using in-memory approach for {n_samples} samples")
        return generate_training_data_legacy(config)


def generate_training_data_legacy(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Legacy in-memory data generation for small datasets.
    
    This is the original implementation that keeps all data in memory.
    Only used for small datasets to maintain performance.
    """
    from src.data.memory_monitor import monitor_memory
    
    logger.info("Generating synthetic training data in memory (legacy mode)...")
    
    # Extract configuration
    maxorder = config['model']['maxorder']
    n_samples = config['training']['n_samples']
    generator_mode = config['data']['generator_mode']
    seed = config['data']['seed']
    loss_type = config['model'].get('loss_type', 'coefficient')
    
    # Create data generator
    if generator_mode == "random":
        generator = DataGenerator.for_ml_training()
    else:
        generator = DataGenerator.for_pipeline()
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    with monitor_memory("legacy_data_generation", log_before_after=True):
        
        # Generate dataset in one go (suitable for small datasets)
        start_time = time.time()
        dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {n_samples} samples in {generation_time:.2f}s")
        
        # Extract field components
        E_theta = dataset['amplitude'][..., 0]  # Shape: (n_samples, n_phi, n_theta)
        E_phi = dataset['amplitude'][..., 1]
        
        # Handle complex fields
        if np.iscomplexobj(E_theta):
            E_theta = np.abs(E_theta).astype(np.float32)
            E_phi = np.abs(E_phi).astype(np.float32)
        else:
            E_theta = E_theta.astype(np.float32)
            E_phi = E_phi.astype(np.float32)
        
        # Always generate coefficient targets
        y_coeffs = pack_coefficients(dataset['coefficients_e'], dataset['coefficients_m']).astype(np.float32)
        
        # Generate P field targets if using physics loss
        # Transpose to (N, n_theta, n_phi) = (N, 179, 360) to match DifferentiableMultipoleField
        # output and the test-data file convention (179 theta rows × 360 phi columns).
        y_P = None
        if loss_type == "physics":
            logger.info("Computing P field targets...")
            y_P = (E_theta**2 + E_phi**2).transpose(0, 2, 1).astype(np.float32)
            logger.info(f"P field targets shape: {y_P.shape}  (n_samples, n_theta, n_phi)")
    
    data_info = {
        'generation_time': generation_time,
        'dataset_shape': E_theta.shape,
        'n_modes': y_coeffs.shape[1] // 4,
        'loss_type': loss_type,
        'has_p_field_targets': y_P is not None,
        'batches_used': 1,
        'batch_size': n_samples,
        'samples_per_second': n_samples / generation_time,
        'mode': 'legacy_memory'
    }
    
    return E_theta, E_phi, y_coeffs, y_P, data_info


def setup_preprocessing_batched_no_leakage(config: Dict[str, Any], E_theta: np.ndarray, E_phi: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> PreprocessingPipeline:
    """Set up and fit preprocessing pipeline on training data only (no data leakage)."""
    logger.info("Setting up preprocessing pipeline on training data only...")
    
    from src.api.preprocessing import PreprocessingConfig
    
    # Create preprocessing config
    preproc_config = PreprocessingConfig(
        pca_components=config['preprocessing']['pca_components'],
        pca_oversample=config['preprocessing']['pca_oversample'],
        normalize_features=config['preprocessing']['normalize_features'],
        normalize_targets=config['preprocessing']['normalize_targets']
    )
    
    # Create preprocessing pipeline
    preprocessing = PreprocessingPipeline(preproc_config)
    
    # Get training data
    E_theta_train = E_theta[train_idx]
    E_phi_train = E_phi[train_idx]
    y_train = y[train_idx]
    n_train = len(train_idx)
    
    # For large training sets, use a subset for PCA fitting to save memory and time
    if n_train > 5000:
        # Use random subset of training data for PCA fitting
        fit_samples = 5000
        logger.info(f"Using {fit_samples} random samples from TRAINING SET for PCA fitting")
        
        # Create random indices from training data only
        np.random.seed(42)  # Fixed seed for reproducibility
        fit_indices = np.random.choice(n_train, fit_samples, replace=False)
        
        E_theta_fit = E_theta_train[fit_indices]
        E_phi_fit = E_phi_train[fit_indices]
        y_fit = y_train[fit_indices]
        
        # Fit on training subset only
        preprocessing.fit(E_theta_fit, E_phi_fit, targets=y_fit)
        
        # Clear fitting data
        del E_theta_fit, E_phi_fit, y_fit
        
    else:
        # Small training set: fit on all training data
        logger.info(f"Using all {n_train} training samples for PCA fitting")
        preprocessing.fit(E_theta_train, E_phi_train, targets=y_train)
    
    # Clean up training data arrays
    del E_theta_train, E_phi_train, y_train
    
    logger.info(f"Preprocessing fitted on training data only: {preprocessing.get_stats()}")
    return preprocessing


def setup_preprocessing_batched(config: Dict[str, Any], E_theta: np.ndarray, E_phi: np.ndarray, y: np.ndarray) -> PreprocessingPipeline:
    """Set up and fit preprocessing pipeline with memory-efficient batching for large datasets."""
    logger.info("Setting up preprocessing pipeline...")
    
    from src.api.preprocessing import PreprocessingConfig
    
    # Create preprocessing config
    preproc_config = PreprocessingConfig(
        pca_components=config['preprocessing']['pca_components'],
        pca_oversample=config['preprocessing']['pca_oversample'],
        normalize_features=config['preprocessing']['normalize_features'],
        normalize_targets=config['preprocessing']['normalize_targets']
    )
    
    # Create preprocessing pipeline
    preprocessing = PreprocessingPipeline(preproc_config)
    
    # For large datasets, use a subset for PCA fitting to save memory and time
    n_samples = E_theta.shape[0] 
    if n_samples > 5000:
        # Use random subset for PCA fitting (still statistically valid)
        fit_samples = 5000
        logger.info(f"Using {fit_samples} random samples for PCA fitting (from {n_samples} total)")
        
        # Create random indices for fitting
        np.random.seed(42)  # Fixed seed for reproducibility
        fit_indices = np.random.choice(n_samples, fit_samples, replace=False)
        
        E_theta_fit = E_theta[fit_indices]
        E_phi_fit = E_phi[fit_indices]
        y_fit = y[fit_indices]
        
        # Fit on subset
        preprocessing.fit(E_theta_fit, E_phi_fit, targets=y_fit)
        
        # Clear fitting data
        del E_theta_fit, E_phi_fit, y_fit
        
    else:
        # Small dataset: fit on all data
        preprocessing.fit(E_theta, E_phi, targets=y)
    
    logger.info(f"Preprocessing fitted: {preprocessing.get_stats()}")
    return preprocessing


def transform_features_batched(preprocessing: PreprocessingPipeline, E_theta: np.ndarray, E_phi: np.ndarray, batch_size: int = 2000) -> np.ndarray:
    """Transform features in batches to avoid memory issues with large datasets."""
    n_samples = E_theta.shape[0]
    
    if n_samples <= batch_size:
        # Small dataset: process all at once  
        return preprocessing.transform_features(E_theta, E_phi)
    
    # Large dataset: process in batches
    logger.info(f"Transforming {n_samples} samples in batches of {batch_size}")
    
    X_list = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        if batch_idx % 5 == 0:  # Log every 5th batch
            logger.info(f"Transforming batch {batch_idx + 1}/{n_batches}")
        
        # Transform batch
        E_theta_batch = E_theta[start_idx:end_idx]
        E_phi_batch = E_phi[start_idx:end_idx]
        
        X_batch = preprocessing.transform_features(E_theta_batch, E_phi_batch)
        X_list.append(X_batch)
        
        # Clear batch data
        del E_theta_batch, E_phi_batch, X_batch
    
    # Concatenate results
    logger.info("Concatenating transformed batches...")
    X = np.concatenate(X_list, axis=0)
    del X_list
    
    logger.info(f"Feature transformation complete: {X.shape}")
    return X


def transform_targets_batched(preprocessing: PreprocessingPipeline, targets: np.ndarray, batch_size: int = 5000) -> np.ndarray:
    """Transform targets in batches to avoid memory issues with large datasets."""
    if targets.shape[0] <= batch_size:
        # Small dataset: process all at once
        return preprocessing.target_normalizer.transform(targets)
    
    # Large dataset: process in batches
    n_samples = targets.shape[0]
    logger.info(f"Transforming {n_samples} target samples in batches of {batch_size}")
    
    transformed_list = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        if batch_idx % 2 == 0:  # Log every 2nd batch
            logger.info(f"Transforming target batch {batch_idx + 1}/{n_batches}")
        
        targets_batch = targets[start_idx:end_idx]
        transformed_batch = preprocessing.target_normalizer.transform(targets_batch)
        transformed_list.append(transformed_batch)
        
        del targets_batch, transformed_batch
    
    # Concatenate results
    result = np.concatenate(transformed_list, axis=0)
    del transformed_list
    
    return result


def setup_preprocessing_streaming_no_leakage(
    config: Dict[str, Any], 
    E_theta_path: str, 
    E_phi_path: str, 
    y_coeffs_path: str,
    train_idx: np.ndarray
) -> PreprocessingPipeline:
    """Set up preprocessing pipeline fitting only on training data (no data leakage)."""
    from src.api.preprocessing import PreprocessingConfig
    from src.data.memory_monitor import monitor_memory
    
    preproc_config = PreprocessingConfig(
        pca_components=config['preprocessing']['pca_components'],
        pca_oversample=config['preprocessing']['pca_oversample'],
        normalize_features=config['preprocessing']['normalize_features'],
        normalize_targets=config['preprocessing']['normalize_targets']
    )
    
    with monitor_memory("streaming_preprocessing_setup"):
        
        # Load full data
        E_theta_mm = np.load(E_theta_path, mmap_mode='r')
        E_phi_mm = np.load(E_phi_path, mmap_mode='r')
        y_coeffs_mm = np.load(y_coeffs_path, mmap_mode='r')
        n_samples = len(E_theta_mm)
        n_train = len(train_idx)
        
        logger.info(f"Loaded memory-mapped data: {n_samples} samples, {n_train} for training")
        
        # Create preprocessing pipeline
        preprocessing = PreprocessingPipeline(preproc_config)
        
        # Use subset of training data for PCA fitting to save memory and time
        if n_train > 5000:
            fit_samples = 5000
            logger.info(f"Using {fit_samples} random samples from TRAINING SET for PCA fitting")
            
            # Create random indices from training indices only
            np.random.seed(42)  # Fixed seed for reproducibility
            fit_indices = np.random.choice(train_idx, fit_samples, replace=False)
        else:
            fit_indices = train_idx
            logger.info(f"Using all {n_train} training samples for PCA fitting")
            
        # Use fancy indexing to get training subset (this will copy data to memory)
        E_theta_fit = E_theta_mm[fit_indices]
        E_phi_fit = E_phi_mm[fit_indices]
        y_fit = y_coeffs_mm[fit_indices]
        
        # Fit on training subset only
        preprocessing.fit(E_theta_fit, E_phi_fit, targets=y_fit)
        
        # Clear fitting data
        del E_theta_fit, E_phi_fit, y_fit
    
    logger.info(f"Streaming preprocessing fitted on training data only: {preprocessing.get_stats()}")
    return preprocessing


def setup_preprocessing_streaming(
    config: Dict[str, Any], 
    E_theta_path: str, 
    E_phi_path: str, 
    y_coeffs_path: str
) -> PreprocessingPipeline:
    """
    Set up and fit preprocessing pipeline for streaming data using memory-mapped files.
    
    Args:
        config: Configuration dictionary
        E_theta_path: Path to E_theta memory-mapped file
        E_phi_path: Path to E_phi memory-mapped file  
        y_coeffs_path: Path to y_coeffs memory-mapped file
        
    Returns:
        Fitted PreprocessingPipeline
    """
    from src.api.preprocessing import PreprocessingConfig
    from src.data.memory_monitor import monitor_memory, get_memory_usage
    
    logger.info("Setting up streaming preprocessing pipeline...")
    
    # Create preprocessing config
    preproc_config = PreprocessingConfig(
        pca_components=config['preprocessing']['pca_components'],
        pca_oversample=config['preprocessing']['pca_oversample'],
        normalize_features=config['preprocessing']['normalize_features'],
        normalize_targets=config['preprocessing']['normalize_targets']
    )
    
    with monitor_memory("streaming_preprocessing_setup", log_before_after=True):
        
        # Load memory-mapped arrays
        E_theta_mm = np.load(E_theta_path, mmap_mode='r')
        E_phi_mm = np.load(E_phi_path, mmap_mode='r') 
        y_coeffs_mm = np.load(y_coeffs_path, mmap_mode='r')
        
        n_samples = E_theta_mm.shape[0]
        logger.info(f"Loaded memory-mapped data: {n_samples} samples")
        
        # Create preprocessing pipeline
        preprocessing = PreprocessingPipeline(preproc_config)
        
        # Use subset for PCA fitting to save memory and time
        if n_samples > 5000:
            fit_samples = 5000
            logger.info(f"Using {fit_samples} random samples for PCA fitting (from {n_samples} total)")
            
            # Create random indices for fitting
            np.random.seed(42)  # Fixed seed for reproducibility
            fit_indices = np.random.choice(n_samples, fit_samples, replace=False)
            
            # Use fancy indexing to get subset (this will copy data to memory)
            E_theta_fit = E_theta_mm[fit_indices]
            E_phi_fit = E_phi_mm[fit_indices]
            y_fit = y_coeffs_mm[fit_indices]
            
            # Fit on subset
            preprocessing.fit(E_theta_fit, E_phi_fit, targets=y_fit)
            
            # Clear fitting data
            del E_theta_fit, E_phi_fit, y_fit
            
        else:
            # Small dataset: fit on all data (will load into memory)
            logger.info("Loading full dataset for preprocessing fit (small dataset)")
            E_theta_array = np.array(E_theta_mm)
            E_phi_array = np.array(E_phi_mm)
            y_coeffs_array = np.array(y_coeffs_mm)
            
            preprocessing.fit(E_theta_array, E_phi_array, targets=y_coeffs_array)
            
            del E_theta_array, E_phi_array, y_coeffs_array
    
    logger.info(f"Streaming preprocessing fitted: {preprocessing.get_stats()}")
    return preprocessing


def transform_features_streaming(
    preprocessing: PreprocessingPipeline,
    E_theta_path: str,
    E_phi_path: str,
    output_path: str,
    batch_size: int = 2000
) -> str:
    """
    DEPRECATED: Unused duplicate function. Use PreprocessingPipeline.transform_features_streaming() instead.
    Transform features in streaming mode, writing results to memory-mapped file.
    
    Args:
        preprocessing: Fitted preprocessing pipeline
        E_theta_path: Path to E_theta memory-mapped file
        E_phi_path: Path to E_phi memory-mapped file
        output_path: Path for output transformed features
        batch_size: Batch size for processing
        
    Returns:
        Path to transformed features file
    """
    from src.data.memory_monitor import monitor_memory, get_memory_usage
    
    logger.info(f"Transforming features in streaming mode...")
    
    with monitor_memory("streaming_feature_transform", log_before_after=True):
        
        # Load memory-mapped input arrays
        E_theta_mm = np.load(E_theta_path, mmap_mode='r')
        E_phi_mm = np.load(E_phi_path, mmap_mode='r')
        n_samples = E_theta_mm.shape[0]
        
        # Determine output shape by processing a small batch
        probe_batch_size = min(10, n_samples)
        E_theta_probe = E_theta_mm[:probe_batch_size]
        E_phi_probe = E_phi_mm[:probe_batch_size]
        X_probe = preprocessing.transform_features(E_theta_probe, E_phi_probe)
        n_features = X_probe.shape[1]
        del E_theta_probe, E_phi_probe, X_probe
        
        # Create output memory-mapped array
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        X_mm = np.lib.format.open_memmap(
            str(output_path), mode='w+', dtype=np.float32,
            shape=(n_samples, n_features)
        )
        
        logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
        logger.info(f"Output shape: ({n_samples}, {n_features})")
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            if batch_idx % 5 == 0:
                current_memory = get_memory_usage()
                logger.info(f"  Processing batch {batch_idx + 1}/{n_batches} (Memory: {current_memory:.1f}MB)")
            
            # Load batch (copies to memory)
            E_theta_batch = E_theta_mm[start_idx:end_idx]
            E_phi_batch = E_phi_mm[start_idx:end_idx]
            
            # Transform batch
            X_batch = preprocessing.transform_features(E_theta_batch, E_phi_batch)
            
            # Write to memory-mapped output
            X_mm[start_idx:end_idx] = X_batch.astype(np.float32)
            
            # Clear batch data
            del E_theta_batch, E_phi_batch, X_batch
            
            # Periodic garbage collection
            if batch_idx % 5 == 0:
                import gc
                gc.collect()
        
        # Flush output to disk
        X_mm.flush()
        
        logger.info(f"✅ Feature transformation complete: {output_path}")
    
    return str(output_path)


def split_indices(n_samples: int, config: Dict[str, Any], seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/val/test sets to avoid data leakage."""
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    
    # Calculate split sizes
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    # Create indices and shuffle with explicit seed for reproducibility
    indices = np.arange(n_samples)
    if seed is not None:
        # Use separate random state to avoid interfering with global RNG
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    
    # Split indices
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    logger.info(f"Index split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return train_idx, val_idx, test_idx


def split_data(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    """Split data into train/val/test sets."""
    n_samples = len(X)
    train_idx, val_idx, test_idx = split_indices(n_samples, config)
    
    # Split data
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model_from_config(config: Dict[str, Any], input_dim: int, output_dim: int) -> Any:
    """Create MLP model from configuration with actual data dimensions."""
    model_config = config['model']
    training_config = config['training']
    
    # Create model with combined configuration and actual dimensions
    registry = get_model_registry()
    
    # Prepare model parameters
    model_params = {
        "model_type": "mlp",
        "input_dim": input_dim,  # Use actual input dimension from preprocessing
        "output_dim": output_dim,  # Use actual output dimension from targets
        "hidden_size": model_config['hidden_size'],
        "n_hidden_layers": model_config['n_hidden_layers'],
        "dropout_rate": model_config['dropout_rate'],
        "activation": model_config['activation'],
        "learning_rate": training_config['learning_rate'],
        "epochs": training_config['epochs'],
        "batch_size": training_config['batch_size'],
        "device": config['device'],
        # Physics loss parameters
        "maxorder": model_config['maxorder'],
        "grid_n_theta": model_config.get('grid_n_theta', 179),
        "grid_n_phi": model_config.get('grid_n_phi', 360),
        "loss_type": model_config.get('loss_type', 'coefficient'),
        "physics_grid_type": model_config.get('physics_grid_type', 'equiangular'),
        "physics_grid_resolution_factor": model_config.get('physics_grid_resolution_factor', 1.0),
        "physics_field_weight": model_config.get('physics_field_weight', 0.1)
    }
    
    model = registry.create_model(**model_params)
    
    logger.info(f"Created MLP model: {model.get_model_info()}")
    return model


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    experiment_dir: Path
) -> Dict[str, Any]:
    """Train the MLP model."""
    logger.info("Starting model training...")
    
    # Train model
    start_time = time.time()
    training_result = model.fit(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f}s")
    logger.info(f"Final train loss: {training_result.get('final_train_loss', 'N/A')}")
    logger.info(f"Final val loss: {training_result.get('final_val_loss', 'N/A')}")
    
    # Save model
    model_save_path = experiment_dir / "model"
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return training_result


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessing: PreprocessingPipeline,
    config: Dict[str, Any],
    experiment_dir: Path
) -> tuple[Dict[str, float], np.ndarray]:
    """Evaluate trained model on test set."""
    logger.info("Evaluating model on test set...")
    
    # Make predictions (model always outputs coefficients) using batched prediction
    logger.info(f"Making predictions on {X_test.shape[0]} test samples...")
    if hasattr(model, 'predict_safe'):
        predictions = model.predict_safe(X_test, force_batch=True)
    else:
        # Fallback for models without predict_safe
        if hasattr(model, 'predict_batch'):
            predictions = model.predict_batch(X_test)
        else:
            predictions = model.predict(X_test)
    
    # Check if this is a physics-trained model
    loss_type = config['model'].get('loss_type', 'coefficient')
    
    if loss_type == "physics":
        # For physics models, convert predictions to P field and compare with P field targets
        logger.info("Physics model evaluation: converting coefficient predictions to P fields")
        
        try:
            from src.models.physics_layers import DifferentiableMultipoleField
            import torch
            
            # Create field generator with same parameters as training
            maxorder = config['model']['maxorder']
            grid_n_phi = config['model'].get('grid_n_phi', 360)
            grid_n_theta = config['model'].get('grid_n_theta', 179)
            
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder, 
                grid_shape=(grid_n_phi, grid_n_theta)
            )
            
            # Convert predictions to P fields; shapes now match (N, n_theta, n_phi)
            pred_coeffs_tensor = torch.from_numpy(predictions).float()
            pred_P = field_gen(pred_coeffs_tensor).detach().numpy()
            
            # Compute metrics on P fields
            test_mse = float(np.mean((y_test - pred_P) ** 2))
            test_mae = float(np.mean(np.abs(y_test - pred_P)))
            test_rmse = float(np.sqrt(test_mse))
            
            # Compute R² score on P fields
            ss_res = np.sum((y_test - pred_P) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
            
            logger.info(f"Physics evaluation: P field MSE = {test_mse:.6f}, R² = {r2_score:.6f}")
            
        except Exception as e:
            logger.warning(f"Failed to compute P field metrics: {e}")
            logger.info("Falling back to coefficient comparison (may not be meaningful)")
            # Fallback: just compute some metrics on coefficients vs flattened y_test
            y_test_flat = y_test.reshape(y_test.shape[0], -1)[:, :predictions.shape[1]]
            test_mse = float(np.mean((y_test_flat - predictions) ** 2))
            test_mae = float(np.mean(np.abs(y_test_flat - predictions)))
            test_rmse = float(np.sqrt(test_mse))
            r2_score = 0.0  # Not meaningful
    else:
        # Standard coefficient evaluation
        logger.info("Coefficient model evaluation")
        test_mse = float(np.mean((y_test - predictions) ** 2))
        test_mae = float(np.mean(np.abs(y_test - predictions)))
        test_rmse = float(np.sqrt(test_mse))
        
        # Compute R² score
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    metrics = {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': r2_score
    }
    
    logger.info(f"Test metrics: MSE={test_mse:.6f}, MAE={test_mae:.6f}, R²={r2_score:.4f}")
    
    # Generate evaluation plots
    logger.info("Generating evaluation plots...")
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots based on model type
    loss_type = config['model'].get('loss_type', 'coefficient')
    maxorder = config['model']['maxorder']
    
    try:
        if loss_type == "physics":
            logger.info("Generating physics-informed model plots...")
            
            # Import physics layers for P field conversion
            from src.models.physics_layers import DifferentiableMultipoleField
            import torch
            
            # Create field generator matching training setup
            grid_n_phi = config['model'].get('grid_n_phi', 360)
            grid_n_theta = config['model'].get('grid_n_theta', 179)
            
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder,
                grid_shape=(grid_n_phi, grid_n_theta)
            )
            
            # Convert coefficient predictions to P fields
            pred_coeffs_tensor = torch.from_numpy(predictions).float()
            pred_P = field_gen(pred_coeffs_tensor).detach().numpy()
            
            # Handle potential shape mismatch
            if pred_P.shape != y_test.shape:
                if pred_P.shape == (y_test.shape[0], y_test.shape[2], y_test.shape[1]):
                    pred_P = pred_P.transpose(0, 2, 1)
                    logger.info("Transposed predicted P field to match target shape")
            
            # 1. P Field scatter plot (key physics evaluation)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Sample points for visualization (handle different array sizes)
            P_true_flat = y_test.flatten()
            P_pred_flat = pred_P.flatten()
            
            # Use the smaller array size for sampling
            min_length = min(len(P_true_flat), len(P_pred_flat))
            n_sample_points = min(5000, min_length)
            
            # Handle different array sizes properly
            if len(P_true_flat) != len(P_pred_flat):
                logger.info(f"Shape mismatch: true={len(P_true_flat)}, pred={len(P_pred_flat)}")
                
                # Crop both arrays to the same size for meaningful comparison
                min_length = min(len(P_true_flat), len(P_pred_flat))
                n_sample_points = min(n_sample_points, min_length)
                
                # Sample CORRESPONDING positions from both arrays (use fixed seed for reproducibility)
                np.random.seed(42)  # Fixed seed for consistent plot sampling
                indices = np.random.choice(min_length, n_sample_points, replace=False)
                sampled_true = P_true_flat[:min_length][indices]
                sampled_pred = P_pred_flat[:min_length][indices]
                
                logger.info(f"Using {n_sample_points} corresponding samples for scatter plot")
            else:
                # Same size - sample with matching indices (use fixed seed for reproducibility)
                np.random.seed(42)  # Fixed seed for consistent plot sampling
                indices = np.random.choice(len(P_true_flat), n_sample_points, replace=False)
                sampled_true = P_true_flat[indices]  
                sampled_pred = P_pred_flat[indices]
            
            # Debug: Log actual values being plotted
            logger.info(f"P field scatter plot values:")
            logger.info(f"  True P:  mean={sampled_true.mean():.6f}, std={sampled_true.std():.6f}, range=[{sampled_true.min():.6f}, {sampled_true.max():.6f}]")
            logger.info(f"  Pred P:  mean={sampled_pred.mean():.6f}, std={sampled_pred.std():.6f}, range=[{sampled_pred.min():.6f}, {sampled_pred.max():.6f}]")
            
            ax.scatter(sampled_true, sampled_pred, alpha=0.4, s=2)
            
            # Perfect prediction line (use sampled data for consistent scaling)
            min_val = min(sampled_true.min(), sampled_pred.min())
            max_val = max(sampled_true.max(), sampled_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax.set_xlabel('True P Field Values')
            ax.set_ylabel('Predicted P Field Values')
            ax.set_title('Physics Model: P Field Predictions vs True Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add R² score (use sampled data for consistent calculation)
            ss_res = np.sum((sampled_true - sampled_pred) ** 2)
            ss_tot = np.sum((sampled_true - np.mean(sampled_true)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            ax.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "p_field_predictions_scatter.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Per-sample 2D P-field heatmap comparisons (test split)
            n_sample_plots = config.get('plotting', {}).get('n_sample_plots', 2)
            n_test_total = len(y_test)
            for i in range(min(n_sample_plots, n_test_total)):
                try:
                    plot_p_field_comparison(
                        y_test, pred_P, sample_idx=i,
                        title_prefix=f"Test (synthetic) | sample {i+1}/{n_test_total}",
                        save_path=plots_dir / f"p_field_comparison_test_sample_{i}.png"
                    )
                    logger.info(f"✅ P-field heatmap saved: test sample {i+1}/{n_test_total}")
                except Exception as e:
                    logger.warning(f"Failed to generate P-field heatmap for test sample {i}: {e}")
            
            # Generate per-sample P-field scatter summary (2x2 grid)
            try:
                logger.info("Generating multi-sample P-field summary plot...")
                n_summary_samples = min(4, len(predictions))
                
                if n_summary_samples >= 4:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('Physics Model: Multi-Sample P Field Comparison', fontsize=16)
                    
                    for idx in range(4):
                        row, col = idx // 2, idx % 2
                        ax = axes[row, col]
                        
                        # Predicted P field from coefficient predictions
                        pred_coeffs_sample = predictions[idx:idx+1]
                        pred_P_sample = field_gen(torch.from_numpy(pred_coeffs_sample).float()).detach().numpy()
                        
                        # True P field from test targets (actual aligned data)
                        true_P_sample = y_test[idx:idx+1]
                        
                        # Align shapes if needed
                        if pred_P_sample.shape != true_P_sample.shape:
                            if pred_P_sample.shape == (true_P_sample.shape[0], true_P_sample.shape[2], true_P_sample.shape[1]):
                                pred_P_sample = pred_P_sample.transpose(0, 2, 1)
                        
                        true_flat = true_P_sample.flatten()[:1000]
                        pred_flat = pred_P_sample.flatten()[:1000]
                        
                        ax.scatter(true_flat, pred_flat, alpha=0.5, s=1)
                        
                        min_val = min(true_flat.min(), pred_flat.min())
                        max_val = max(true_flat.max(), pred_flat.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                        
                        ss_res = np.sum((true_flat - pred_flat) ** 2)
                        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        
                        ax.set_title(f'Sample {idx+1} (R²={r2:.3f})')
                        ax.set_xlabel('True P Field')
                        ax.set_ylabel('Predicted P Field')
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / "multi_sample_summary.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info("✅ Multi-sample summary plot saved")
                
            except Exception as e:
                logger.warning(f"Failed to generate multi-sample summary plot: {e}")
            
            logger.info(f"Physics model plots saved to {plots_dir}")
            
        else:
            # Standard coefficient model plots
            logger.info("Generating coefficient model plots...")
            
            # Standard prediction scatter plot
            plot_prediction_scatter(
                y_test, predictions,
                title="Test Set: Predictions vs True Values",
                save_path=plots_dir / "predictions_scatter.png"
            )
            
            # Coefficient comparison for first few samples
            for i in range(min(3, len(X_test))):
                plot_coefficient_comparison(
                    y_test, predictions, maxorder, sample_idx=i,
                    save_path=plots_dir / f"coefficients_sample_{i}.png"
                )
            
            # Field comparison plots for test samples
            n_sample_plots = config.get('plotting', {}).get('n_sample_plots', 2)
            plot_test_samples = config.get('plotting', {}).get('plot_test_samples', True)
            
            if plot_test_samples:
                n_samples_to_plot = min(n_sample_plots, len(X_test))
                logger.info(f"Generating {n_samples_to_plot} comprehensive field comparison plots...")
                for i in range(n_samples_to_plot):
                    sample_title = f"Sample {i+1} [test]"
                    plot_field_comparison(
                        y_test, predictions, sample_idx=i, maxorder=maxorder,
                        title_prefix=sample_title,
                        save_path=plots_dir / f"field_comparison_test_sample_{i}.png"
                    )
            
            logger.info(f"Coefficient model plots saved to {plots_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to generate evaluation plots: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics, predictions


def setup_mlflow_tracking(config: Dict[str, Any], experiment_dir: Path):
    """Set up MLflow experiment tracking."""
    experiment_name = config['experiment']['name']
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start run with tags
    tags = config['experiment'].get('tags', [])
    mlflow_tags = {f"tag_{i}": tag for i, tag in enumerate(tags)}
    mlflow_tags.update({
        "experiment_dir": str(experiment_dir),
        "description": config['experiment']['description']
    })
    
    return mlflow.start_run(tags=mlflow_tags)


def log_to_mlflow(config: Dict[str, Any], training_result: Dict[str, Any], 
                  test_metrics: Dict[str, float], data_info: Dict[str, Any]):
    """Log all results to MLflow."""
    # Log parameters
    mlflow.log_params({
        "maxorder": config['model']['maxorder'],
        "hidden_size": config['model']['hidden_size'],
        "n_hidden_layers": config['model']['n_hidden_layers'],
        "dropout_rate": config['model']['dropout_rate'],
        "learning_rate": config['training']['learning_rate'],
        "epochs": config['training']['epochs'],
        "batch_size": config['training']['batch_size'],
        "n_samples": config['training']['n_samples'],
        "pca_components": config['preprocessing']['pca_components'],
        "device": config['device']
    })
    
    # Log metrics
    mlflow.log_metrics({
        "final_train_loss": training_result.get('final_train_loss', 0),
        "final_val_loss": training_result.get('final_val_loss', 0),
        "training_time": training_result.get('training_time', 0),
        **test_metrics
    })
    
    # Log data info
    mlflow.log_metrics({
        "generation_time": data_info['generation_time'],
        "n_modes": data_info['n_modes']
    })


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MLP model for multipole analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--maxorder", type=int, help="Override maxorder from config")
    parser.add_argument("--hidden_size", type=int, help="Override model hidden_size from config")
    parser.add_argument("--device", type=str, help="Override device from config")
    parser.add_argument("--resume", type=str, help="Resume from experiment directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config(Path(args.config))
        
        # Apply command line overrides
        if args.maxorder:
            config['model']['maxorder'] = args.maxorder
        if args.hidden_size:
            config['model']['hidden_size'] = args.hidden_size
        if args.device:
            config['device'] = args.device
        
        # Set global random seeds for reproducibility
        global_seed = config['data'].get('seed', config.get('seed', 42))
        set_random_seeds(global_seed)
        
        # Print environment info
        print("\n" + "="*60)
        print("EXPERIMENT ENVIRONMENT")
        print("="*60)
        print_environment_info()
        
        # Validate ML environment
        validate_ml_environment(require_torch=True)
        
        # Set up experiment directory
        experiment_dir = setup_experiment_directory(config)
        
        # Set up enhanced MLFlow tracking
        experiment_name = config['experiment']['name']
        run_name = f"{experiment_name}_{int(time.time())}"
        
        # Create experiment manager
        experiment_manager = create_experiment_manager(experiment_name, config)
        
        if experiment_manager.start_experiment(run_name=run_name, tags=config['experiment'].get('tags', [])):
            
            # Generate training data
            E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
            
            # Log data generation info
            experiment_manager.tracker.log_metrics({
                "data_generation_time": data_info['generation_time'],
                "n_modes": data_info['n_modes']
            })
            
            # STEP 1: Determine dataset size and split indices BEFORE preprocessing to avoid data leakage
            using_streaming = isinstance(E_theta, str)
            
            if using_streaming:
                # Get dataset size from memory-mapped files
                n_samples = np.load(E_theta, mmap_mode='r').shape[0]
            else:
                # Get dataset size from arrays
                n_samples = E_theta.shape[0]
            
            # Split indices BEFORE any preprocessing (prevents data leakage)
            data_seed = config['data'].get('seed', config.get('seed', 42))
            train_idx, val_idx, test_idx = split_indices(n_samples, config, seed=data_seed)
            logger.info("✅ Data indices split BEFORE preprocessing to prevent data leakage")
            
            # STEP 2: Setup preprocessing using ONLY training data
            if using_streaming:
                logger.info("Using streaming preprocessing pipeline (leak-free)...")
                preprocessing = setup_preprocessing_streaming_no_leakage(
                    config, E_theta, E_phi, y_coeffs, train_idx
                )
                
                # Transform features with streaming
                logger.info("Transforming features in streaming mode...")
                cache_dir = Path(data_info['cache_dir'])
                X_path = cache_dir / "X_transformed.npy"
                preprocessing.transform_features_streaming(E_theta, E_phi, str(X_path))
                
                # Load transformed features
                X = np.load(X_path, mmap_mode='r')
                logger.info(f"Loaded transformed features: {X.shape}")
                
                # Select targets based on loss type and load from files
                loss_type = config['model'].get('loss_type', 'coefficient')
                if loss_type == "physics":
                    if y_P is None:
                        raise ValueError("Physics loss requested but no P field targets generated")
                    y_targets = np.load(y_P, mmap_mode='r')
                    logger.info(f"Using P field targets for physics loss: {y_targets.shape}")
                else:
                    y_targets = np.load(y_coeffs, mmap_mode='r')
                    logger.info(f"Using coefficient targets: {y_targets.shape}")
            else:
                logger.info("Using in-memory preprocessing pipeline (leak-free)...")
                preprocessing = setup_preprocessing_batched_no_leakage(
                    config, E_theta, E_phi, y_coeffs, train_idx
                )
                
                # Transform features (with batching for large datasets)
                logger.info("Transforming features...")
                X = transform_features_batched(preprocessing, E_theta, E_phi)
                
                # Select targets based on loss type
                loss_type = config['model'].get('loss_type', 'coefficient')
                if loss_type == "physics":
                    if y_P is None:
                        raise ValueError("Physics loss requested but no P field targets generated")
                    y_targets = y_P
                    logger.info("Using P field targets for physics loss")
                else:
                    y_targets = y_coeffs
                    logger.info("Using coefficient targets for MSE loss")
            
            # STEP 3: Split transformed data using predetermined indices
            logger.info("Splitting transformed data using predetermined indices...")
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train, y_val, y_test = y_targets[train_idx], y_targets[val_idx], y_targets[test_idx]
            logger.info(f"Final data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            # Normalize targets if configured (only for coefficient targets)
            if config['preprocessing']['normalize_targets'] and loss_type == "coefficient":
                logger.info("Normalizing coefficient targets...")
                # Apply target normalization directly to the split data in batches
                # The target normalizer was already fitted on the packed coefficients
                if preprocessing.target_normalizer is not None and preprocessing.target_normalizer.is_fitted:
                    y_train = transform_targets_batched(preprocessing, y_train)
                    y_val = transform_targets_batched(preprocessing, y_val)
                    y_test = transform_targets_batched(preprocessing, y_test)
                    logger.info("✅ Coefficient targets normalized using fitted pipeline")
                else:
                    logger.warning("Target normalizer not fitted, skipping normalization")
            elif loss_type == "physics":
                logger.info("Skipping target normalization for P field targets (physics loss)")
            else:
                logger.info("Target normalization disabled")
            
            # Create model with actual dimensions from processed data
            input_dim = X.shape[1]  # Actual feature dimension after preprocessing
            
            # For physics loss, model always predicts coefficients (not P field)
            if loss_type == "physics":
                # Use data_info for output dimension when streaming
                if using_streaming:
                    output_dim = data_info['coeffs_shape'][0]  # Model outputs coefficients
                else:
                    output_dim = y_coeffs.shape[1]
                logger.info(f"Physics loss: model outputs {output_dim} coefficients, trained on P fields")
            else:
                # For coefficient targets, use the target dimension
                if using_streaming:
                    output_dim = data_info['coeffs_shape'][0]  # Coefficients
                else:
                    output_dim = y_targets.shape[1]
                logger.info(f"Coefficient loss: model outputs {output_dim} coefficients")
            
            model = create_model_from_config(config, input_dim, output_dim)
            
            # Train model
            training_result = train_model(
                model, X_train, y_train, X_val, y_val, config, experiment_dir
            )
            
            # Set up plots directory before it is used by validation and training plots
            plots_dir = experiment_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Generate validation plots if requested
            plot_validation_samples = config.get('plotting', {}).get('plot_validation_samples', False)
            if plot_validation_samples and X_val is not None and y_val is not None:
                logger.info("Generating validation sample plots...")
                try:
                    # Make predictions on validation set
                    if hasattr(model, 'predict_safe'):
                        val_predictions = model.predict_safe(X_val, force_batch=True)
                    else:
                        val_predictions = model.predict(X_val)
                    
                    # Generate validation plots
                    val_plots_dir = plots_dir / "validation_samples"
                    val_plots_dir.mkdir(exist_ok=True)
                    
                    n_sample_plots = config.get('plotting', {}).get('n_sample_plots', 2)
                    n_val_samples = min(n_sample_plots, len(X_val))
                    
                    loss_type = config['model'].get('loss_type', 'coefficient')
                    maxorder = config['model']['maxorder']
                    
                    if loss_type == "physics":
                        # Convert predicted coefficients -> predicted P field for visualisation
                        from src.models.physics_layers import DifferentiableMultipoleField
                        import torch
                        grid_n_phi = config['model'].get('grid_n_phi', 360)
                        grid_n_theta = config['model'].get('grid_n_theta', 179)
                        field_gen_val = DifferentiableMultipoleField(
                            maxorder=maxorder,
                            grid_shape=(grid_n_phi, grid_n_theta)
                        )
                        # pred_P_val shape: (N, n_theta, n_phi) matches y_val after grid alignment fix
                        pred_P_val = field_gen_val(
                            torch.from_numpy(val_predictions).float()
                        ).detach().numpy()

                        n_val_total = len(y_val)
                        for i in range(n_val_samples):
                            plot_p_field_comparison(
                                y_val, pred_P_val, sample_idx=i,
                                title_prefix=f"Validation | sample {i+1}/{n_val_total}",
                                save_path=val_plots_dir / f"p_field_validation_sample_{i}.png"
                            )
                    else:
                        # For coefficient models
                        for i in range(n_val_samples):
                            plot_field_comparison(
                                y_val, val_predictions, sample_idx=i, maxorder=maxorder,
                                title_prefix=f"Sample {i+1} [validation]",
                                save_path=val_plots_dir / f"validation_sample_{i}.png"
                            )
                    
                    logger.info(f"✅ Generated {n_val_samples} validation sample plots in {val_plots_dir}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate validation plots: {e}")
            
            # Training-sample P-field heatmap comparison (physics models only)
            if config['model'].get('loss_type', 'coefficient') == "physics":
                try:
                    logger.info("Generating P-field comparison plots for training samples...")
                    from src.models.physics_layers import DifferentiableMultipoleField
                    import torch
                    _maxorder  = config['model']['maxorder']
                    _grid_n_phi   = config['model'].get('grid_n_phi', 360)
                    _grid_n_theta = config['model'].get('grid_n_theta', 179)
                    _field_gen_tr = DifferentiableMultipoleField(
                        maxorder=_maxorder,
                        grid_shape=(_grid_n_phi, _grid_n_theta)
                    )
                    _n_tr_plots = config.get('plotting', {}).get('n_sample_plots', 2)
                    _n_tr_plots = min(_n_tr_plots, len(X_train))
                    if hasattr(model, 'predict_safe'):
                        _tr_preds = model.predict_safe(X_train[:_n_tr_plots], force_batch=True)
                    else:
                        _tr_preds = model.predict(X_train[:_n_tr_plots])
                    # pred_P_tr shape: (N, n_theta, n_phi) matches y_train after grid alignment fix
                    _pred_P_tr = _field_gen_tr(
                        torch.from_numpy(_tr_preds).float()
                    ).detach().numpy()
                    _n_train_total = len(y_train)
                    for i in range(_n_tr_plots):
                        plot_p_field_comparison(
                            y_train[:_n_tr_plots], _pred_P_tr, sample_idx=i,
                            title_prefix=f"Train | sample {i+1}/{_n_train_total}",
                            save_path=plots_dir / f"p_field_train_sample_{i}.png"
                        )
                    logger.info(f"✅ Generated {_n_tr_plots} training P-field heatmap plots")
                except Exception as e:
                    logger.warning(f"Failed to generate training P-field plots: {e}")

            # Evaluate model
            test_metrics, predictions = evaluate_model(model, X_test, y_test, preprocessing, config, experiment_dir)
            
            # Log test metrics to experiment manager
            experiment_manager.log_model_performance(test_metrics)
            
            # Generate training plots
            logger.info("Generating training summary plots...")
            
            try:
                # Training curves plot (if training history is available)
                if 'training_history' in training_result:
                    # Convert training history from list-of-dicts to dict-of-lists
                    history_list = training_result['training_history']
                    if isinstance(history_list, list) and len(history_list) > 0:
                        # Convert: [{"epoch": 1, "train_loss": 0.5}] -> {"train_loss": [0.5], ...}
                        keys = history_list[0].keys()
                        history_dict = {key: [entry[key] for entry in history_list] for key in keys}
                        
                        plot_training_curves(
                            history_dict,
                            save_path=plots_dir / "training_curves.png"
                        )
                        logger.info("✅ Training curves plot generated")
                    else:
                        logger.warning("Training history is empty or invalid format")
                
                # Comprehensive experiment summary
                # Also convert history format for summary plot
                history_for_summary = {}
                if 'training_history' in training_result:
                    history_list = training_result.get('training_history', [])
                    if isinstance(history_list, list) and len(history_list) > 0:
                        keys = history_list[0].keys()
                        history_for_summary = {key: [entry[key] for entry in history_list] for key in keys}
                
                create_experiment_summary_plot(
                    config, 
                    history_for_summary,
                    test_metrics,
                    save_path=plots_dir / "experiment_summary.png"
                )
                
                logger.info(f"Training plots saved to {plots_dir}")
                
                # Log plots to MLFlow (including validation plots if they exist)
                experiment_manager.log_plots(plots_dir)
                
                # Also log validation plots if they exist
                val_plots_dir = plots_dir / "validation_samples"
                if val_plots_dir.exists() and any(val_plots_dir.iterdir()):
                    logger.info("Logging validation plots to MLFlow...")
                    experiment_manager.log_plots(val_plots_dir)
                
            except Exception as e:
                logger.warning(f"Failed to generate training plots: {e}")
            
            # Inline holdout (E_in_plane) evaluation — physics models only
            if config['model'].get('loss_type', 'coefficient') == "physics":
                try:
                    from src.core.config import Config as CoreConfig
                    from src.core.dataset_loader import TestDatasetLoader
                    _hc = CoreConfig()
                    _feat_dir = _hc.paths.test_features_dir
                    _tgt_dir  = _hc.paths.test_targets_dir
                    import os as _os
                    if _os.path.exists(_feat_dir) and _os.path.exists(_tgt_dir):
                        logger.info("Running inline holdout evaluation on E_in_plane test data...")
                        _loader = TestDatasetLoader(
                            features_dir=_feat_dir,
                            targets_dir=_tgt_dir
                        )
                        _maxorder = config['model']['maxorder']
                        _n_holdout = config.get('plotting', {}).get('n_holdout_plots', 3)
                        _E_theta_h, _E_phi_h, _a_e_h, _a_m_h = _loader.load_dataset(
                            maxorder=_maxorder, limit=_n_holdout
                        )
                        if len(_E_theta_h) > 0:
                            _n_h = _E_theta_h.shape[0]
                            _grid_n_phi   = config['model'].get('grid_n_phi', 360)
                            _grid_n_theta = config['model'].get('grid_n_theta', 179)
                            _exp_size = _grid_n_phi * _grid_n_theta
                            if _E_theta_h.shape[1] == _exp_size:
                                _E_theta_h3 = _E_theta_h.reshape(_n_h, _grid_n_phi, _grid_n_theta)
                                _E_phi_h3   = _E_phi_h.reshape(_n_h, _grid_n_phi, _grid_n_theta)
                            else:
                                _E_theta_h3 = _E_theta_h.reshape(_n_h, _E_theta_h.shape[1] // _grid_n_theta, _grid_n_theta)
                                _E_phi_h3   = _E_phi_h.reshape(_n_h, _E_phi_h.shape[1] // _grid_n_theta, _grid_n_theta)

                            # TestDatasetLoader returns complex fields; PCA was fitted on real magnitudes
                            _E_theta_h3_real = np.abs(_E_theta_h3).astype(np.float32)
                            _E_phi_h3_real   = np.abs(_E_phi_h3).astype(np.float32)

                            _X_h = preprocessing.transform_features(_E_theta_h3_real, _E_phi_h3_real)

                            if hasattr(model, 'predict_safe'):
                                _h_preds = model.predict_safe(_X_h, force_batch=True)
                            else:
                                _h_preds = model.predict(_X_h)

                            from src.models.physics_layers import DifferentiableMultipoleField
                            import torch
                            _h_field_gen = DifferentiableMultipoleField(
                                maxorder=_maxorder,
                                grid_shape=(_grid_n_phi, _grid_n_theta)
                            )
                            _P_pred_h = _h_field_gen(
                                torch.from_numpy(_h_preds).float()
                            ).detach().numpy()  # (N, n_theta, n_phi)

                            # True P = |E_theta|^2 + |E_phi|^2; transpose (N,n_phi,n_theta)->(N,n_theta,n_phi)
                            _P_true_h = (
                                _E_theta_h3_real ** 2 + _E_phi_h3_real ** 2
                            ).transpose(0, 2, 1).astype(np.float32)

                            _n_h_total = len(_P_true_h)
                            for i in range(min(_n_holdout, _n_h_total)):
                                plot_p_field_comparison(
                                    _P_true_h, _P_pred_h, sample_idx=i,
                                    title_prefix=f"Holdout (E_in_plane) | sample {i+1}/{_n_h_total}",
                                    save_path=plots_dir / f"p_field_holdout_sample_{i}.png"
                                )
                            logger.info(f"✅ Generated {min(_n_holdout, _n_h_total)} holdout P-field heatmap plots")
                        else:
                            logger.warning("No holdout samples loaded — check E_in_plane data paths")
                    else:
                        logger.info("Holdout data paths not found — skipping inline holdout evaluation")
                except Exception as _he:
                    logger.warning(f"Inline holdout evaluation failed: {_he}")

            # Dummy single-coefficient samples (physics models only)
            # Each sample has exactly one non-zero packed coefficient = 1.0.
            # The true P is the single-mode radiation pattern; the predicted P
            # shows how well the model inverts that simple input.
            if config['model'].get('loss_type', 'coefficient') == "physics":
                try:
                    logger.info("Generating dummy single-coefficient sample plots...")
                    import torch as _torch
                    _maxorder_d  = config['model']['maxorder']
                    _grid_n_phi_d  = config['model'].get('grid_n_phi', 360)
                    _grid_n_theta_d = config['model'].get('grid_n_theta', 179)
                    _n_modes_d   = _maxorder_d * (_maxorder_d + 2)

                    from src.models.physics_layers import DifferentiableMultipoleField as _DMF
                    _field_gen_d = _DMF(
                        maxorder=_maxorder_d,
                        grid_shape=(_grid_n_phi_d, _grid_n_theta_d)
                    )
                    _dg_d = DataGenerator.for_ml_training()

                    # Build human-readable labels for each packed coefficient slot:
                    # slots 0.._n_modes_d-1       → Re(E) for mode (l,m)
                    # slots _n_modes_d..2*n-1      → Im(E)
                    # slots 2*n..3*n-1             → Re(M)
                    # slots 3*n..4*n-1             → Im(M)
                    _modes_d = get_mode_list(_maxorder_d)  # list of (l,m) tuples
                    _type_labels = (
                        [f"Re(E) l={l} m={m}" for l, m in _modes_d] +
                        [f"Im(E) l={l} m={m}" for l, m in _modes_d] +
                        [f"Re(M) l={l} m={m}" for l, m in _modes_d] +
                        [f"Im(M) l={l} m={m}" for l, m in _modes_d]
                    )

                    # Choose which coefficient indices to show:
                    # first 2 E-real modes, first 2 M-real modes (or fewer if n_modes is tiny)
                    _dummy_indices = list(range(min(2, _n_modes_d)))                      # Re(E) 0,1
                    _dummy_indices += list(range(2*_n_modes_d, 2*_n_modes_d + min(2, _n_modes_d)))  # Re(M) 0,1

                    _dummy_dir = plots_dir / "dummy_samples"
                    _dummy_dir.mkdir(exist_ok=True)

                    for _ci in _dummy_indices:
                        _label  = _type_labels[_ci]
                        _packed = np.zeros((1, 4 * _n_modes_d), dtype=np.float32)
                        _packed[0, _ci] = 1.0

                        # True P from DMF  (1, n_theta, n_phi)
                        _P_true_d = _field_gen_d(
                            _torch.from_numpy(_packed).float()
                        ).detach().numpy()

                        # E-field magnitudes from DataGenerator for preprocessing
                        _a_e_c, _a_m_c = unpack_coefficients(_packed)  # complex (1, n_modes)
                        _amp = _dg_d.field_generator.compute_field_from_array(
                            _a_e_c[0], _a_m_c[0], _maxorder_d
                        )  # (n_phi, n_theta, 2)
                        _Et_d = np.abs(_amp[np.newaxis, ..., 0]).astype(np.float32)
                        _Ep_d = np.abs(_amp[np.newaxis, ..., 1]).astype(np.float32)

                        _X_d = preprocessing.transform_features(_Et_d, _Ep_d)
                        if hasattr(model, 'predict_safe'):
                            _pred_c_d = model.predict_safe(_X_d, force_batch=True)
                        else:
                            _pred_c_d = model.predict(_X_d)

                        _P_pred_d = _field_gen_d(
                            _torch.from_numpy(_pred_c_d).float()
                        ).detach().numpy()  # (1, n_theta, n_phi)

                        _safe = _label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
                        plot_p_field_comparison(
                            _P_true_d, _P_pred_d, sample_idx=0,
                            title_prefix=f"Dummy | {_label} only",
                            save_path=_dummy_dir / f"p_field_dummy_{_safe}.png"
                        )

                    logger.info(f"✅ Generated {len(_dummy_indices)} dummy single-coefficient plots in {_dummy_dir}")
                except Exception as _de:
                    logger.warning(f"Dummy sample plots failed: {_de}")

            # Save preprocessing pipeline
            preprocessing_save_path = experiment_dir / "preprocessing"
            preprocessing.save(preprocessing_save_path)
            
            # Register model in MLFlow
            logger.info("Registering model in MLFlow...")
            try:
                # Create input example for model signature
                input_example = X_test[:5] if len(X_test) >= 5 else X_test
                
                model_name = f"{experiment_name}_model"
                model_version = experiment_manager.register_model(
                    model=model,
                    model_name=model_name,
                    input_example=input_example,
                    performance_metrics=test_metrics,
                    auto_promote=True
                )
                
                if model_version:
                    logger.info(f"Model registered as version {model_version}")
                    
            except Exception as e:
                logger.warning(f"Model registration failed: {e}")
            
            # Log experiment artifacts
            experiment_manager.log_artifacts_directory(experiment_dir, "experiment_artifacts")
            
            # Finish MLFlow experiment
            try:
                # Count model parameters (handle both PyTorch and custom models)
                if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                    model_params = sum(p.numel() for p in model.model.parameters())
                elif hasattr(model, 'parameters'):
                    model_params = sum(p.numel() for p in model.parameters())
                else:
                    model_params = 0  # Unknown parameter count
            except:
                model_params = 0
            
            experiment_manager.finish_experiment(summary_metrics={
                **test_metrics,
                "training_time": training_result.get('training_time', 0),
                "total_epochs": config['training']['epochs'],
                "model_parameters": model_params
            })
            
            # Print summary
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY WITH MLFLOW")
            print("="*60)
            print(f"Experiment: {config['experiment']['name']}")
            print(f"Directory: {experiment_dir}")
            print(f"Test MSE: {test_metrics['test_mse']:.6f}")
            print(f"Test R²: {test_metrics['test_r2']:.4f}")
            print(f"Training time: {training_result.get('training_time', 0):.2f}s")
            
            # Show MLFlow experiment URL
            experiment_url = experiment_manager.get_experiment_url()
            if experiment_url:
                print(f"MLFlow UI: {experiment_url}")
            
            print("="*60)
            
        else:
            logger.warning("MLFlow tracking unavailable - running without experiment tracking")
            
            # Continue with minimal training (without MLFlow)
            logger.info("Running minimal training without MLflow tracking...")
            E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
            
            # Use proper preprocessing setup (no data leakage)
            using_streaming = isinstance(E_theta, str)
            
            if using_streaming:
                n_samples = np.load(E_theta, mmap_mode='r').shape[0]
            else:
                n_samples = E_theta.shape[0]
            
            # Split indices first (prevent data leakage)
            data_seed = config['data'].get('seed', config.get('seed', 42))
            train_idx, val_idx, test_idx = split_indices(n_samples, config, seed=data_seed)
            
            # Setup preprocessing on training data only
            if using_streaming:
                preprocessing = setup_preprocessing_streaming_no_leakage(
                    config, E_theta, E_phi, y_coeffs, train_idx
                )
                # Transform features
                cache_dir = Path(data_info['cache_dir'])
                X_path = cache_dir / "X_transformed_minimal.npy"
                preprocessing.transform_features_streaming(E_theta, E_phi, str(X_path))
                X = np.load(X_path, mmap_mode='r')
                
                # Select targets
                loss_type = config['model'].get('loss_type', 'coefficient')
                if loss_type == "physics":
                    y_targets = np.load(y_P, mmap_mode='r')
                else:
                    y_targets = np.load(y_coeffs, mmap_mode='r')
            else:
                preprocessing = setup_preprocessing_batched_no_leakage(
                    config, E_theta, E_phi, y_coeffs, train_idx
                )
                X = transform_features_batched(preprocessing, E_theta, E_phi)
                
                # Select targets
                loss_type = config['model'].get('loss_type', 'coefficient')
                if loss_type == "physics":
                    y_targets = y_P
                else:
                    y_targets = y_coeffs
            
            # Split data using predetermined indices
            X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
            y_train, y_val, y_test = y_targets[train_idx], y_targets[val_idx], y_targets[test_idx]
            
            # Create model with correct dimensions
            input_dim = X_train.shape[1]
            if loss_type == "physics":
                output_dim = y_coeffs.shape[1] if not using_streaming else data_info['coeffs_shape'][0]
            else:
                output_dim = y_train.shape[1]
                
            model = create_model_from_config(config, input_dim, output_dim)
            training_result = train_model(model, X_train, y_train, X_val, y_val, config, Path("./minimal_run"))
            test_metrics, predictions = evaluate_model(model, X_test, y_test, preprocessing, config, experiment_dir)
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED (NO MLFLOW)")
            print("="*60)
            print(f"Test MSE: {test_metrics['test_mse']:.6f}")
            print(f"Test R²: {test_metrics['test_r2']:.4f}")
            print("="*60)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()