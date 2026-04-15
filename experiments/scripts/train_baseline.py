#!/usr/bin/env python3
"""
Baseline Model Training Script

This script trains baseline Ridge/Linear regression models for comparison
with neural network approaches.

Usage:
    python train_baseline.py --config configs/baseline_comparison.yaml
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import yaml
import mlflow

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.data_generator import DataGenerator, pack_coefficients
from src.models.registry import create_baseline
from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_experiment_directory(config: Dict[str, Any]) -> Path:
    """Set up experiment directory with timestamp."""
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = Path("experiments/results") / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = experiment_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def generate_training_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate synthetic training data."""
    logger.info("Generating synthetic training data...")
    
    maxorder = config['model']['maxorder']
    n_samples = config['training']['n_samples']
    seed = config['data']['seed']
    
    # Create data generator
    generator = DataGenerator.for_ml_training()
    np.random.seed(seed)
    
    # Generate dataset
    start_time = time.time()
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    generation_time = time.time() - start_time
    
    logger.info(f"Generated {n_samples} samples in {generation_time:.2f}s")
    
    # Use power patterns as features for baseline
    X = dataset['power'].reshape(n_samples, -1)  # Flatten power patterns
    
    # Pack coefficients as targets
    y = pack_coefficients(dataset['coefficients_e'], dataset['coefficients_m'])
    
    return X, y, {
        'generation_time': generation_time,
        'dataset_shape': dataset['amplitude'].shape,
        'n_modes': dataset['coefficients_e'].shape[1],
        'feature_dim': X.shape[1]
    }


def split_data(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    """Split data into train/val/test sets."""
    n_samples = len(X)
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model_from_config(config: Dict[str, Any], input_dim: int, output_dim: int):
    """Create baseline model from configuration."""
    model_config = config['model']
    
    model = create_baseline(
        input_dim=input_dim,
        output_dim=output_dim,
        baseline_type=model_config['baseline_type'],
        ridge_alpha=model_config.get('ridge_alpha', 1.0),
        max_iter=model_config.get('max_iter', 1000)
    )
    
    logger.info(f"Created baseline model: {model.get_model_info()}")
    return model


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_dir: Path
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Train and evaluate baseline model."""
    logger.info("Training baseline model...")
    
    # Train model
    training_result = model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = model.predict(X_test)
    
    # Compute test metrics
    test_mse = float(np.mean((y_test - test_predictions) ** 2))
    test_mae = float(np.mean(np.abs(y_test - test_predictions)))
    test_rmse = float(np.sqrt(test_mse))
    
    # R² score
    ss_res = np.sum((y_test - test_predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    test_metrics = {
        'test_mse': test_mse,
        'test_mae': test_mae, 
        'test_rmse': test_rmse,
        'test_r2': r2_score
    }
    
    # Validation predictions for comparison
    val_predictions = model.predict(X_val)
    val_mse = float(np.mean((y_val - val_predictions) ** 2))
    test_metrics['val_mse'] = val_mse
    
    logger.info(f"Training MSE: {training_result.get('train_mse', 'N/A')}")
    logger.info(f"Validation MSE: {val_mse:.6f}")
    logger.info(f"Test MSE: {test_mse:.6f}")
    logger.info(f"Test R²: {r2_score:.4f}")
    
    # Save model
    model_save_path = experiment_dir / "model"
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return training_result, test_metrics


def setup_mlflow_tracking(config: Dict[str, Any], experiment_dir: Path):
    """Set up MLflow experiment tracking."""
    experiment_name = config['experiment']['name']
    mlflow.set_experiment(experiment_name)
    
    tags = config['experiment'].get('tags', [])
    mlflow_tags = {f"tag_{i}": tag for i, tag in enumerate(tags)}
    mlflow_tags.update({
        "experiment_dir": str(experiment_dir),
        "description": config['experiment']['description']
    })
    
    return mlflow.start_run(tags=mlflow_tags)


def log_to_mlflow(config: Dict[str, Any], training_result: Dict[str, Any],
                  test_metrics: Dict[str, float], data_info: Dict[str, Any]):
    """Log results to MLflow."""
    # Parameters
    mlflow.log_params({
        "model_type": config['model']['type'],
        "baseline_type": config['model']['baseline_type'], 
        "maxorder": config['model']['maxorder'],
        "ridge_alpha": config['model'].get('ridge_alpha', 1.0),
        "n_samples": config['training']['n_samples'],
        "feature_dim": data_info['feature_dim'],
        "n_modes": data_info['n_modes']
    })
    
    # Metrics
    mlflow.log_metrics({
        "train_mse": training_result.get('train_mse', 0),
        "training_time": training_result.get('training_time', 0),
        "generation_time": data_info['generation_time'],
        **test_metrics
    })


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--maxorder", type=int, help="Override maxorder")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(Path(args.config))
        
        if args.maxorder:
            config['model']['maxorder'] = args.maxorder
        
        # Set up experiment
        experiment_dir = setup_experiment_directory(config)
        
        with setup_mlflow_tracking(config, experiment_dir):
            
            # Generate data
            X, y, data_info = generate_training_data(config)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)
            
            # Create model
            model = create_model_from_config(config, X.shape[1], y.shape[1])
            
            # Train and evaluate
            training_result, test_metrics = train_and_evaluate(
                model, X_train, y_train, X_val, y_val, X_test, y_test, experiment_dir
            )
            
            # Log to MLflow
            log_to_mlflow(config, training_result, test_metrics, data_info)
            
            # Print summary
            print("\n" + "="*60)
            print("BASELINE TRAINING COMPLETED")
            print("="*60)
            print(f"Model: {config['model']['baseline_type']}")
            print(f"Test MSE: {test_metrics['test_mse']:.6f}")
            print(f"Test R²: {test_metrics['test_r2']:.4f}")
            print(f"Training time: {training_result.get('training_time', 0):.2f}s")
            print("="*60)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()