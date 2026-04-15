"""
Hyperparameter Optimization Framework with Optuna Integration

Automated hyperparameter tuning for electromagnetic multipole ML models
with MLFlow experiment tracking and intelligent search strategies.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    MLflowCallback = None
    OPTUNA_AVAILABLE = False

import numpy as np
import torch

from src.models.mlflow_integration import get_model_registry
from src.core.data_generator import DataGenerator
from src.api.preprocessing import PreprocessingPipeline
from src.core.config import Config
from .mlflow_manager import create_experiment_manager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    
    # Study settings
    study_name: Optional[str] = None
    direction: str = "minimize"  # "minimize" or "maximize"
    objective_metric: str = "val_loss"  # metric to optimize
    
    # Sampler configuration
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Pruning configuration
    enable_pruning: bool = True
    pruner: str = "median"  # "median", "successive_halving", None
    pruner_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 10
    min_improvement: float = 1e-6
    
    # Search space bounds
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # MLFlow integration
    use_mlflow: bool = True
    experiment_name: str = "hyperparameter_optimization"
    
    # Resource constraints
    max_epochs_per_trial: int = 50
    min_epochs_per_trial: int = 5


class HyperparameterOptimizer:
    """Advanced hyperparameter optimizer using Optuna with MLFlow integration."""
    
    def __init__(self, 
                 config: OptimizationConfig,
                 base_config: Optional[Config] = None):
        self.config = config
        self.base_config = base_config
        self.study: Optional[optuna.Study] = None
        self.experiment_manager: Optional[Any] = None
        
        # Optimization state
        self._best_score = float('inf') if config.direction == 'minimize' else float('-inf')
        self._no_improvement_count = 0
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner."""
        
        # Configure sampler
        if self.config.sampler == "tpe":
            sampler = TPESampler(**self.config.sampler_kwargs)
        elif self.config.sampler == "random":
            sampler = RandomSampler(**self.config.sampler_kwargs)
        elif self.config.sampler == "cmaes":
            sampler = CmaEsSampler(**self.config.sampler_kwargs)
        else:
            sampler = TPESampler()
        
        # Configure pruner
        pruner = None
        if self.config.enable_pruning:
            if self.config.pruner == "median":
                pruner = MedianPruner(**self.config.pruner_kwargs)
            elif self.config.pruner == "successive_halving":
                pruner = SuccessiveHalvingPruner(**self.config.pruner_kwargs)
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial based on search space."""
        
        params = {}
        
        # Default search space if none provided
        if not self.config.search_space:
            # Define reasonable defaults for electromagnetic multipole models
            params.update({
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512, 1024]),
                'n_hidden_layers': trial.suggest_int('n_hidden_layers', 2, 6),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            })
        else:
            # Use configured search space
            for param_name, param_config in self.config.search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
        
        return params
    
    def create_objective_function(self, 
                                training_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Callable:
        """Create objective function for optimization."""
        
        X_train, y_train, _ = training_data
        X_val, y_val, _ = validation_data
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for a single trial."""
            
            try:
                # Suggest hyperparameters
                params = self.suggest_hyperparameters(trial)
                
                # Create MLFlow run for this trial if enabled
                if self.config.use_mlflow and self.experiment_manager:
                    trial_name = f"trial_{trial.number}_{int(time.time())}"
                    
                    # Start nested run
                    with self.experiment_manager.mlflow_manager.start_run(
                        run_name=trial_name, 
                        nested=True
                    ) as run:
                        
                        # Log trial parameters
                        for key, value in params.items():
                            self.experiment_manager.tracker.log_param(f"trial.{key}", value)
                        self.experiment_manager.tracker.log_param("trial.number", trial.number)
                        
                        # Train and evaluate model
                        score = self._train_and_evaluate_model(params, X_train, y_train, X_val, y_val, trial)
                        
                        # Log trial result
                        self.experiment_manager.tracker.log_metric("trial.objective_score", score)
                        self.experiment_manager.tracker.log_metric("trial.completed", 1)
                        
                        return score
                else:
                    # Run without MLFlow
                    return self._train_and_evaluate_model(params, X_train, y_train, X_val, y_val, trial)
                    
            except optuna.TrialPruned:
                # Trial was pruned - this is expected behavior
                if self.config.use_mlflow and self.experiment_manager:
                    self.experiment_manager.tracker.log_metric("trial.pruned", 1)
                raise
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                if self.config.use_mlflow and self.experiment_manager:
                    self.experiment_manager.tracker.log_param("trial.error", str(e))
                    self.experiment_manager.tracker.log_metric("trial.failed", 1)
                
                # Return worst possible score
                return float('inf') if self.config.direction == 'minimize' else float('-inf')
        
        return objective
    
    def _train_and_evaluate_model(self, 
                                params: Dict[str, Any],
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                trial: optuna.Trial) -> float:
        """Train and evaluate model with given hyperparameters."""
        
        # Create model with suggested parameters
        registry = get_model_registry()
        model = registry.create_model(
            model_type="mlp",
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            hidden_size=params.get('hidden_size', 256),
            n_hidden_layers=params.get('n_hidden_layers', 3),
            dropout_rate=params.get('dropout_rate', 0.1),
            learning_rate=params.get('learning_rate', 0.001),
            batch_size=params.get('batch_size', 64),
            epochs=min(params.get('epochs', self.config.max_epochs_per_trial), 
                      self.config.max_epochs_per_trial)
        )
        
        # Train model with pruning callback
        if self.config.enable_pruning:
            # Custom training with intermediate reporting for pruning
            return self._train_with_pruning(model, X_train, y_train, X_val, y_val, trial, params)
        else:
            # Regular training
            result = model.fit(X_train, y_train, X_val, y_val)
            
            # Extract objective score
            if self.config.objective_metric == "val_loss":
                return result.get('final_val_loss', float('inf'))
            elif self.config.objective_metric == "val_r2":
                predictions = model.predict(X_val)
                r2 = 1 - np.var(y_val - predictions) / np.var(y_val)
                return -r2  # Negative because we minimize
            else:
                return result.get(self.config.objective_metric, float('inf'))
    
    def _train_with_pruning(self, 
                          model: Any,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          trial: optuna.Trial,
                          params: Dict[str, Any]) -> float:
        """Train model with Optuna pruning support."""
        
        epochs = min(params.get('epochs', self.config.max_epochs_per_trial), 
                    self.config.max_epochs_per_trial)
        
        # Custom training loop with intermediate reporting
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Perform one epoch of training
            # Note: This is a simplified example - real implementation would
            # need to modify the model's training loop to support epoch-by-epoch reporting
            
            # For now, we'll do periodic evaluation
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Evaluate model
                predictions = model.predict(X_val)
                val_loss = np.mean((y_val - predictions) ** 2)
                
                # Report intermediate value for pruning
                trial.report(val_loss, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                best_val_loss = min(best_val_loss, val_loss)
        
        return best_val_loss
    
    def optimize(self, 
                training_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                callbacks: Optional[List[Callable]] = None) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            training_data: Tuple of (X_train, y_train, metadata)
            validation_data: Tuple of (X_val, y_val, metadata). If None, will split training data
            callbacks: Optional list of callbacks for the study
            
        Returns:
            Completed Optuna study
        """
        
        logger.info("Starting hyperparameter optimization...")
        
        # Setup validation data
        if validation_data is None:
            # Split training data
            X, y, meta = training_data
            split_idx = int(0.8 * len(X))
            training_data = (X[:split_idx], y[:split_idx], meta)
            validation_data = (X[split_idx:], y[split_idx:], meta)
        
        # Setup MLFlow experiment
        if self.config.use_mlflow:
            self.experiment_manager = create_experiment_manager(self.config.experiment_name)
            run_name = f"optuna_study_{int(time.time())}"
            
            if self.experiment_manager.start_experiment(run_name=run_name):
                # Log optimization configuration
                self.experiment_manager.tracker.log_params({
                    "optuna.n_trials": self.config.n_trials,
                    "optuna.sampler": self.config.sampler,
                    "optuna.pruner": self.config.pruner if self.config.enable_pruning else "none",
                    "optuna.direction": self.config.direction,
                    "optuna.objective_metric": self.config.objective_metric
                })
        
        # Create study
        self.study = self.create_study()
        
        # Setup callbacks
        study_callbacks = callbacks or []
        if self.config.use_mlflow and self.experiment_manager:
            # Note: MLflowCallback integration would go here if using Optuna's built-in callback
            pass
        
        # Create objective function
        objective_func = self.create_objective_function(training_data, validation_data)
        
        # Run optimization
        start_time = time.time()
        
        try:
            self.study.optimize(
                objective_func,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                callbacks=study_callbacks
            )
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # Log final results
        if self.config.use_mlflow and self.experiment_manager:
            best_trial = self.study.best_trial
            
            # Log best trial results
            self.experiment_manager.tracker.log_params({
                f"best.{k}": v for k, v in best_trial.params.items()
            })
            self.experiment_manager.tracker.log_metrics({
                "best.objective_value": best_trial.value,
                "best.trial_number": best_trial.number,
                "optimization.total_time": optimization_time,
                "optimization.n_completed_trials": len(self.study.trials),
                "optimization.n_pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            })
            
            # Finish experiment
            self.experiment_manager.finish_experiment()
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_trial.value:.6f}")
        logger.info(f"Best params: {self.study.best_trial.params}")
        
        return self.study
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get configuration with best hyperparameters."""
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")
        
        return self.study.best_trial.params
    
    def save_study(self, path: Path):
        """Save study results to file."""
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")
        
        study_data = {
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params,
                'state': str(self.study.best_trial.state)
            },
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                }
                for trial in self.study.trials
            ],
            'study_name': self.study.study_name,
            'direction': str(self.study.direction)
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        logger.info(f"Study saved to {path}")


def create_optimization_config(
    n_trials: int = 100,
    objective_metric: str = "val_loss",
    search_space: Optional[Dict[str, Any]] = None,
    **kwargs
) -> OptimizationConfig:
    """Create optimization configuration with sensible defaults."""
    
    return OptimizationConfig(
        n_trials=n_trials,
        objective_metric=objective_metric,
        search_space=search_space or {},
        **kwargs
    )


def optimize_mlp_hyperparameters(
    maxorder: int,
    n_samples: int = 1000,
    n_trials: int = 50,
    experiment_name: str = "mlp_hyperparameter_optimization"
) -> Dict[str, Any]:
    """
    Convenience function to optimize MLP hyperparameters for electromagnetic multipole analysis.
    
    Args:
        maxorder: Maximum multipole order
        n_samples: Number of training samples
        n_trials: Number of optimization trials
        experiment_name: Name for MLFlow experiment
        
    Returns:
        Dictionary with best hyperparameters
    """
    
    logger.info(f"Starting hyperparameter optimization for maxorder={maxorder}")
    
    # Generate training data
    logger.info(f"Generating {n_samples} training samples...")
    generator = DataGenerator.for_ml_training()
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    
    # Setup preprocessing
    preprocessing = PreprocessingPipeline()
    E_theta = dataset['amplitude'][..., 0]
    E_phi = dataset['amplitude'][..., 1]
    
    from src.core.data_generator import pack_coefficients
    targets = pack_coefficients(dataset['coefficients_e'], dataset['coefficients_m'])
    
    preprocessing.fit(E_theta, E_phi, targets=targets)
    X = preprocessing.transform_features(E_theta, E_phi)
    y = preprocessing.process_coefficients(dataset['coefficients_e'], dataset['coefficients_m'])
    
    logger.info(f"Prepared data: X={X.shape}, y={y.shape}")
    
    # Create optimization configuration
    config = OptimizationConfig(
        n_trials=n_trials,
        experiment_name=experiment_name,
        objective_metric="val_loss",
        direction="minimize",
        enable_pruning=True,
        early_stopping_rounds=5
    )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(config)
    study = optimizer.optimize(training_data=(X, y, {}))
    
    # Return best configuration
    best_params = optimizer.get_best_config()
    logger.info(f"Best hyperparameters: {best_params}")
    
    return best_params


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for electromagnetic multipole models")
    parser.add_argument("--maxorder", type=int, default=5, help="Maximum multipole order")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--experiment_name", type=str, default="hyperparameter_optimization", help="MLFlow experiment name")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    best_params = optimize_mlp_hyperparameters(
        maxorder=args.maxorder,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name
    )
    
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*50)
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print("="*50)