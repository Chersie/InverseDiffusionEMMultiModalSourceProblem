#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates a trained model on the E_in_plane test dataset.
Loads the model from the MLFlow registry and logs evaluation metrics.
"""

import os
# Disable MPS completely to prevent crashes during model loading
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_DISABLE_MPS'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import sys
import argparse
from pathlib import Path
import logging
import time
import numpy as np

# Patch torch.load before importing torch
import torch
original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import Config
from src.core.dataset_loader import TestDatasetLoader
from src.models.mlflow_integration import get_model_registry
from src.api.preprocessing import PreprocessingPipeline
from experiments.utils.mlflow_manager import mlflow_training_session
from experiments.utils.plotting import plot_prediction_scatter, plot_field_comparison, plot_p_field_comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_on_test_set(model_name: str, model_version: str = None, model_stage: str = "Production", limit: int = None):
    """
    Evaluate a model on the test dataset.
    """
    logger.info(f"Evaluating model '{model_name}' (version: {model_version or 'latest'}, stage: {model_stage})")
    
    config = Config() # Load default config for paths
    
    # 1. Load Model and Preprocessing
    logger.info("Getting registry..."); registry = get_model_registry()
    
    # Load model using proper MLflow registry to ensure consistency
    # This respects model_name, model_version, and model_stage parameters
    try:
        logger.info(f"Loading model '{model_name}' (version={model_version}, stage={model_stage}) from MLflow registry...")
        model = registry.load_model(model_name, version=model_version, stage=model_stage)
        logger.info("✅ Successfully loaded model via MLflow registry")
    except Exception as registry_error:
        logger.warning(f"MLflow registry loading failed: {registry_error}")
        
        # Only fall back to direct pickle loading if registry fails
        # But be more careful to match the model name if possible
        try:
            import pickle
            import glob
            
            logger.warning("Falling back to direct pickle loading (less reliable)...")
            
            # Find model.pkl files
            pattern = f"mlartifacts/*/models/*/artifacts/model.pkl"
            pkl_files = glob.glob(pattern)
            
            if pkl_files:
                # Use the most recent model.pkl as last resort
                latest_pkl = max(pkl_files, key=os.path.getctime)
                logger.warning(f"Loading most recent pickle file: {latest_pkl}")
                logger.warning("WARNING: This may not be the requested model!")
                
                with open(latest_pkl, "rb") as f:
                    model = pickle.load(f)
                    
                logger.info("✅ Loaded model via direct pickle loading (fallback)")
            else:
                raise FileNotFoundError("No model files found")
        except Exception as pickle_error:
            logger.error(f"All model loading methods failed: {pickle_error}")
            raise
    
    if model is None:
        logger.error(f"Could not load model '{model_name}' from registry.")
        return
        
    # We need the preprocessing pipeline used during training.
    # Ideally, this should be logged as an artifact with the model.
    # For now, we'll try to find it in the latest run of the model's experiment.
    # A robust solution would fetch the specific artifact from the model version's run_id.
    
    # Let's assume the preprocessing pipeline is saved locally in a known location for this example,
    # OR we can re-fit it if we don't have it (though re-fitting on test data is bad practice).
    # Since we don't have a direct link to the preprocessing artifact in this simple script,
    # we'll ask the user to provide the path to the preprocessing pipeline, or we'll try to load a default one.
    
    # For demonstration, let's assume we have a saved preprocessing pipeline from a recent run.
    # In a real scenario, you'd download the artifact from MLFlow:
    # client = registry.client
    # version_info = registry.get_model_version(model_name, model_version, model_stage)
    # local_path = client.download_artifacts(version_info.run_id, "experiment_artifacts/preprocessing")
    # preprocessing = PreprocessingPipeline.load(local_path)
    
    # PROPER ML EVALUATION: Load preprocessing pipeline from training artifacts
    # We NEVER fit preprocessing on test data - that would be data leakage!
    logger.info("Loading preprocessing pipeline from training artifacts...")
    
    try:
        # Get the model version info to extract the run_id
        import mlflow
        from mlflow import MlflowClient
        
        client = MlflowClient()
        
        # Get model version information - ensuring it matches the loaded model
        if model_version:
            version_info = client.get_model_version(model_name, model_version)
        else:
            # Get latest version, respecting the stage parameter
            if model_stage and model_stage.lower() != "none":
                stages_to_check = [model_stage]
            else:
                # Check all stages if no specific stage requested
                stages_to_check = ["Production", "Staging", "None"]
            
            latest_versions = client.get_latest_versions(model_name, stages=stages_to_check)
            if not latest_versions:
                raise Exception(f"No versions found for model '{model_name}' with stages {stages_to_check}")
            
            # Use the first match (should be the latest within the requested stages)
            version_info = latest_versions[0]
            
        logger.info(f"Found model version {version_info.version} from run {version_info.run_id}")
        
        # Download preprocessing artifacts from the training run
        preprocessing_path = mlflow.artifacts.download_artifacts(
            run_id=version_info.run_id, 
            artifact_path="experiment_artifacts/preprocessing"
        )
        
        if os.path.exists(preprocessing_path) and os.listdir(preprocessing_path):
            from src.api.preprocessing import PreprocessingConfig
            preprocessing_config = PreprocessingConfig()
            preprocessing = PreprocessingPipeline(preprocessing_config)
            preprocessing.load(preprocessing_path)
            logger.info("✅ Loaded preprocessing pipeline from MLFlow training run")
        else:
            raise FileNotFoundError(f"No fitted preprocessing pipeline found at {preprocessing_path}")
            
    except Exception as e:
        logger.error(f"❌ Cannot load preprocessing pipeline: {e}")
        logger.error("❌ EVALUATION CANNOT PROCEED WITHOUT PROPER PREPROCESSING")
        logger.error("❌ The preprocessing pipeline must be:")
        logger.error("   1. Fitted on TRAINING data during model training")
        logger.error("   2. Saved as an MLFlow artifact under 'experiment_artifacts/preprocessing'")
        logger.error("   3. Loaded here from the training run artifacts")
        logger.error("❌ We NEVER fit preprocessing on test data - that's data leakage!")
        logger.error("❌ Please retrain your model and ensure preprocessing artifacts are saved.")
        return
    
    # 2. Load Test Dataset
    loader = TestDatasetLoader(
        features_dir=config.paths.test_features_dir,
        targets_dir=config.paths.test_targets_dir
    )
    
    # We need to know the maxorder the model was trained on.
    # We can infer this from the model's output dimension or config if available.
    # For simplicity, we'll use a default or ask for it.
    maxorder = 5 # Default
    if hasattr(model, 'config') and hasattr(model.config, 'maxorder'):
        maxorder = model.config.maxorder
        
    E_theta, E_phi, a_e, a_m = loader.load_dataset(maxorder=maxorder, limit=limit)
    
    # 3. Preprocess Data
    logger.info("Preprocessing test data...")
    
    # Reshape flattened test data to 3D format expected by preprocessing
    # Test data comes as (n_samples, 64440) but preprocessing expects (n_samples, n_phi, n_theta)
    n_samples = E_theta.shape[0]
    n_phi = 360  # Standard grid size
    n_theta = 179  # Standard grid size
    expected_size = n_phi * n_theta
    
    logger.info(f"Reshaping test data from {E_theta.shape} to ({n_samples}, {n_phi}, {n_theta})")
    
    if E_theta.shape[1] != expected_size:
        logger.warning(f"Expected flattened size {expected_size}, got {E_theta.shape[1]}")
        # Try to infer grid dimensions
        total_points = E_theta.shape[1]
        # Common grid sizes: 360x179, 180x90, 720x360, etc.
        possible_combinations = [
            (360, 179), (180, 90), (720, 360), (180, 179), (360, 90)
        ]
        for n_phi_try, n_theta_try in possible_combinations:
            if n_phi_try * n_theta_try == total_points:
                n_phi, n_theta = n_phi_try, n_theta_try
                logger.info(f"Inferred grid dimensions: {n_phi} × {n_theta}")
                break
        else:
            # Default to square-ish grid
            n_phi = int(np.sqrt(total_points))
            n_theta = total_points // n_phi
            logger.warning(f"Using approximate grid: {n_phi} × {n_theta}")
    
    # Reshape to 3D
    try:
        E_theta_3d = E_theta.reshape(n_samples, n_phi, n_theta)
        E_phi_3d = E_phi.reshape(n_samples, n_phi, n_theta)
        logger.info(f"✅ Successfully reshaped test data to 3D: {E_theta_3d.shape}")
        
        X_test = preprocessing.transform_features(E_theta_3d, E_phi_3d)
        y_test = preprocessing.process_coefficients(a_e, a_m)
    except ValueError as reshape_error:
        logger.error(f"Failed to reshape test data: {reshape_error}")
        logger.error(f"E_theta shape: {E_theta.shape}, expected reshape: ({n_samples}, {n_phi}, {n_theta})")
        return
    
    # 4. Evaluate
    # Check if model was trained with physics loss (requires P-field evaluation for consistency)
    physics_trained = False
    if hasattr(model, 'config') and hasattr(model.config, 'loss_type'):
        physics_trained = model.config.loss_type == "physics"
    elif hasattr(model, '_config') and hasattr(model._config, 'loss_type'):
        physics_trained = model._config.loss_type == "physics"
    
    if physics_trained:
        logger.info("Physics-trained model detected: coefficient metrics AND P-field metrics will be computed.")
        
    logger.info("Running inference...")
    start_time = time.time()
    # Use batched prediction for memory safety
    if hasattr(model, 'predict_safe'):
        predictions = model.predict_safe(X_test, force_batch=True)
    else:
        # Fallback for models without predict_safe
        if hasattr(model, 'predict_batch'):
            predictions = model.predict_batch(X_test)
        else:
            predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate coefficient metrics
    mse = np.mean((y_test - predictions) ** 2)
    mae = np.mean(np.abs(y_test - predictions))
    
    # Calculate R² using the standard SS-based definition (same as training script)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate throughput safely (avoid division by zero)
    samples_per_second = float(len(X_test) / max(inference_time, 1e-9))
    
    metrics = {
        "test_dataset_mse": float(mse),
        "test_dataset_mae": float(mae),
        "test_dataset_r2": float(r2),
        "inference_time_seconds": float(inference_time),
        "samples_per_second": samples_per_second
    }
    
    # For physics-trained models, also compute P-field metrics which match the training loss
    if physics_trained:
        try:
            from src.models.physics_layers import DifferentiableMultipoleField
            import torch
            
            grid_n_phi = E_theta_3d.shape[1]
            grid_n_theta = E_theta_3d.shape[2]
            
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder,
                grid_shape=(grid_n_phi, grid_n_theta)
            )
            
            # True P field: E_theta_3d has shape (N, n_phi, n_theta) from TestDatasetLoader.
            # Transpose to (N, n_theta, n_phi) to match DifferentiableMultipoleField output.
            P_true = (np.abs(E_theta_3d) ** 2 + np.abs(E_phi_3d) ** 2).transpose(0, 2, 1).astype(np.float32)
            
            # Predicted P field: (N, n_theta, n_phi) — already aligned with P_true after fix
            pred_tensor = torch.from_numpy(predictions).float()
            P_pred = field_gen(pred_tensor).detach().numpy()
            
            p_mse = float(np.mean((P_true - P_pred) ** 2))
            p_mae = float(np.mean(np.abs(P_true - P_pred)))
            ss_res_p = np.sum((P_true - P_pred) ** 2)
            ss_tot_p = np.sum((P_true - np.mean(P_true)) ** 2)
            p_r2 = float(1 - (ss_res_p / ss_tot_p)) if ss_tot_p > 0 else 0.0
            
            metrics["physics_p_field_mse"] = p_mse
            metrics["physics_p_field_mae"] = p_mae
            metrics["physics_p_field_r2"] = p_r2
            logger.info(f"Physics P-field metrics: MSE={p_mse:.6f}, MAE={p_mae:.6f}, R²={p_r2:.4f}")

            # Per-sample 2D P-field heatmap comparison plots for holdout test
            holdout_plot_dir = Path("experiments/results/eval_plots")
            holdout_plot_dir.mkdir(parents=True, exist_ok=True)
            n_holdout_total = len(P_true)
            n_holdout_plots = min(3, n_holdout_total)
            for i in range(n_holdout_plots):
                try:
                    plot_p_field_comparison(
                        P_true, P_pred, sample_idx=i,
                        title_prefix=f"Holdout (E_in_plane) | sample {i+1}/{n_holdout_total}",
                        save_path=holdout_plot_dir / f"p_field_comparison_holdout_sample_{i}.png"
                    )
                    logger.info(f"✅ P-field heatmap saved: holdout sample {i+1}/{n_holdout_total}")
                except Exception as plot_err:
                    logger.warning(f"Failed to generate P-field heatmap for holdout sample {i}: {plot_err}")

        except Exception as p_err:
            logger.warning(f"Could not compute P-field metrics for physics model: {p_err}")
    
    logger.info(f"Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
        
    # 5. Log to MLFlow
    experiment_name = "test_dataset_evaluation"
    run_name = f"eval_{model_name}_{model_version or 'latest'}"
    
    with mlflow_training_session(experiment_name, run_name=run_name) as session:
        session.tracker.log_params({
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
            "test_samples": len(X_test),
            "maxorder": maxorder
        })
        
        session.log_model_performance(metrics)
        
        # Generate and log plot
        plot_dir = Path("experiments/results/eval_plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"scatter_{run_name}.png"
        
        plot_prediction_scatter(
            y_test, predictions,
            title=f"Test Dataset Predictions: {model_name}",
            save_path=plot_path
        )
        
        session.tracker.log_artifact(plot_path, "evaluation_plots")
        
        # Generate comprehensive field comparison plots for selected test samples
        logger.info("Generating field comparison plots for test samples...")
        maxorder = 5  # Default maxorder for field reconstruction
        # Try to infer maxorder from model or use a reasonable default
        if hasattr(model, 'config') and hasattr(model.config, 'maxorder'):
            maxorder = model.config.maxorder
        
        n_field_plots = min(3, len(y_test))  # Plot first 3 test samples
        for i in range(n_field_plots):
            sample_title = f"Sample {i+1} [test]"
            field_plot_path = plot_dir / f"field_comparison_{run_name}_sample_{i}.png"
            
            try:
                plot_field_comparison(
                    y_test, predictions, sample_idx=i, maxorder=maxorder,
                    title_prefix=sample_title,
                    save_path=field_plot_path
                )
                session.tracker.log_artifact(field_plot_path, "evaluation_plots")
            except Exception as e:
                logger.warning(f"Failed to generate field comparison plot for sample {i}: {e}")
        
        # Log per-sample P-field heatmaps to MLflow if they were generated
        if physics_trained:
            holdout_plot_dir = Path("experiments/results/eval_plots")
            for i in range(min(3, len(y_test))):
                hp = holdout_plot_dir / f"p_field_comparison_holdout_sample_{i}.png"
                if hp.exists():
                    session.tracker.log_artifact(hp, "evaluation_plots")

        logger.info(f"Logged evaluation results to MLFlow experiment '{experiment_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model in MLFlow registry")
    parser.add_argument("--model-version", type=str, help="Specific version to evaluate")
    parser.add_argument("--model-stage", type=str, default="Production", help="Model stage (e.g., Production, Staging)")
    parser.add_argument("--limit", type=int, help="Limit number of test samples")
    
    args = parser.parse_args()
    
    evaluate_on_test_set(
        model_name=args.model_name,
        model_version=args.model_version,
        model_stage=args.model_stage,
        limit=args.limit
    )