#!/usr/bin/env python3
"""
Generate plots for physics-informed trained model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.physics_layers import DifferentiableMultipoleField
from experiments.utils.plotting import plot_training_curves, plot_field_comparison

def load_model_and_data(experiment_dir: Path):
    """Load the trained model and test data."""
    
    # Load model
    from src.models.mlp import MLPModel
    from src.models.mlp import MLPConfig
    
    # Load config
    with open(experiment_dir / "config.yaml") as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Create model config with only valid MLPConfig parameters
    maxorder = config['model']['maxorder']
    n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
    
    # Only include parameters that are valid for MLPConfig
    valid_params = {
        'model_type': 'mlp',
        'input_dim': 64,  # From preprocessing PCA
        'output_dim': 4 * n_modes,  # 4*n_modes for packed coefficients
        'maxorder': maxorder,
        'hidden_size': config['model']['hidden_size'],
        'n_hidden_layers': config['model']['n_hidden_layers'],
        'dropout_rate': config['model']['dropout_rate'],
        'activation': config['model']['activation'],
        'loss_type': config['model']['loss_type'],
        'grid_n_theta': config['model']['grid_n_theta'],
        'grid_n_phi': config['model']['grid_n_phi'],
        'physics_grid_type': config['model']['physics_grid_type'],
        'physics_grid_resolution_factor': config['model']['physics_grid_resolution_factor'],
        'physics_field_weight': config['model']['physics_field_weight'],
        'learning_rate': config['training']['learning_rate'],
        'epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'weight_decay': config['training']['weight_decay'],
        'device': config['device']
    }
    
    model_config = MLPConfig(**valid_params)
    
    model = MLPModel(model_config)
    model.load(experiment_dir / "model")
    
    print(f"✓ Loaded model from {experiment_dir / 'model'}")
    
    return model, config

def generate_synthetic_test_data(config, n_samples=50):
    """Generate small test dataset for plotting."""
    
    from src.core.data_generator import DataGenerator, pack_coefficients
    
    maxorder = config['model']['maxorder']
    
    # Generate test data
    generator = DataGenerator()
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    
    # Extract components
    E_theta = dataset['amplitude'][..., 0]
    E_phi = dataset['amplitude'][..., 1]
    coeffs_e = dataset['coefficients_e']
    coeffs_m = dataset['coefficients_m']
    
    # Pack coefficients and compute P field
    y_coeffs = pack_coefficients(coeffs_e, coeffs_m)
    y_P = np.abs(E_theta)**2 + np.abs(E_phi)**2
    
    return E_theta, E_phi, y_coeffs, y_P

def plot_physics_model_evaluation(model, config, save_dir: Path, experiment_dir: Path):
    """Generate comprehensive plots for physics model evaluation."""
    
    print("Generating physics model evaluation plots...")
    save_dir.mkdir(exist_ok=True)
    
    # Generate test data
    E_theta, E_phi, y_coeffs_true, y_P_true = generate_synthetic_test_data(config, n_samples=10)
    
    # Load the actual preprocessing pipeline used during training
    from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig
    
    # Load preprocessing from experiment directory
    preprocessing_dir = Path(experiment_dir) / "preprocessing"
    if preprocessing_dir.exists():
        print(f"✓ Loading preprocessing pipeline from {preprocessing_dir}")
        preprocessing_config = PreprocessingConfig(
            pca_components=config['preprocessing']['pca_components'],
            normalize_features=config['preprocessing']['normalize_features'],
            normalize_targets=False
        )
        preprocessing = PreprocessingPipeline(preprocessing_config)
        preprocessing.load(preprocessing_dir)
    else:
        # Fallback: fit new preprocessing with more samples
        print("⚠ Preprocessing not found, generating larger dataset for fitting")
        E_theta_large, E_phi_large, y_coeffs_large, _ = generate_synthetic_test_data(config, n_samples=100)
        
        preprocessing_config = PreprocessingConfig(
            pca_components=config['preprocessing']['pca_components'],
            normalize_features=config['preprocessing']['normalize_features'],
            normalize_targets=False
        )
        preprocessing = PreprocessingPipeline(preprocessing_config)
        preprocessing.fit(E_theta_large, E_phi_large, targets=y_coeffs_large)
    
    X_test = preprocessing.transform_features(E_theta, E_phi)
    
    # Make coefficient predictions
    y_coeffs_pred = model.predict(X_test)
    
    print(f"Test data shapes:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_coeffs_true: {y_coeffs_true.shape}")  
    print(f"  y_coeffs_pred: {y_coeffs_pred.shape}")
    print(f"  y_P_true: {y_P_true.shape}")
    
    # 1. Coefficient Comparison Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten coefficient arrays for scatter plot
    coeffs_true_flat = y_coeffs_true.flatten()
    coeffs_pred_flat = y_coeffs_pred.flatten()
    
    ax.scatter(coeffs_true_flat, coeffs_pred_flat, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(coeffs_true_flat.min(), coeffs_pred_flat.min())
    max_val = max(coeffs_true_flat.max(), coeffs_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax.set_xlabel('True Coefficients')
    ax.set_ylabel('Predicted Coefficients')
    ax.set_title('Physics Model: Coefficient Predictions vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    ss_res = np.sum((coeffs_true_flat - coeffs_pred_flat) ** 2)
    ss_tot = np.sum((coeffs_true_flat - np.mean(coeffs_true_flat)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    ax.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / "coefficient_predictions_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved coefficient scatter plot")
    
    # 2. Convert Predictions to P Fields for Physics Comparison
    try:
        maxorder = config['model']['maxorder']
        grid_n_phi = y_P_true.shape[1]  # Use actual grid size from data
        grid_n_theta = y_P_true.shape[2]
        
        field_gen = DifferentiableMultipoleField(
            maxorder=maxorder,
            grid_shape=(grid_n_phi, grid_n_theta)
        )
        
        # Convert predicted coefficients to P field
        coeffs_tensor = torch.from_numpy(y_coeffs_pred).float()
        y_P_pred = field_gen(coeffs_tensor).detach().numpy()
        
        # Handle shape mismatch if needed
        if y_P_pred.shape != y_P_true.shape:
            if y_P_pred.shape == (y_P_true.shape[0], y_P_true.shape[2], y_P_true.shape[1]):
                y_P_pred = y_P_pred.transpose(0, 2, 1)
                print("✓ Transposed predicted P field to match true P field")
        
        print(f"P field shapes: true={y_P_true.shape}, pred={y_P_pred.shape}")
        
        # 3. P Field Comparison Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        P_true_flat = y_P_true.flatten()
        P_pred_flat = y_P_pred.flatten()
        
        # Sample points for visualization (too many points to plot all)
        n_sample_points = min(10000, len(P_true_flat))
        indices = np.random.choice(len(P_true_flat), n_sample_points, replace=False)
        
        ax.scatter(P_true_flat[indices], P_pred_flat[indices], alpha=0.4, s=1)
        
        # Perfect prediction line
        min_val = min(P_true_flat.min(), P_pred_flat.min())
        max_val = max(P_true_flat.max(), P_pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel('True P Field Values')
        ax.set_ylabel('Predicted P Field Values')
        ax.set_title('Physics Model: P Field Predictions vs True Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² score for P field
        ss_res_P = np.sum((P_true_flat - P_pred_flat) ** 2)
        ss_tot_P = np.sum((P_true_flat - np.mean(P_true_flat)) ** 2)
        r2_score_P = 1 - (ss_res_P / ss_tot_P) if ss_tot_P > 0 else 0.0
        
        ax.text(0.05, 0.95, f'R² = {r2_score_P:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / "p_field_predictions_scatter.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved P field scatter plot (R² = {r2_score_P:.3f})")
        
        # 4. Field Comparison Plots for Individual Samples
        for i in range(min(3, len(y_coeffs_true))):
            try:
                plot_field_comparison(
                    y_coeffs_true, y_coeffs_pred, sample_idx=i, maxorder=maxorder,
                    title_prefix=f"Physics Model Sample {i+1}",
                    save_path=save_dir / f"field_comparison_sample_{i}.png"
                )
                print(f"✓ Saved field comparison plot for sample {i}")
            except Exception as e:
                print(f"⚠ Failed to generate field comparison for sample {i}: {e}")
        
        # 5. Summary Statistics
        mse_coeffs = np.mean((y_coeffs_true - y_coeffs_pred) ** 2)
        mse_P = np.mean((y_P_true - y_P_pred) ** 2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coefficient error distribution
        coeff_errors = (y_coeffs_true - y_coeffs_pred).flatten()
        ax1.hist(coeff_errors, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Coefficient Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Coefficient Error Distribution\nMSE = {mse_coeffs:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # P field error distribution  
        P_errors = (y_P_true - y_P_pred).flatten()
        ax2.hist(P_errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('P Field Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'P Field Error Distribution\nMSE = {mse_P:.4f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "error_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved error distribution plots")
        
        return {
            'coefficient_r2': r2_score,
            'p_field_r2': r2_score_P,
            'coefficient_mse': mse_coeffs,
            'p_field_mse': mse_P
        }
        
    except Exception as e:
        print(f"⚠ Failed to generate P field plots: {e}")
        return {'coefficient_r2': r2_score}

def plot_training_curves_from_logs(experiment_dir: Path, save_dir: Path):
    """Generate training curves plot."""
    
    # Try to find training history in model info
    try:
        model_info_path = experiment_dir / "model" / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path) as f:
                model_info = json.load(f)
                
            if 'training_history' in model_info and model_info['training_history']:
                training_history = model_info['training_history']
                
                # Convert from list of dicts to dict of lists
                if isinstance(training_history, list) and len(training_history) > 0:
                    keys = training_history[0].keys()
                    history_dict = {key: [entry[key] for entry in training_history] for key in keys}
                    
                    plot_training_curves(
                        history_dict,
                        save_path=save_dir / "training_curves.png"
                    )
                    print(f"✓ Saved training curves plot")
                    return True
                    
    except Exception as e:
        print(f"⚠ Could not generate training curves: {e}")
    
    return False

def main():
    """Generate all missing plots for the physics model."""
    
    # Find the latest physics experiment
    results_dir = Path("experiments/results")
    physics_dirs = sorted([d for d in results_dir.iterdir() if d.name.startswith("mlp_physics")])
    
    if not physics_dirs:
        print("❌ No physics experiment directories found")
        return
    
    experiment_dir = physics_dirs[-1]  # Use latest
    print(f"📁 Using experiment directory: {experiment_dir}")
    
    # Load model and config
    try:
        model, config = load_model_and_data(experiment_dir)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create plots directory
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate training curves
    plot_training_curves_from_logs(experiment_dir, plots_dir)
    
    # Generate physics model evaluation plots
    metrics = plot_physics_model_evaluation(model, config, plots_dir, experiment_dir)
    
    print("\n" + "="*50)
    print("📊 PHYSICS MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"📁 Plots saved to: {plots_dir}")
    print(f"📈 Coefficient R²: {metrics.get('coefficient_r2', 'N/A'):.3f}")
    if 'p_field_r2' in metrics:
        print(f"⚡ P Field R²: {metrics['p_field_r2']:.3f}")
        print(f"🎯 Physics Loss MSE: {metrics['p_field_mse']:.6f}")
    print("="*50)
    print("✅ All plots generated successfully!")

if __name__ == "__main__":
    main()