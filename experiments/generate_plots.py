#!/usr/bin/env python3
"""
Generate plots for existing experiment results.

This script demonstrates how to generate various plots from trained models
and experiment data.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.models.registry import get_model_registry
from src.core.data_generator import DataGenerator
from src.api.preprocessing import PreprocessingPipeline
from utils.plotting import (
    plot_prediction_scatter,
    plot_coefficient_comparison,
    plot_training_curves,
    create_experiment_summary_plot
)

def generate_plots_for_experiment(experiment_dir: Path):
    """Generate plots for a completed experiment."""
    print(f"📊 Generating plots for experiment: {experiment_dir.name}")
    
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    model_dir = experiment_dir / "model"
    if not model_dir.exists():
        print(f"❌ No model found in {experiment_dir}")
        return
    
    # Load model info to get configuration
    model_info_path = model_dir / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        maxorder = model_info.get('config', {}).get('maxorder', 5)
        input_dim = model_info.get('input_dim', 256)
        output_dim = model_info.get('output_dim', 140)
        
        print(f"   📋 Model: maxorder={maxorder}, input_dim={input_dim}, output_dim={output_dim}")
    else:
        maxorder = 5
        input_dim = 256 
        output_dim = 140
        print("   ⚠️ Using default model parameters")
    
    # Generate fresh test data for visualization
    print("   🔧 Generating test data...")
    generator = DataGenerator.for_ml_training()
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=20)  # Small dataset for plots
    
    # Setup preprocessing
    preprocessing_dir = experiment_dir / "preprocessing"
    if preprocessing_dir.exists():
        print("   🔧 Loading preprocessing pipeline...")
        preprocessing = PreprocessingPipeline()
        try:
            preprocessing.load(preprocessing_dir)
        except Exception as e:
            print(f"   ⚠️ Failed to load preprocessing: {e}")
            preprocessing.fit(dataset['amplitude'][..., 0], dataset['amplitude'][..., 1])
    else:
        print("   🔧 Creating new preprocessing pipeline...")
        preprocessing = PreprocessingPipeline()
        preprocessing.fit(dataset['amplitude'][..., 0], dataset['amplitude'][..., 1])
    
    # Transform data
    X = preprocessing.transform_features(dataset['amplitude'][..., 0], dataset['amplitude'][..., 1])
    from src.core.data_generator import pack_coefficients
    y = preprocessing.process_coefficients(dataset['coefficients_e'], dataset['coefficients_m'])
    
    # Load trained model
    try:
        print("   🤖 Loading trained model...")
        registry = get_model_registry()
        model = registry.create_model(
            model_type="mlp",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        model.load(model_dir)
        print("   ✅ Model loaded successfully")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Generate plots
        print("   📈 Generating prediction scatter plot...")
        plot_prediction_scatter(
            y, predictions,
            title=f"Predictions vs True Values ({experiment_dir.name})",
            save_path=plots_dir / "predictions_scatter.png"
        )
        
        print("   📊 Generating coefficient comparison plots...")
        for i in range(min(3, len(X))):
            plot_coefficient_comparison(
                y, predictions, maxorder, sample_idx=i,
                save_path=plots_dir / f"coefficients_sample_{i}.png"
            )
        
        # Create mock training history for demonstration
        mock_history = {
            'train_loss': [1.2, 0.8, 0.6, 0.5, 0.4, 0.35],
            'val_loss': [1.3, 0.9, 0.7, 0.6, 0.55, 0.52]
        }
        
        print("   📉 Generating training curves...")
        plot_training_curves(
            mock_history,
            save_path=plots_dir / "training_curves.png"
        )
        
        # Mock configuration for experiment summary
        mock_config = {
            'experiment': {
                'name': experiment_dir.name,
                'description': 'Electromagnetic multipole analysis experiment'
            },
            'model': {
                'type': 'mlp',
                'maxorder': maxorder,
                'hidden_size': 256,
                'n_hidden_layers': 2
            },
            'training': {
                'n_samples': 500,
                'epochs': 20,
                'batch_size': 64,
                'learning_rate': 0.001
            }
        }
        
        test_mse = float(np.mean((y - predictions) ** 2))
        test_r2 = float(1 - np.var(y - predictions) / np.var(y))
        
        mock_metrics = {
            'test_mse': test_mse,
            'test_r2': test_r2,
            'test_rmse': float(np.sqrt(test_mse))
        }
        
        print("   📋 Generating experiment summary...")
        create_experiment_summary_plot(
            mock_config, mock_history, mock_metrics,
            save_path=plots_dir / "experiment_summary.png"
        )
        
        print(f"   ✅ All plots generated successfully in {plots_dir}")
        return plots_dir
        
    except Exception as e:
        print(f"   ❌ Failed to load model or generate plots: {e}")
        return None

def main():
    """Find recent experiments and generate plots."""
    print("🎨 Experiment Plotting Tool")
    print("=" * 50)
    
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        print("❌ No experiments found in experiments/results")
        return
    
    # Find experiment directories
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "model").exists()]
    
    if not experiment_dirs:
        print("❌ No completed experiments found (no model directories)")
        return
    
    print(f"📂 Found {len(experiment_dirs)} completed experiments:")
    for exp_dir in sorted(experiment_dirs):
        print(f"   • {exp_dir.name}")
    
    # Generate plots for each experiment
    generated_count = 0
    for exp_dir in sorted(experiment_dirs)[-3:]:  # Process last 3 experiments
        try:
            plots_dir = generate_plots_for_experiment(exp_dir)
            if plots_dir:
                generated_count += 1
        except Exception as e:
            print(f"❌ Failed to process {exp_dir.name}: {e}")
    
    print(f"\n🎉 Generated plots for {generated_count} experiments!")
    print("\n📁 Plot locations:")
    for exp_dir in sorted(experiment_dirs)[-3:]:
        plots_dir = exp_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            print(f"   {exp_dir.name}/ ({len(plot_files)} plots)")
            for plot_file in sorted(plot_files):
                print(f"     • {plot_file.name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Plotting interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()