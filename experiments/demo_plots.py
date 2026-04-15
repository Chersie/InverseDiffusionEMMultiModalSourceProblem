#!/usr/bin/env python3
"""
Demo script showing the visualization capabilities of the ML pipeline.

This creates example plots demonstrating the various visualization functions
available for experiment analysis.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from utils.plotting import (
    plot_prediction_scatter,
    plot_coefficient_comparison,
    plot_training_curves,
    plot_loss_distribution,
    create_experiment_summary_plot
)

def create_demo_plots():
    """Create demonstration plots showing the visualization capabilities."""
    print("🎨 Creating demonstration plots...")
    
    demo_dir = Path("experiments/results/demo_plots")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training curves
    print("📈 1. Training curves...")
    training_history = {
        'train_loss': [2.1, 1.5, 1.2, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.37, 0.35, 0.33, 0.32, 0.31, 0.30],
        'val_loss': [2.2, 1.6, 1.3, 1.0, 0.8, 0.72, 0.65, 0.6, 0.58, 0.56, 0.54, 0.53, 0.52, 0.51, 0.50]
    }
    
    plot_training_curves(
        training_history,
        save_path=demo_dir / "1_training_curves.png"
    )
    
    # 2. Prediction scatter plot
    print("📊 2. Prediction scatter plot...")
    n_samples = 500
    n_features = 50
    y_true = np.random.randn(n_samples, n_features)
    # Add some correlation with noise
    y_pred = 0.85 * y_true + 0.15 * np.random.randn(n_samples, n_features)
    
    plot_prediction_scatter(
        y_true, y_pred,
        title="MLP Model: Predictions vs True Values",
        save_path=demo_dir / "2_prediction_scatter.png"
    )
    
    # 3. Coefficient comparison
    print("📉 3. Coefficient comparison...")
    maxorder = 5
    n_modes = maxorder * (maxorder + 2)
    coeffs_true = np.random.randn(2 * n_modes * 2)  # E&M, real&imag
    coeffs_pred = coeffs_true + 0.1 * np.random.randn(2 * n_modes * 2)
    
    plot_coefficient_comparison(
        coeffs_true[np.newaxis, :], coeffs_pred[np.newaxis, :], maxorder,
        sample_idx=0,
        save_path=demo_dir / "3_coefficient_comparison.png"
    )
    
    # 4. Loss distribution
    print("📊 4. Loss distribution...")
    losses = np.random.lognormal(mean=-1, sigma=0.5, size=1000)
    
    plot_loss_distribution(
        losses,
        title="Test Loss Distribution Across Samples",
        save_path=demo_dir / "4_loss_distribution.png"
    )
    
    # 5. Comprehensive experiment summary
    print("📋 5. Experiment summary...")
    config = {
        'experiment': {
            'name': 'electromagnetic_mlp_demo',
            'description': 'Demonstration of electromagnetic multipole MLP training'
        },
        'model': {
            'type': 'mlp',
            'maxorder': 5,
            'hidden_size': 256,
            'n_hidden_layers': 3
        },
        'training': {
            'n_samples': 1000,
            'epochs': 15,
            'batch_size': 64,
            'learning_rate': 0.001
        }
    }
    
    test_metrics = {
        'test_mse': 0.130335,
        'test_r2': 0.5292,
        'test_mae': 0.285,
        'test_rmse': 0.361
    }
    
    create_experiment_summary_plot(
        config, training_history, test_metrics,
        save_path=demo_dir / "5_experiment_summary.png"
    )
    
    # 6. Training progress with different scenarios
    print("📈 6. Different training scenarios...")
    
    # Good training
    good_history = {
        'train_loss': [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.10, 0.09, 0.08, 0.075],
        'val_loss': [1.1, 0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.19, 0.18, 0.17]
    }
    plot_training_curves(good_history, save_path=demo_dir / "6a_good_training.png")
    
    # Overfitting scenario
    overfit_history = {
        'train_loss': [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
        'val_loss': [1.1, 0.6, 0.4, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    }
    plot_training_curves(overfit_history, save_path=demo_dir / "6b_overfitting_training.png")
    
    print(f"\n✅ Demo plots created successfully!")
    print(f"📁 Location: {demo_dir}")
    
    # List all created files
    plot_files = list(demo_dir.glob("*.png"))
    print(f"📊 Generated {len(plot_files)} plots:")
    for plot_file in sorted(plot_files):
        print(f"   • {plot_file.name}")
    
    return demo_dir

def show_plot_capabilities():
    """Show what plotting capabilities are available."""
    print("\n🎨 Available Plotting Functions:")
    print("=" * 50)
    
    capabilities = [
        ("Training Curves", "Plot training and validation loss over epochs"),
        ("Prediction Scatter", "Compare predicted vs true values with R² score"),
        ("Coefficient Comparison", "Bar charts comparing E/M multipole coefficients"),
        ("Loss Distribution", "Histogram of loss values across test samples"),
        ("Experiment Summary", "Comprehensive overview with config and metrics"),
        ("Custom Visualizations", "Field patterns, power maps, and comparisons")
    ]
    
    for name, description in capabilities:
        print(f"📊 {name:20s}: {description}")
    
    print("\n🔧 Integration Points:")
    print("• Training scripts automatically generate plots")
    print("• Experiment results include plots/ directory")
    print("• Manual plot generation with generate_plots.py")
    print("• Custom visualizations in models/visualization/")

if __name__ == "__main__":
    try:
        show_plot_capabilities()
        demo_dir = create_demo_plots()
        
        print(f"\n🎯 To add plots to your experiments:")
        print("1. Use experiments/train_simple.py (includes basic plots)")
        print("2. Use experiments/scripts/train_mlp.py (full experiment plots)")
        print("3. Run experiments/generate_plots.py for existing experiments")
        print("4. Customize using utils/plotting.py functions")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()