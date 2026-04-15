#!/usr/bin/env python3
"""
CLI entry point for physics-aware neural networks.

Usage
-----
    python -m src.cli.run_train_physics_aware --help

Examples
--------
# Multipole-aware network (respects spherical harmonic structure)
python -m src.cli.run_train_physics_aware \
  --model-type multipole_aware \
  --n-samples 10000 --maxorder 5 \
  --hidden-size 512 --epochs 100

# Energy-conserving network (enforces power conservation)  
python -m src.cli.run_train_physics_aware \
  --model-type energy_conserving \
  --energy-conservation-weight 2.0 \
  --n-samples 10000 --maxorder 5

# Hybrid physics-ML model (analytical baseline + learned corrections)
python -m src.cli.run_train_physics_aware \
  --model-type hybrid \
  --n-samples 10000 --maxorder 5 \
  --hidden-size 256 --epochs 150

# Symmetry-aware network (respects EM reciprocity)
python -m src.cli.run_train_physics_aware \
  --model-type symmetry_aware \
  --symmetry-regularization 0.5 \
  --n-samples 10000 --maxorder 5
"""
from __future__ import annotations

import argparse

from models.training.physics_aware_pipeline import PhysicsAwareConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train physics-aware neural networks: multipole-structured, energy-conserving, hybrid, or symmetry-aware models."
    )
    
    # Dataset parameters
    parser.add_argument("--n-samples", type=int, default=10_000,
                        help="Number of synthetic samples to generate (default: 10000)")
    parser.add_argument("--maxorder", type=int, default=15,
                        help="Maximum multipole order L (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset generation and splits (default: 42)")
    parser.add_argument("--rebuild-dataset", action="store_true",
                        help="Force dataset regeneration even if cached files exist")
    
    # Splits
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    
    # PCA preprocessing
    parser.add_argument("--pca-components", type=int, default=256,
                        help="Number of PCA components (default: 256)")
    parser.add_argument("--pca-oversample", type=int, default=16)
    parser.add_argument("--pca-iterations", type=int, default=0)
    
    # Physics-aware architecture selection
    parser.add_argument("--model-type", 
                        choices=["multipole_aware", "energy_conserving", "hybrid", "symmetry_aware"],
                        default="multipole_aware",
                        help="Physics-aware model architecture (default: multipole_aware)")
    
    # Standard neural network parameters  
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Width of hidden layers (default: 512)")
    parser.add_argument("--n-hidden-layers", type=int, default=2,
                        help="Number of hidden layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability (default: 0.1)")
    
    # Physics-aware constraints
    parser.add_argument("--energy-conservation-weight", type=float, default=1.0,
                        help="Weight for energy conservation penalty (default: 1.0)")
    parser.add_argument("--symmetry-regularization", type=float, default=0.1,
                        help="Weight for symmetry regularization (default: 0.1)")
    parser.add_argument("--amplitude-loss-weight", type=float, default=1.0,
                        help="Weight for amplitude loss term (default: 1.0)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device string, e.g. 'cpu' or 'cuda' (default: cpu)")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    config = PhysicsAwareConfig(
        n_samples=args.n_samples,
        maxorder=args.maxorder,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pca_components=args.pca_components,
        pca_oversample=args.pca_oversample,
        pca_iterations=args.pca_iterations,
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        n_hidden_layers=args.n_hidden_layers,
        dropout=args.dropout,
        energy_conservation_weight=args.energy_conservation_weight,
        symmetry_regularization=args.symmetry_regularization,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        rebuild_dataset=args.rebuild_dataset,
        amplitude_loss_weight=args.amplitude_loss_weight,
    )
    
    print(f"Physics-Aware Model: {config.model_type}")
    print(f"Config: L={config.maxorder}, N={config.n_samples}, "
          f"hidden={config.hidden_size}×{config.n_hidden_layers}, "
          f"epochs={config.epochs}")
    
    if config.model_type == "energy_conserving":
        print(f"Energy conservation weight: {config.energy_conservation_weight}")
    elif config.model_type == "symmetry_aware":  
        print(f"Symmetry regularization: {config.symmetry_regularization}")
    elif config.model_type == "hybrid":
        print("Hybrid model: analytical baseline + learned corrections")
    elif config.model_type == "multipole_aware":
        print("Multipole-aware: separate processing per spherical harmonic order")
    
    metrics = train_and_evaluate(config)
    
    print(f"\nPhysics-aware training finished ({config.model_type}).")
    print("Key metrics:")
    
    # Standard metrics
    for key in ["coeff_mse_test", "p_mse", "weighted_mse", "beam_pointing_error_deg"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6e}")
    
    # Physics-aware metrics
    print("\nPolarization metrics:")
    for key in ["theta_mse_power_weighted", "phi_mse_power_weighted", "polarization_correlation"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6e}")
    
    # Coefficient validation
    print("\nCoefficient validation:")
    for key in ["coeff_mag_error_rms", "coeff_phase_error_rms_deg", "coeff_complex_error_rms"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6e}")


if __name__ == "__main__":
    main()