#!/usr/bin/env python3
"""
CLI entry point for the MLP physics trainer (v0 neural model).

Usage
-----
    python -m src.cli.run_train_mlp --help

Minimal example:
    python -m src.cli.run_train_mlp \\
        --n-samples 10000 --maxorder 5 \\
        --hidden-size 512 --n-hidden-layers 2 --dropout 0.1 \\
        --epochs 100 --learning-rate 1e-3 --pca-components 256

Compare against the baseline:
    python -m src.cli.run_train_baseline --trainer ridge    # linear, coeff MSE
    python -m src.cli.run_train_baseline --trainer physics  # linear, power loss
    python -m src.cli.run_train_mlp                         # MLP,    power loss  ← this
"""
from __future__ import annotations

import argparse

from models.training.mlp_pipeline import MlpConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP physics model: [X_theta || X_phi] → PCA → MLP → coefficients → power loss."
    )
    # Dataset
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
    # PCA
    parser.add_argument("--pca-components", type=int, default=256,
                        help="Number of PCA components (default: 256)")
    parser.add_argument("--pca-oversample", type=int, default=16)
    parser.add_argument("--pca-iterations", type=int, default=0)
    # MLP architecture
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Width of each MLP hidden layer (default: 512)")
    parser.add_argument("--n-hidden-layers", type=int, default=2,
                        help="Number of hidden layers in the MLP (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability between hidden layers (default: 0.1)")
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--amplitude-loss-weight",
        type=float,
        default=1.0,
        help="Weight on log area-weighted total power vs truth (same as physics baseline).",
    )
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device string, e.g. 'cpu' or 'cuda' (default: cpu)")
    # MLflow logging during training
    parser.add_argument("--val-log-frequency", type=int, default=10,
                        help="Log validation metrics every N epochs (default: 10)")
    parser.add_argument("--detailed-metrics-frequency", type=int, default=20,
                        help="Log detailed physics metrics every N epochs (default: 20)")
    # Memory optimization
    parser.add_argument("--disable-memory-efficient-pca", action="store_true",
                        help="Disable memory-efficient PCA (use standard concatenation approach)")
    parser.add_argument("--transform-batch-size", type=int, default=0,
                        help="Batch size for PCA transforms (0=auto: 4-16 based on dataset size)")
    parser.add_argument("--ultra-conservative", action="store_true",
                        help="Use ultra-small transform batches (force batch size = 2) for maximum memory safety")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MlpConfig(
        n_samples=args.n_samples,
        maxorder=args.maxorder,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pca_components=args.pca_components,
        pca_oversample=args.pca_oversample,
        pca_iterations=args.pca_iterations,
        hidden_size=args.hidden_size,
        n_hidden_layers=args.n_hidden_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        rebuild_dataset=args.rebuild_dataset,
        amplitude_loss_weight=args.amplitude_loss_weight,
        val_log_frequency=args.val_log_frequency,
        detailed_metrics_frequency=args.detailed_metrics_frequency,
        use_memory_efficient_pca=not args.disable_memory_efficient_pca,
        transform_batch_size=args.transform_batch_size,
        ultra_conservative=args.ultra_conservative,
    )
    print(f"MLP config: L={config.maxorder}, N={config.n_samples}, "
          f"hidden={config.hidden_size}×{config.n_hidden_layers}, "
          f"pca={config.pca_components}, epochs={config.epochs}")
    metrics = train_and_evaluate(config)
    print("MLP training finished.")
    for key, value in metrics.items():
        try:
            # Try to format as scientific notation for numeric values
            print(f"  {key}: {value:.6e}")
        except (ValueError, TypeError):
            # For non-numeric values, print as string
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
