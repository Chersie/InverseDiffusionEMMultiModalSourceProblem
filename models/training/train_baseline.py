from __future__ import annotations

import argparse

from models.training.baseline_pipeline import BaselineConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model: P_UT -> multipole coefficients.")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--maxorder", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--trainer", choices=["ridge", "physics"], default="ridge")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--pca-oversample", type=int, default=16)
    parser.add_argument("--pca-iterations", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--coeff-loss-weight", type=float, default=1.0)
    parser.add_argument("--power-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--amplitude-loss-weight",
        type=float,
        default=1.0,
        help="Physics trainer: weight on (log P_pred - log P_true)² for area-weighted total power P.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        n_samples=args.n_samples,
        maxorder=args.maxorder,
        seed=args.seed,
        trainer=args.trainer,
        ridge_alpha=args.ridge_alpha,
        pca_components=args.pca_components,
        pca_oversample=args.pca_oversample,
        pca_iterations=args.pca_iterations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        coeff_loss_weight=args.coeff_loss_weight,
        power_loss_weight=args.power_loss_weight,
        amplitude_loss_weight=args.amplitude_loss_weight,
        device=args.device,
        rebuild_dataset=args.rebuild_dataset,
    )
    metrics = train_and_evaluate(config)
    print("Baseline training complete.")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()
