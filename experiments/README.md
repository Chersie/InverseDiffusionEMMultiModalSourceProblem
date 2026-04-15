# Experiments

This directory contains organized experiments for training ML models for electromagnetic multipole analysis.

## Directory Structure

```
experiments/
├── configs/           # Configuration files for different experiments
├── scripts/          # Training scripts and utilities
├── results/         # Training results and artifacts
├── logs/           # Training logs and metrics
└── notebooks/     # Jupyter notebooks for analysis (optional)
```

## Quick Start

### 1. Train a Basic MLP Model
```bash
cd experiments
python scripts/train_mlp.py --config configs/mlp_basic.yaml
```

### 2. Train with Custom Configuration
```bash
python scripts/train_mlp.py --config configs/mlp_large.yaml --maxorder 10
```

### 3. Resume Training
```bash
python scripts/train_mlp.py --config configs/mlp_basic.yaml --resume results/mlp_basic_20240414/
```

## Configuration Files

- `configs/mlp_basic.yaml` - Basic MLP configuration for testing
- `configs/mlp_large.yaml` - Large MLP for production training
- `configs/mlp_physics.yaml` - Physics-aware MLP with specialized losses
- `configs/baseline_comparison.yaml` - Baseline models for comparison

## Results

Training results are automatically organized by:
- Timestamp: `results/experiment_name_YYYYMMDD_HHMMSS/`
- Configuration: Model config, hyperparameters, and training settings
- Artifacts: Trained models, preprocessing pipelines, and metrics
- Plots: Training curves, validation metrics, and analysis plots

## Experiment Tracking

All experiments are automatically tracked with MLflow:
- Metrics: Training/validation losses, physics-aware metrics
- Parameters: All hyperparameters and configuration settings
- Artifacts: Model checkpoints and preprocessing pipelines
- Plots: Training visualizations and analysis

## Best Practices

1. **Use configuration files** for reproducible experiments
2. **Tag experiments** with meaningful names and descriptions
3. **Track hyperparameters** systematically
4. **Save intermediate checkpoints** for long training runs
5. **Compare baselines** before trying complex models