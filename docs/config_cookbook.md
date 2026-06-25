# Hydra config cookbook

## Default training run

```bash
uv run mpinv-train
```

Equivalent to: tiny grid (`n_phi=24, n_theta=12, L=4`), MLP small, coefficient MSE, AdamW, 3 epochs, MLflow local server at `http://127.0.0.1:5000`.

## Run without MLflow

```bash
uv run mpinv-train tracking=mlflow_off
```

## Switch model

```bash
uv run mpinv-train model=mlp_pyramid
uv run mpinv-train model=mlp_bottleneck
uv run mpinv-train model=mlp_residual
uv run mpinv-train model=linear            # closed-form-equivalent baseline
```

## Switch loss

```bash
uv run mpinv-train loss=coef_mse           # MSE in packed-coefficient space
uv run mpinv-train loss=physics_power      # sin θ-weighted MSE in power-pattern space
```

## Train on the project grid (full L=15)

```bash
uv run mpinv-train data=synthetic_l15 model=mlp_pyramid trainer=default
```

Beware: the first run computes the L=15 VSH basis (~250 MB) and caches it under `data/cache/`. Subsequent runs reuse the cache.

## Override hyperparameters

```bash
uv run mpinv-train model.cfg.hidden_size=512 optimiser.lr=3e-4 trainer.max_epochs=100
```

## Composite features (PCA + FFT radial + SH power)

```bash
uv run mpinv-train features=pca_cv
```

## HPO sweep with Optuna 4.x

```bash
uv run mpinv-sweep
uv run mpinv-sweep sweep.n_trials=32 sweep.sampler=tpe
uv run mpinv-sweep sweep.storage=sqlite:///optuna.db sweep.load_if_exists=true
```

Each trial logs as a child run nested under the parent sweep run in MLflow.

## Pre-generate a streaming corpus

```bash
uv run mpinv-generate-data --output-dir data/processed/train --n-total 16384 --grid full --l-max 15
```

The shards are named with a per-process, per-millisecond token to avoid collisions during parallel HPO.

## Validate the physics layer

```bash
uv run mpinv-validate-physics --n-phi 24 --n-theta 12 --l-max 4
uv run mpinv-validate-physics --n-phi 360 --n-theta 179 --theta-start 1 --theta-end 179 --l-max 15
```
