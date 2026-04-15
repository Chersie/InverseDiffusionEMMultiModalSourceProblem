# Pipeline Overview

Numerical objects and tensor shapes reference: `docs/numerical_shapes.md`.

## Synthetic/Test Pipeline

1. Generate multipole library (`fast` or `slow` mode).
2. Generate synthetic `Fields.txt` from latin-square coefficients.
3. Decompose fields into multipole coefficients.
4. Visualize multipoles in 2D/3D.

Entrypoint: `python run_pipeline_test.py` (delegates to `src/cli/run_pipeline_test.py`).

## Inverse Mie Pipeline

1. Convert pictures to tables (when picture inputs exist).
2. Convert tables to `Fields.txt`.
3. Decompose fields with selected multipole library.
4. Run inverse Mie fitting from `Results_<field_stem>.txt`.

Entrypoint: `python run_inverse_mie_pipeline.py` (delegates to `src/cli/run_inverse_mie_pipeline.py`).

## Validation

- `python src/cli/validate_grid_files.py`
  - checks row counts and angular row ordering for `Fields.txt` and a representative library mode file.

## Tracking

Both CLI entrypoints start an MLflow run through `models/tracking/mlflow_utils.py`.
If MLflow is not installed, tracking degrades gracefully to no-op logging so execution remains non-breaking.

## ML Baseline Pipeline

Goal: learn `P_UT -> (a_E, a_M)`, then reconstruct `E^` and `P^` and evaluate `P` vs `P^`.

1. Build or load synthetic dataset (`N` samples):
   - `E_UT` generated from latin-square-style multipole coefficients.
   - `P_UT = |E_theta|^2 + |E_phi|^2`.
   - labels from projection: `a_E(l,m)`, `a_M(l,m)` (packed real/imag).
2. Train baseline regressor:
   - randomized PCA on power features + multi-output ridge.
3. Predict coefficients on test split.
4. Reconstruct `E^` from predicted coefficients using multipole basis.
5. Compute `P^` and evaluate `MSE/MAE/relL2` against `P_UT`.

Entrypoint:

- `python -m src.cli.run_train_baseline --n-samples 10000 --maxorder 15 --seed 42`
- Frozen-physics decoder mode (predictor trainable, decoder constant):
  - `python -m src.cli.run_train_baseline --trainer physics --n-samples 10000 --maxorder 15 --seed 42 --epochs 40`

Outputs:

- Dataset: `data/ml/datasets/baseline_L{L}_N{N}_seed{seed}/`
- Splits: `data/ml/splits/baseline_L{L}_N{N}_seed{seed}.npz`
- Basis cache: `data/ml/features/basis_L{L}.npz`
- Trained artifact: `models/artifacts/baseline_L{L}_N{N}_seed{seed}/`

Notes:

- `--trainer ridge` (default): closed-form baseline.
- `--trainer physics`: keeps `(a_E,a_M) -> E^ -> P^` decoder fixed and optimizes predictor with combined coefficient + power losses.
- Physics trainer requires PyTorch installed in your environment.
