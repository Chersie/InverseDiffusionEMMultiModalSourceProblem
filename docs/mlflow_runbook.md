# MLflow runbook

Backed by R3 in [research/framework-rebuild/manifest.md](../research/framework-rebuild/manifest.md). The framework targets MLflow 3.x.

## Start the local server

```bash
./scripts/start_mlflow_server.sh
```

Defaults: `sqlite:///mlflow.db` for the backend, `./mlartifacts` for the artifact root, `http://127.0.0.1:5000` for the UI. All three are overridable via env vars (`MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACT_ROOT`, `MLFLOW_HOST`, `MLFLOW_PORT`).

## What gets logged per run

- All Hydra params, flattened (e.g. `model.cfg.hidden_size`, `optimiser.lr`).
- The full resolved config as a YAML artifact (`config.yaml`).
- Training metrics every N steps: `train/loss`, `train/lr`, `train/grad_norm`, `perf/step_time_ms`, `perf/batch_time_ms`, plus the per-component breakdown of multi-term losses.
- Validation metrics every N epochs: `val/loss`, `val/coef_mse`.
- End-of-run report metrics: `report/coef_mse`, `report/coef_r2`, `report/coef_mse_amb_aware`, `report/field_mse_w`, `report/field_nrmse_w`.
- End-of-run figures: `coef_histograms.pdf`, `coef_scatter.pdf`, `per_l_breakdown.pdf`, `field_comparison.pdf`, `pca_explained_variance.pdf`.
- Dataset metadata via `mlflow.data.from_numpy(...)` + `log_input(...)` (training, validation contexts; metadata only — bytes are not stored on the server).

## Sweep runs

The `mpinv-sweep` CLI opens a parent run and creates one nested child run per Optuna trial, tagged with `mpinv.run_kind=hpo_trial` and `mpinv.trial_number=<int>`. The parent run additionally logs `sweep_summary.json` with the best trial and the search-space spec.

## Model registry

Stages are deprecated in MLflow 2.9+; we use **aliases** instead:

```python
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("mpinv-mlp", "production", version="3")
# Reference at load time:
mlflow.pyfunc.load_model("models:/mpinv-mlp@production")
```

## Cleanup

```bash
mlflow gc --older-than 30d   # permanently free disk after deletes
```
