# Getting started

## Install

```bash
cd diplom_clean
uv venv --python 3.12
uv sync --extra dev --extra cv
```

## First training run

```bash
uv run mpinv-train tracking=mlflow_off
```

This trains an MLP for 3 epochs on a tiny synthetic dataset (24×12 grid, L=4) and writes a per-run report (loss curves, scatter, residuals) under `outputs/<timestamp>/report/`.

## With MLflow

```bash
./scripts/start_mlflow_server.sh &     # http://127.0.0.1:5000
uv run mpinv-train
```

Open the UI in a browser; the run will appear under the `mpinv` experiment.

## Validate the physics layer

```bash
uv run mpinv-validate-physics
```

## HPO sweep

```bash
uv run mpinv-sweep sweep.n_trials=8 tracking=mlflow_off
```

## Run the full test suite

```bash
uv run pytest tests
```

Expect ~70 tests, ~25s on a laptop CPU.
