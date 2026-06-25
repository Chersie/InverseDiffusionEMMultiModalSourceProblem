# AGENTS.md

Cross-tool conventions for any agent working in this repository.

## Tooling

- Python is pinned to `3.12.*` in [pyproject.toml](pyproject.toml).
- The package manager is `uv`. Never run `pip install` directly; use `uv add ...` or `uv sync`.
- Lint and format with `ruff`. Type-check with `ty`. Both configs live in the repo root.
- Tests are `pytest`; markers `slow`, `gpu`, `integration`, `physics`.

## Architectural invariants

- **One canonical angular-grid layout**: `(B, n_theta=179, n_phi=360)` everywhere torch-harmonics is involved. Numpy/einsum synthesis uses `(B, n_phi=360, n_theta=179)` and converts at a single boundary.
- **Phase units**: radians internally; degrees only in the seven-column real-antenna file format and converted in `data/real_antenna_loader.py`.
- **Coefficient packing**: `[Re a^E, Im a^E, Re a^M, Im a^M]`, length `4K=1020` for `L=15`, `K=255`. Defined once in `core/packing.py`.
- **One registry per concept**: `MODELS`, `LOSSES`, `FEATURE_EXTRACTORS`, `METRICS`, `CALLBACKS`. Each is a `dict[str, type[...]]` defined exactly once. Importing duplicates is a bug.
- **No silent shape changes** inside losses. Assert; do not resize.
- **No PyTorch Lightning**. Custom callback-driven `Trainer` per the practice.pdf rationale.
- **No `mlflow.pytorch.autolog`**. Explicit logging only.

## Hydra invariants

- All configs live in [configs/](configs/), composed via `defaults:` lists.
- Structured config schemas live next to the modules they configure (e.g. `MLPConfig` in `models/mlp.py`).
- Use `hydra.utils.instantiate(cfg.x)` only at leaves. Composition between modules uses plain function calls.

## MLflow invariants

- Backend store: `sqlite:///mlflow.db`. Artifact root: `./mlartifacts`.
- Use **aliases**, not stages, for the model registry.
- `mlflow.pyfunc.log_model(name=...)` — never `artifact_path=` (deprecated).
- Datasets via `mlflow.data.from_numpy(...)` + `mlflow.log_input(dataset, context=...)`.

## Research discipline

- For any non-trivial library API touch, an entry in [research/framework-rebuild/manifest.md](research/framework-rebuild/manifest.md) must back it (per [RESEARCHER.md](RESEARCHER.md)).
- The manifest is on disk **before** the implementation lands.
