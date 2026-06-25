# Research manifest — Framework rebuild (`mpinv`)

Task slug: `framework-rebuild`. Date: 2026-05-07.

This manifest is the Phase 3 record (per [RESEARCHER.md](../../RESEARCHER.md)) of every external library API the new `mpinv` framework depends on. Every import, function-call signature, and configuration key in `src/mpinv/**` and `configs/**` must trace back to one of the entries here.

---

## Phase 1 — Comprehension

**Task understood as**: build a clean, single-package Python framework named `mpinv` that solves the inverse problem `P_UT → (a^E, a^M)` formalised in [presentation/ch1_full.md](../../presentation/ch1_full.md). Replaces the legacy [/Users/chersie/Desktop/diplom](/Users/chersie/Desktop/diplom) (cited in the plan as buggy and messy). Must support: synthetic + real-antenna data with augmentations and lazy/streaming loading, multiple feature pipelines (PCA, FFT-radial, HOG, SH-power), multiple model architectures (linear baselines + MLP variants), at least two loss types (coefficient MSE and physics power loss through a differentiable VSH decoder), Hydra-driven configs, MLflow tracking and model bundling, Optuna HPO, and a comprehensive analysis suite.

**Technology / topic inventory**:

| Topic | Status | Resolved by |
|---|---|---|
| `torch-harmonics` inverse vector real SHT API | `[NEEDS RESEARCH]` | R1 |
| Hydra structured configs and instantiation | `[NEEDS RESEARCH]` | R2 |
| Hydra-Optuna sweeper plugin status | `[NEEDS RESEARCH]` | R2b |
| Optuna 4.x sampler/pruner/study API + MLflow callback | `[NEEDS RESEARCH]` | R2c |
| MLflow 3.x tracking server, runs, params, metrics, figures | `[NEEDS RESEARCH]` | R3 |
| MLflow 3.x dataset entity (`mlflow.data.from_*`, `log_input`) | `[NEEDS RESEARCH]` | R3 |
| MLflow 3.x `pyfunc.log_model(name=...)` rename and registry aliases | `[NEEDS RESEARCH]` | R3 |
| PyTorch 2.6+ AMP `autocast` / `GradScaler` defaults | `[KNOWN]` | (PyTorch docs; standard pattern) |
| scikit-learn `IncrementalPCA` for streaming fit | `[KNOWN]` | (sklearn docs; standard pattern) |
| Hatchling build backend in `pyproject.toml` | `[KNOWN]` | (pyproject standard) |
| Project conventions (L=15, K=255, 360×179, packing) | `[KNOWN]` | inherited verbatim from [presentation/ch1_full.md](../../presentation/ch1_full.md) §1.1–1.2 |

---

## R1 — `torch-harmonics` inverse vector real SHT

- **Citations**:
  - NVIDIA, `torch-harmonics`, version `0.9.0`, released 2026-04-16 (Linux x86_64 wheels only).
  - NVIDIA, `torch-harmonics`, version `0.6.5`, last release with a pure-Python `py3-none-any.whl` wheel. **The framework pins `torch-harmonics==0.6.5`** because `0.7.x`+ ship only as C++/CUDA sdists requiring `nvcc`; this is impractical on macOS arm64. The `InverseRealVectorSHT` public API is identical in 0.6.5 and 0.9.0 (verified by reading `torch_harmonics/sht.py` on both tags).
- **Sources consulted**:
  - PyPI <https://pypi.org/project/torch-harmonics/>.
  - GitHub <https://github.com/NVIDIA/torch-harmonics>, branch `main` and tag `v0.6.5`.
  - Source file <https://raw.githubusercontent.com/NVIDIA/torch-harmonics/v0.6.5/torch_harmonics/sht.py>.
  - The example notebook `notebooks/partial_derivatives.ipynb` in that repo.
- **Verified facts**:
  - **Constructor** (current `main`): `InverseRealVectorSHT(nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True)`.
  - **Coefficient layout**: `(..., 2, lmax, mmax)` complex tensor. The leading `2` is the channel axis, with channels = (**spheroidal**, **toroidal**) in `torch-harmonics` terminology. For an electric-vs-magnetic VSH split, **spheroidal corresponds to the electric (E / TM) family** and **toroidal corresponds to the magnetic (M / TE) family** under standard sign conventions.
  - `l` is **zero-indexed** and `lmax` is **non-inclusive** (slice-style semantics). Therefore for `L = 15` (modes l = 1..15) we pass **`lmax = 16, mmax = 16`** to the constructor; the `l = 0` row of the coefficient tensor stays zero (the trivial monopole is unused).
  - Only `m ≥ 0` is stored explicitly because of conjugate symmetry. `m = 0` lives at index `0`; `m = l` lives at index `l`. The library handles the negative-`m` reconstruction internally via Hermitian symmetry; the user must **not** add the `m < 0` contributions manually.
  - **Output shape** of `forward(coeffs)` is `(..., 2, nlat, nlon)`. The two output channels are the (**θ-component**, **φ-component**) of the tangential vector field.
  - **Grids supported by VSHT**: `equiangular` (Clenshaw–Curtiss, **poles included** at `θ = 0` and `θ = π`); `legendre-gauss` (poles excluded, Gauss–Legendre nodes); `lobatto` (Gauss–Lobatto, poles included).
  - **Differentiable**: gradients flow through `InverseRealVectorSHT.forward` end-to-end; the only known historical gradient bug was an upstream PyTorch `irfft` CUDA issue fixed in `torch ≥ 2.1.1` (irrelevant since `torch-harmonics 0.9.0` requires `torch ≥ 2.6.0`).
- **Project decision (binding for R1)**:
  - The project's grid is 1° equiangular with **poles excluded** (179 polar samples θ = 1°..179°, 360 azimuthal samples φ = 0°..359°). This is **not** the same as `torch-harmonics`'s `equiangular` grid (which includes the poles).
  - Resolution: instantiate `InverseRealVectorSHT(nlat=181, nlon=360, lmax=16, mmax=16, grid="equiangular")`. The 181 polar samples land at θ = `i · π / 180` for `i = 0..180`, i.e. `0°, 1°, 2°, ..., 180°`. The 179-sample project grid is the inner slice `[1:180]`.
  - The boundary layer between project layout and torch-harmonics layout slices `field[:, :, 1:180, :]` on the way out and zero-pads on the way in.
- **Legacy bugs explained**:
  - The legacy `lmax = maxorder + 1` in [/Users/chersie/Desktop/diplom/src/models/physics_layers.py](/Users/chersie/Desktop/diplom/src/models/physics_layers.py) was **numerically correct** (because `lmax` is non-inclusive) but the legacy comment ("inactive band 0") explained it for the wrong reason.
  - The legacy `coefficients_to_sht_format` accumulated `+m` and `−m` contributions into the same slot with `+=`, which doubles the `|m| > 0` modes for a real SHT (the library already accounts for the negative-m mirror). The new code stores **only the `m ≥ 0` complex amplitude** at `[..., l, m]`.
- **Use in framework**:
  - [src/mpinv/losses/differentiable_field.py](../../src/mpinv/losses/differentiable_field.py) instantiates `InverseRealVectorSHT(nlat=181, nlon=360, lmax=16, mmax=16, grid="equiangular")`.
  - The packed-coefficient `(B, 4K)` real vector is mapped to `(B, 2, 16, 16)` complex via [src/mpinv/core/packing.py](../../src/mpinv/core/packing.py) → `pack_to_sht_grid`.
  - The `(B, 2, 181, 360)` raw output is sliced to `(B, 2, 179, 360)` and combined as `P = E_θ.abs()**2 + E_φ.abs()**2`.

---

## R2 — Hydra structured configs

- **Citation**: `hydra-core` version `1.3.2`, released 2023-02-23.
- **Sources consulted**:
  - PyPI <https://pypi.org/project/hydra-core/>.
  - Hydra docs <https://hydra.cc/docs/intro/>, <https://hydra.cc/docs/advanced/instantiate_objects/overview/>, <https://hydra.cc/docs/tutorials/structured_config/intro/>.
  - GitHub <https://github.com/facebookresearch/hydra>.
- **Verified facts**:
  - Structured configs use plain Python `dataclasses` (Pydantic is not required and not officially supported as a config primitive).
  - Registration: `from hydra.core.config_store import ConfigStore; cs = ConfigStore.instance(); cs.store(group="model", name="mlp_small", node=MLPConfig)`.
  - Top-level YAML composition uses a `defaults:` list: `defaults: [{model: mlp_small}, {data: synthetic_l15}, _self_]`.
  - Entry-point decorator: `@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train")`. The `version_base="1.3"` argument is the recommended modern setting (silences the no-version warning; pins compose semantics).
  - `hydra.utils.instantiate(cfg.x, **kwargs)` constructs an object from `cfg.x._target_` plus the remaining keys; `_recursive_=True` is the default and recursively instantiates nested configs.
  - `OmegaConf` 2.3 ships bundled. Useful resolvers: `${oc.env:VAR,default}`, `${now:%Y-%m-%d_%H-%M-%S}`, `${hydra:runtime.output_dir}`, `${hydra:job.name}`.
- **Use in framework**:
  - [src/mpinv/cli/_configstore.py](../../src/mpinv/cli/_configstore.py) registers every dataclass schema with `ConfigStore`.
  - All CLIs use `@hydra.main(version_base="1.3", config_path="../../../configs", config_name=...)`.
  - Object construction via `hydra.utils.instantiate` only at leaves (model, optimiser, scheduler, individual feature extractors). Composition between leaves is plain Python.

---

## R2b — `hydra-optuna-sweeper`: not used

- **Citation**: `hydra-optuna-sweeper` version `1.2.0`, released 2022-05-18, last update 2022.
- **Sources consulted**:
  - PyPI <https://pypi.org/project/hydra-optuna-sweeper/>.
  - GitHub <https://github.com/facebookresearch/hydra/tree/main/plugins/hydra_optuna_sweeper>.
- **Verified facts**:
  - The plugin pins `optuna < 3.0.0, >= 2.10.0`. It is incompatible with Optuna 4.x and is effectively unmaintained (no release in ≈4 years).
- **Project decision (binding)**:
  - The framework does **not** depend on `hydra-optuna-sweeper`.
  - HPO is implemented as a custom `mpinv-sweep` CLI ([src/mpinv/cli/sweep.py](../../src/mpinv/cli/sweep.py)) that calls `optuna.create_study(...).optimize(...)` directly, parameterised by Hydra config (`configs/sweep/optuna_mlp.yaml`).
  - Each Optuna trial is one MLflow nested run via `optuna_integration.MLflowCallback`.

---

## R2c — Optuna 4.x

- **Citation**: `optuna` version `4.8.0`, released 2026-03-16.
- **Sources consulted**:
  - PyPI <https://pypi.org/project/optuna/>.
  - Optuna docs <https://optuna.readthedocs.io/en/stable/>.
  - `optuna-integration` package <https://pypi.org/project/optuna-integration/>.
- **Verified facts**:
  - `study = optuna.create_study(study_name=..., direction="minimize", sampler=optuna.samplers.TPESampler(seed=...), pruner=optuna.pruners.MedianPruner(), storage="sqlite:///optuna.db", load_if_exists=True)`.
  - `study.optimize(objective, n_trials=N, n_jobs=...)`.
  - `trial.suggest_int(name, low, high, step=1, log=False)`, `trial.suggest_float(name, low, high, step=None, log=False)`, `trial.suggest_categorical(name, choices)`.
  - **MLflow integration moved**: `optuna.integration.MLflowCallback` is now `optuna_integration.MLflowCallback` (separate package `optuna-integration[mlflow]`).
  - Construction: `MLflowCallback(tracking_uri=..., metric_name="val/loss", create_experiment=False, mlflow_kwargs={"nested": True})`.
- **Use in framework**:
  - [src/mpinv/cli/sweep.py](../../src/mpinv/cli/sweep.py) reads `cfg.sweep`, constructs the sampler/pruner/storage, defines an objective that builds a child Hydra config per trial, runs training, and returns `val/loss`.

---

## R3 — MLflow 3.x

- **Citation**: `mlflow` version `3.12.0`, current stable as of 2026-05-07. Requires Python `>= 3.10`.
- **Sources consulted**:
  - PyPI <https://pypi.org/project/mlflow/>.
  - GitHub `v3.12.0` source <https://github.com/mlflow/mlflow/tree/v3.12.0>.
  - MLflow docs <https://mlflow.org/docs/latest/index.html>.
- **Verified facts**:
  - `mlflow.set_tracking_uri("sqlite:///mlflow.db")` or `"http://127.0.0.1:5000"`.
  - `mlflow.set_experiment(name)` / `mlflow.create_experiment(name, artifact_location=...)`.
  - `mlflow.start_run(run_id=None, experiment_id=None, run_name=None, nested=False, parent_run_id=None, tags=None, description=None, log_system_metrics=None)`. **`nested=True` creates a child run inside the active parent.**
  - `mlflow.log_param(key, value)`, `mlflow.log_params(dict)`, `mlflow.log_metric(key, value, step=None)`, `mlflow.log_metrics(dict, step=None)`, `mlflow.set_tag(key, value)`, `mlflow.set_tags(dict)`.
  - `mlflow.log_figure(figure, artifact_file, *, save_kwargs=None)`. Accepts matplotlib `Figure` and plotly `Figure`. Format inferred from `artifact_file` extension (`.png`, `.pdf`, `.svg`, `.html`).
  - `mlflow.log_text(text, artifact_file)`, `mlflow.log_dict(d, artifact_file)`, `mlflow.log_artifact(local_path, artifact_path=None)`.
  - **Datasets**: `mlflow.data.from_numpy(features, targets=None, source=None, name=None, digest=None)` and `mlflow.data.from_pandas(df, source=None, name=None, targets=None, digest=None)`. Log via `mlflow.log_input(dataset, context=..., tags=None, model=None)` where `context` is a free-form string (we use `"training" | "validation" | "holdout" | "dummy"`). **Persists metadata only (name, digest, schema, profile, source) — not the bytes.**
  - **Model logging**: `mlflow.pyfunc.log_model(name=..., python_model=..., artifacts=None, input_example=None, signature=None, pip_requirements=None, registered_model_name=None)`. `artifact_path=` is **deprecated** in favour of `name=`.
  - `class MyPyfunc(mlflow.pyfunc.PythonModel): def predict(self, context, model_input, params=None): ...`.
  - **Model registry**: stages (`Staging`, `Production`, `Archived`) deprecated since 2.9. Use **aliases** via `from mlflow import MlflowClient; client = MlflowClient(); client.set_registered_model_alias(name, alias, version)`. Reference: `models:/{name}@{alias}` in `mlflow.pyfunc.load_model(...)`.
  - **Autolog**: `mlflow.pytorch.autolog()` is essentially a no-op for vanilla `nn.Module` (it only intercepts TensorBoard `add_scalar`/`add_hparams` from PyTorch Lightning). The framework does explicit logging only.
  - **Local server defaults to `uvicorn`** with `--workers 4`. Recommended invocation: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000`.
  - Cleanup: `MlflowClient.delete_run(run_id)` is a soft delete (lifecycle stage → `deleted`); `mlflow gc --older-than 30d` permanently frees disk.
- **Use in framework**:
  - [src/mpinv/tracking/mlflow_sink.py](../../src/mpinv/tracking/mlflow_sink.py) implements the `Sink` protocol against the verified API surface above.
  - [src/mpinv/tracking/dataset_logger.py](../../src/mpinv/tracking/dataset_logger.py) wraps `mlflow.data.from_numpy` / `mlflow.data.from_pandas` and `mlflow.log_input`.
  - Model bundles (PCA + normaliser + `nn.Module`) are persisted as `mlflow.pyfunc.log_model(name="model", python_model=...)`. Registry promotion uses `set_registered_model_alias`.
  - The local launch script is [scripts/start_mlflow_server.sh](../../scripts/start_mlflow_server.sh).

---

## R4 — Multi-head per-`l`-band MLP and stage-wise training

Added 2026-05-27 in support of the [.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md](../../.cursor/plans/multi-head_per-l_mode_training_57916da2.plan.md) plan. The architectural choices below are backed by the cited sources, not invented.

- **Sources consulted**:
  - PyTorch source `torch/nn/modules/linear.py` (`torch==2.4.x`, the version pinned in [pyproject.toml](../../pyproject.toml) — `torch>=2.4,<2.6`). Read locally at `.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:114-122`.
  - PyTorch docs <https://pytorch.org/docs/2.4/generated/torch.nn.Linear.html> and <https://pytorch.org/docs/2.4/notes/autograd.html#locally-disable-gradient-computation> for the `requires_grad` semantics on `nn.Parameter`.
  - PyTorch docs <https://pytorch.org/docs/2.4/optim.html#per-parameter-options> for per-param-group LR (used by the `lower_lr_after_stage1` backbone policy).
  - The packing layout, fixed once in [src/mpinv/core/packing.py](../../src/mpinv/core/packing.py): `[Re a^E | Im a^E | Re a^M | Im a^M]` with inner ordering `l = 1..L`, `m = -l..+l`. Per-`l` block size per quarter is `2 l + 1`, total per-`l` width across all four quarters is `4 (2 l + 1)`. With `L = 5` the per-`l` widths are 12, 20, 28, 36, 44 and `4K = 140`.
- **Verified facts**:
  - `nn.Linear.reset_parameters()` calls `kaiming_uniform_(self.weight, a=math.sqrt(5))` and `uniform_(self.bias, -1/sqrt(fan_in), +1/sqrt(fan_in))`. This is the exact init `nn.Linear.__init__` runs, so calling `linear.reset_parameters()` after a `zero_` is the canonical "re-initialise to default-distributed weights" path. **No alternative private init function is needed.**
  - Setting `param.requires_grad = False` on a parameter:
    - excludes it from `model.parameters()` filtered by `requires_grad`, so an optimiser built from `(p for p in model.parameters() if p.requires_grad)` will not see it (the framework's `_filter_params` in [src/mpinv/training/optim.py](../../src/mpinv/training/optim.py) already does this);
    - does **not** prevent the value from contributing to forward passes; gradient just doesn't flow back through the parameter (it can still flow through inputs).
  - `optim.param_groups` accepts a list of dicts each carrying their own `lr` / `weight_decay`; this is the supported mechanism for the `lower_lr_after_stage1` backbone policy.
  - The packing's 4-way quarter split (`[Re aE | Im aE | Re aM | Im aM]`) means each per-`l` head writes `2 l + 1` real entries into **four** disjoint stride-K slots of the output vector, one per quarter. The scatter is therefore a single `index_copy_(1, idx, ...)` per head with `idx` of length `4 (2 l + 1)`.
- **Decisions for this plan**:
  - Multi-head MLP groups are a **partition** of `{1, 2, ..., L}` — overlap or gaps raise `ValueError`. This is what makes the concatenation cover the full packed vector exactly once.
  - Re-init at stage boundary: `head.reset_parameters()` (the canonical PyTorch path verified above). The user's "zero modes 2-5 → re-initialise head 2 with non-zero weights when activating it" reduces to `head.weight.zero_(); head.bias.zero_()` for inactive heads and `head.reset_parameters()` at activation.
  - Backbone policies: `freeze_after_stage1` (default), `trainable_always`, `lower_lr_after_stage1` with a configurable factor (default 0.1).
- **Use in framework**:
  - [src/mpinv/models/multi_head_mlp.py](../../src/mpinv/models/multi_head_mlp.py) — `MultiHeadMLP`, `MultiHeadMLPConfig`, scatter `index_map`, `transplant_heads`.
  - [src/mpinv/training/staged.py](../../src/mpinv/training/staged.py) — `StagedTrainer`, `StagedTrainerConfig`.
  - [src/mpinv/models/mlp.py](../../src/mpinv/models/mlp.py) — refactored to expose a `make_backbone(cfg)` helper that returns the body up to (but not including) the final `Linear`. The original `MLP.forward` keeps its previous semantics.

---

## Hard-rule check (RESEARCHER.md "Hard rule on empty research")

Phase 3 produced new information for R1, R2, R2b, R2c, R3, R4. All `[NEEDS RESEARCH]` items from Phase 1 are resolved before Phase 4 implementation begins.

## Sources I consulted but did not use

- `pyshtools` — alternative spherical-harmonics library, ruled out because it does not have a vector SHT and would require a separate scalar SHT pipeline; we keep `torch-harmonics` for differentiability.
- `lightning` — practice.pdf argues against; our Trainer is custom.
- `wandb` — not in scope; MLflow is the explicit user requirement.
- `pyrootutils` / `dotenv` — config helpers we may add later if needed.
