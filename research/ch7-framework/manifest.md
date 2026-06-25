# Research Manifest — Chapter 7 (Framework)

**Scope**: factual claims about the `mpinv` package made in `paper/08_chapter7_framework.md`.

## R1 — Package layout

**Source**: directory listing of `src/mpinv/` (Bash `ls`).
**Verified subpackages**:
- `core/` — `area_weights.py`, `grid.py`, `packing.py`, `seeds.py`, `shapes.py`, `types.py`
- `data/` — `synthetic_generator.py`, `augment.py`, `real_antenna_loader.py`, `real_augmented_pipeline.py`, `basis_decomposer.py`, `splits.py`, `dummy_probe.py`, `memmap_dataset.py`, `_basis_cache.py`
- `features/` — `raw_flat.py`, `subsample.py`, `pca.py`, `fft_radial.py`, `sh_power.py`, `composite.py`, `normalisers.py`, `power_pipeline.py`, `modes.py`, `hog.py`, `registry.py`
- `models/` — `base.py`, `mlp.py`, `multi_head_mlp.py`, `linear_baselines.py`, `registry.py`
- `losses/` — `coef_mse.py`, `physics_power.py`, `rank_bin.py`, `differentiable_field.py`, `registry.py`
- `training/` — `trainer.py`, `staged.py`, `optim.py`, `amp.py`, `sanity.py`
- `callbacks/` — `base.py`, `checkpoint_cb.py`, `early_stopping_cb.py`, `grad_clip_cb.py`, `logging_cb.py`, `memory_watchdog_cb.py`, `timing_cb.py`, `validation_cb.py`
- `analysis/` — `metrics/` (`coefficient_metrics.py`, `field_metrics.py`, `mode_metrics.py`), `plots/` (eight plotters), `reports/run_report.py`
- `tracking/` — `dataset_logger.py`, `mlflow_sink.py`, `params.py`
- `cli/` — `train.py`, `evaluate.py`, `sweep.py`, `generate_data.py`, `validate_physics.py`, `report.py`, `data_stats.py`, `_builders.py`, `_configstore.py`

## R2 — Public package surface

**Source**: `src/mpinv/__init__.py`.
**Verified re-exports**: `GRID_DEFAULT`, `K_MODES`, `L_MAX`, `PACKED_DIM`, `GridSpec`, `pack_coefficients`, `unpack_coefficients`, `__version__ = "0.1.0"`.

## R3 — Grid invariant

**Source**: `src/mpinv/core/grid.py` lines 23–101.
**Verified**: `GridSpec(n_phi=360, n_theta=179, theta_start_deg=1.0, theta_end_deg=179.0)` is the single source of truth. Docstring: «every other module imports `GRID_DEFAULT` rather than re-declaring shapes inline.»

## R4 — Packing invariant

**Source**: `src/mpinv/core/packing.py` lines 28–35.
**Verified**: `L_MAX = 15`, `K_MODES = L_MAX * (L_MAX + 2) = 255`, `PACKED_DIM = 4 * K_MODES = 1020`. Packed layout: `[Re a^E, Im a^E, Re a^M, Im a^M]`. (Note: thesis text uses `L = 5, K = 35, 4K = 140` per Chapter 1 §1.1 — this is the thesis truncation convention; the framework supports up to `L = 15` to allow growth via Матрёшка-trick per Chapter 3 §3.2.)

## R5 — Registries

**Source**: `src/mpinv/models/registry.py`, `src/mpinv/losses/registry.py`, `src/mpinv/features/registry.py`.
**Verified**: Three mirrored registries `MODELS`, `LOSSES`, `FEATURE_EXTRACTORS`, populated through `@register_model`, `@register_loss`, `@register_feature` decorators. Each rejects re-registration to a different class.

## R6 — CLI entry points

**Source**: `pyproject.toml` lines 56–63.
**Verified**: Seven scripts — `mpinv-train`, `mpinv-evaluate`, `mpinv-sweep`, `mpinv-generate-data`, `mpinv-validate-physics`, `mpinv-report`, `mpinv-data-stats`.

## R7 — Hydra and MLflow

**Source**: `pyproject.toml` lines 30, 32, `configs/train.yaml`, `src/mpinv/cli/train.py` lines 28–48.
**Verified**: `hydra-core==1.3.2`, `mlflow>=3.10,<4`. Top-level config composes nine groups: `data`, `features`, `model`, `loss`, `optimiser`, `scheduler`, `trainer`, `callbacks`, `tracking`. MLflow integration via `mpinv.tracking.mlflow_sink.MLflowSink`.

## R8 — Callbacks

**Source**: `src/mpinv/callbacks/__init__.py`.
**Verified**: Seven concrete callbacks plus `Callback` base — `CheckpointCallback`, `EarlyStoppingCallback`, `GradClipCallback`, `LoggingCallback`, `MemoryWatchdogCallback`, `TimingCallback`, `ValidationCallback`.

## R9 — Training surface

**Source**: `src/mpinv/training/` listing.
**Verified**: `Trainer` in `trainer.py`, `StagedTrainer` in `staged.py`, optimiser/scheduler builders in `optim.py`, mixed precision in `amp.py`, smoke checks in `sanity.py`.

## R10 — torch-harmonics dependency

**Source**: `pyproject.toml` lines 27–29 and inline comment.
**Verified**: `torch-harmonics==0.6.5` is pinned because newer releases ship CUDA sdists; the framework uses `InverseRealVectorSHT` API which is stable across 0.6.5 → 0.9.0.

## What is NOT in the manifest

Out of scope for Chapter 7 text: precise function signatures (these belong in code reading, not thesis prose); experimental configuration values (these belong in Chapter 6); MLflow run IDs (these belong in artefacts on disk).
