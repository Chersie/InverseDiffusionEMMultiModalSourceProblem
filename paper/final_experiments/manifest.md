# Research manifest — Final-experiments scaffolding (`paper/final_experiments/`)

Task slug: `final-experiments`. Date: 2026-06-09.

This manifest is the Phase 3 record (per [RESEARCHER.md](../../RESEARCHER.md)) of every API touch made to support the proposal-driven Step 0 → Step 3 experiment plan in [paper/proposal.md](../proposal.md). The plan-of-record is [.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md](../../.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md). Every new function, configuration key, and CLI flag in this scope must trace back to one of the entries below.

---

## Phase 1 — Comprehension

**Task understood as**: extend the existing `mpinv-train` Hydra CLI so the four-step proposal experiment plan (base → regularization → feature → scheduling) can be driven entirely from yaml configs in [paper/final_experiments/](.); add the two missing proposal features that the current code does not implement (truncated-target P; "all_trainable_active_boost" backbone policy); avoid duplicating the real-augmented data pipeline now scattered across two scripts.

**Technology / topic inventory**:

| Topic | Status | Resolved by |
|---|---|---|
| `MultiHeadMLP` Hydra integration | `[NEEDS RESEARCH]` | R1 |
| `StagedTrainer` Hydra integration | `[NEEDS RESEARCH]` | R2 |
| Real-augmented pipeline → library | `[NEEDS RESEARCH]` | R3 |
| Proposal axis A: backbone-and-head training policies | `[NEEDS RESEARCH]` | R4 |
| Proposal axis B: target-P truncation | `[NEEDS RESEARCH]` | R5 |
| Composite metric for cell ranking | `[KNOWN]` | proposal §"Метрики" + chat clarifications |
| Hydra `--multirun` BasicSweeper grid syntax | `[KNOWN]` | (Hydra 1.3 docs) |

---

## R1 — `MultiHeadMLP` Hydra integration

- **Sources consulted**:
  - [src/mpinv/models/multi_head_mlp.py](../../src/mpinv/models/multi_head_mlp.py) (current implementation, registered via `@register_model("multi_head_mlp")`).
  - [configs/model/multi_head_mlp_5x200.yaml](../../configs/model/multi_head_mlp_5x200.yaml) (existing YAML, fixed `output_dim: 140`, `groups: [[1], …, [5]]`).
  - [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) lines 180–189 of the pre-edit file (model dispatch — only handled `MLP` and `LinearBaseline`).
- **Verified facts**:
  - `MultiHeadMLPConfig.__post_init__` validates `output_dim == 4 · l_max · (l_max + 2)` and that `groups` partitions `{1, …, l_max}` exactly. Hydra delivers `groups` as a `ListConfig` of `ListConfig`s, which is not a plain `list[list[int]]` and triggers `validate_groups`'s `int(l)` coercion path successfully — but to be safe and to keep `_post_init_` invariants explicit, we coerce to `list[list[int]]` in the dispatcher before constructing the config.
  - The CLI already injects `input_dim` from the feature pipeline and `output_dim = 4 · K` from `data.l_max`. The existing `multi_head_mlp_5x200.yaml` declares `output_dim: 140` (= 4 · 5 · 7) — consistent with `data.l_max = 5`. We additionally cross-check `cfg.l_max == data.l_max` and raise `ValueError` on mismatch.
- **Use in framework**:
  - [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) — new `elif model_target == "mpinv.models.multi_head_mlp.MultiHeadMLP":` branch in the model dispatch.
  - **Production backbone**: per a follow-up chat directive, the experiment yamls in this directory were switched from `mlp_5x200` / `multi_head_mlp_5x200` (the schema reference cited above) to the smaller [configs/model/mlp_4x60.yaml](../../configs/model/mlp_4x60.yaml) and matching [configs/model/multi_head_mlp_4x60.yaml](../../configs/model/multi_head_mlp_4x60.yaml). Both follow the schema verified above; only `hidden_size` (60) and `n_hidden_layers` (4) differ from the cited reference.

---

## R2 — `StagedTrainer` Hydra integration

- **Sources consulted**:
  - [src/mpinv/training/staged.py](../../src/mpinv/training/staged.py): `StagedTrainerConfig`, `StagedTrainer.fit(...)`, `StageReport`. Public surface verified by reading the module (last touched on 2026-04-30 per the on-disk timestamp).
  - [configs/training/staged.yaml](../../configs/training/staged.yaml) (existing).
- **Verified facts**:
  - `StagedTrainer.fit(...)` requires (a) `optim_cfg: OptimiserConfig` (NOT a built optimiser — built per stage), (b) `sched_cfg: SchedulerConfig | None`, (c) per-stage callback and sink *factories* (because `EarlyStoppingCallback` and `CheckpointCallback` are stateful and must be fresh per stage — verified by reading the module docstring at [src/mpinv/training/staged.py L246–249](../../src/mpinv/training/staged.py#L246-L249)).
  - The inner `Trainer` is wrapped: `StagedTrainer(staged_cfg, trainer=Trainer(tr_cfg))`. The constructor explicitly overrides `self.trainer.cfg.max_epochs = staged_cfg.stage_max_epochs`.
  - `StageReport` is a `@dataclass(slots=True)` and supports `dataclasses.asdict()` for JSON serialisation.
  - Setting `cfg.training._target_ = "mpinv.training.staged.StagedTrainerConfig"` is the only safe way to dispatch from Hydra without breaking the existing single-stage path; we strip `_target_` before passing the dict to the dataclass constructor.
- **Use in framework**:
  - [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) — new branch when `cfg.training._target_ == "mpinv.training.staged.StagedTrainerConfig"`. The branch:
    1. validates the model is `MultiHeadMLP`,
    2. builds `StagedTrainerConfig` from the resolved cfg dict,
    3. wraps a fresh `Trainer(tr_cfg)`,
    4. defines `_callbacks_factory(stage_idx)` that delegates to the existing `_make_callbacks(...)` helper with a per-stage subdir,
    5. calls `staged_trainer.fit(...)` with `optim_cfg`, `sched_cfg`, `callbacks_factory`, `sinks_factory`, and `steps_per_epoch=len(train_loader)`,
    6. dumps `[asdict(r) for r in reports]` to `output_dir/stage_reports.json`.

---

## R3 — Real-augmented pipeline library

- **Sources consulted**:
  - [scripts/run_real_augmented.py](../../scripts/run_real_augmented.py) — pre-existing inline implementation of `_load_real`, `_load_smoke`, `_truncate_and_resynthesise`, `_build_augmented`, `_peek_split_ids` (functions with `argparse.Namespace`-based signatures).
  - [scripts/run_staged_real_augmented.py](../../scripts/run_staged_real_augmented.py) lines 70–87 — `from run_real_augmented import _aug_cache_key, _build_augmented, …`. **Backwards-compatibility constraint**: these symbols must remain importable from the script for `run_staged_real_augmented.py` to keep working.
  - [src/mpinv/data/real_antenna_loader.py](../../src/mpinv/data/real_antenna_loader.py): `RealAntennaLoaderConfig`, `list_real_antenna_samples`, `load_real_antenna`.
  - [src/mpinv/data/augment.py](../../src/mpinv/data/augment.py): `apply_augmentation`, `FieldPhiRollConfig`, `CoefModeDropoutConfig`, `FieldAdditiveNoiseConfig`.
  - [src/mpinv/data/synthetic_generator.py](../../src/mpinv/data/synthetic_generator.py): `SyntheticGenerator`, `SyntheticGeneratorConfig` (used by `load_smoke` and the optional synthetic colored-α=2 test split).
- **Verified facts**:
  - The legacy chunked dropout pattern (`P_chunks: list[np.ndarray]`, default `chunk_size=500`) is intentional: at the canonical `360 × 179` grid, full-batch coefficient-space re-synthesis of 10 000 samples peaks at ~20 GB; chunking caps that at ~500 MB per family.
  - The cache layer (`_aug_cache_*` functions and `_AUG_CACHE_VERSION`) is **not** moved into the library — it is script-level orchestration.
  - The output dict produced by the library is API-compatible with `mpinv.cli._builders.build_data_pipeline`'s return shape (`grid`, `l_max`, `n_train`, `n_val`, `P_train`, `packed_train`, `P_val`, `packed_val`, `batch_size`, `num_workers`, `basis`); we add the optional keys `n_test`, `P_test`, `packed_test`, `n_holdout`, `P_holdout`, `packed_holdout` so the cli/train.py report block can pick the holdout pair from the data dict instead of from `cfg.holdout`.
- **Use in framework**:
  - **NEW** [src/mpinv/data/real_augmented_pipeline.py](../../src/mpinv/data/real_augmented_pipeline.py) — exposes `truncate_and_resynthesise`, `load_real`, `load_smoke`, `build_augmented`, `peek_split_ids`, `build_real_augmented_pipeline`.
  - **NEW** [configs/data/real_augmented_l5.yaml](../../configs/data/real_augmented_l5.yaml) — Hydra wrapper around `build_real_augmented_pipeline`.
  - [scripts/run_real_augmented.py](../../scripts/run_real_augmented.py) — the five inline helpers become thin `from mpinv.data.real_augmented_pipeline import …` wrappers; `_aug_cache_*`, `_eval_split_chunked`, `_predict_chunked`, `_figures_for`, `_maybe_synthetic_test` stay in the script.
  - [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) — the holdout report block now prefers `data["P_holdout"]` / `data["packed_holdout"]` when present (skipping the file-load path).

---

## R4 — Proposal axis A: backbone-and-head training policies

- **Citation**: [paper/proposal.md](../proposal.md) §"Матрёшка-trick для предсказаний / Головы для коэффициентов разных порядков" — three named recipes:
  1. *«способ с заморозкой backbone-модели»*
  2. *«способ с заморозкой только предыдущих голов»*
  3. *«способ с увеличением lr на текущую голову без заморозки предыдущих и backbone»* (**not implemented in current code**)
- **Sources consulted**:
  - [src/mpinv/training/staged.py](../../src/mpinv/training/staged.py) `BackbonePolicy = Literal["trainable_always", "freeze_after_stage1", "lower_lr_after_stage1"]` (pre-edit).
  - `apply_stage_policy(...)` — current behaviour: previous heads are *always* frozen (`set_head_trainable(j, False)` for `j < active`), regardless of policy.
  - `build_stage_optimiser(...)` — current behaviour: only `lower_lr_after_stage1` builds split parameter groups; no policy keeps previous heads in the optimiser.
- **Verified facts**:
  - Mapping proposal recipes to existing literals:
    - "freeze backbone" → `freeze_after_stage1`. ✓
    - "freeze only previous heads, backbone trains" → `trainable_always`. ✓
    - "active head higher LR, NO freezing of previous heads or backbone" → **no existing match**.
  - To match recipe 3, the new policy must (a) keep previous heads `requires_grad=True`, (b) build an optimiser with two parameter groups: `(backbone + previous_heads) @ lr * backbone_lr_factor` and `active_head @ lr`.
- **Project decision (binding for R4)**:
  - Add `BackbonePolicy = Literal["…", "all_trainable_active_boost"]`.
  - In `apply_stage_policy`, set `keep_previous_trainable = (backbone_policy == "all_trainable_active_boost")` and use it when toggling `set_head_trainable(j, …)` for `j < active`.
  - In `build_stage_optimiser`, slice heads into `active_head_params` (head index = `stage_idx - 1`) and `other_head_params` (heads with `j != active`); for the new policy with `stage_idx > 1`, build groups `[backbone + other_head_params @ lr * factor, active_head @ lr]`. At `stage_idx == 1` the policy degenerates (no previous heads exist) and a single full-LR group is built.
  - Tests in [tests/unit/test_staged.py](../../tests/unit/test_staged.py): `test_apply_stage_policy_all_trainable_active_boost_keeps_previous_heads`, `test_build_stage_optimiser_active_boost_uses_two_groups_with_low_and_full_lr`, `test_build_stage_optimiser_active_boost_stage1_is_single_group`.

---

## R5 — Proposal axis B: target-P truncation

- **Citation**: [paper/proposal.md](../proposal.md) §"Матрёшка-trick для предсказаний / loss на голове k можем считать по":
  1. *«всему P»* (current behaviour)
  2. *«P, полученному по коэффициентам до головы k включительно»* (**not implemented in current code**)
- **Sources consulted**:
  - [src/mpinv/losses/physics_power.py](../../src/mpinv/losses/physics_power.py) — `PhysicsPowerLoss.forward(pred_packed, target, target_packed=None)`. Currently always computes the primary term against `target` (full ground-truth P).
  - [src/mpinv/losses/differentiable_field.py](../../src/mpinv/losses/differentiable_field.py) — `DifferentiableMultipoleField` is the canonical decoder; the loss already owns one (or accepts an externally supplied one via `decoder=` kwarg).
  - [src/mpinv/core/packing.py](../../src/mpinv/core/packing.py) — packing layout `[Re a^E, Im a^E, Re a^M, Im a^M]`, length `4 K = 4 · l_max · (l_max + 2)`. The flat per-quarter index for `(l, m)` is `(l - 1)(l + 1) + (m + l)`; the count of modes with `l ∈ {1, …, k}` is `k · (k + 2)` (exact closed form).
  - [src/mpinv/training/staged.py](../../src/mpinv/training/staged.py) `StagedTrainerConfig` (pre-edit) had no truncation knob; the per-stage `fit_one_stage(...)` is the right hook to mutate the loss config.
- **Verified facts**:
  - The packed layout has 4 quarters of width `K`; in each quarter, indices `[k(k+2), K)` correspond to modes `l > k`. Zeroing those indices in every quarter yields a packed vector with the bands `l ∈ {1, …, k}` preserved.
  - `forward` operates on torch tensors. The numpy/torch-agnostic helper `zero_above_band(packed, k, l_max)` uses duck-typed `.clone()` (torch) / `.copy()` (numpy) plus slice assignment, which is supported identically by both.
  - The `coef_aux` term in `PhysicsPowerLoss.forward` is left unchanged: it is taken against the **full** `target_packed`, never against the truncated copy. Documented in the field-level docstring.
  - `target_packed=None` (the default) is incompatible with `truncate_target_to_band != None`: we cannot recompute a truncated target P without the ground-truth coefficients. The forward raises `ValueError` in that branch.
- **Project decision (binding for R5)**:
  - Add `zero_above_band(packed, k, l_max)` to [src/mpinv/core/packing.py](../../src/mpinv/core/packing.py).
  - Add `truncate_target_to_band: int | None = None` to `PhysicsPowerLossConfig`. When set and `< l_max`, recompute the target inside `forward` from `zero_above_band(target_packed, k, l_max)` through the same `self.decoder`. The recomputation lives inside `torch.no_grad()` because the target is a fixed quantity for the loss.
  - Add `truncate_target_to_active_band: bool = False` to `StagedTrainerConfig`. In `fit_one_stage`, after `apply_stage_policy`, if the loss exposes a `cfg.truncate_target_to_band` attribute, set it to `max(group)` for the active head's l-band group. Loss for which the attribute is missing emits a one-line warning and the knob is silently skipped.
  - Tests in [tests/unit/test_packing.py](../../tests/unit/test_packing.py): `test_zero_above_band_keeps_low_l_zeroes_high_l`, `test_zero_above_band_k_equals_lmax_is_noop`, `test_zero_above_band_validates_inputs`, `test_zero_above_band_works_for_torch_tensors`.
  - Tests in [tests/unit/test_losses.py](../../tests/unit/test_losses.py): `test_physics_power_truncate_target_to_band_zero_when_truncation_matches_pred`, `test_physics_power_truncate_target_to_band_requires_target_packed`, `test_physics_power_truncate_to_lmax_is_noop`.
  - Test in [tests/unit/test_staged.py](../../tests/unit/test_staged.py): `test_staged_truncate_target_to_active_band_updates_loss_per_stage`.

---

## R6 — Composite metric for cell ranking

- **Citation**: chat clarification on top of [paper/proposal.md](../proposal.md) §"Метрики":
  - *Regression-inspired*: `R² MSE` on `(P_hat, P_true)`, `MSE` on `(coeff_hat, coeff_true)`.
  - *Ranking-inspired*: `Spearman correlation` on `P`, `bin accuracy` on `P`.
  - The proposal text: *«Метрики, вдохновлённые ранжированием, дают полезный сигнал на первых стадиях обучения, когда мы ещё не можем точно предсказать поле, но уже можем предсказать его форму.»*
- **Sources consulted**:
  - [src/mpinv/analysis/metrics/field_metrics.py](../../src/mpinv/analysis/metrics/field_metrics.py): `weighted_mse_P`, `weighted_nrmse_P`, `weighted_r2_P`, `spearman_rho_P`, `bin_accuracy_P` (all already implemented; verified by reading the module).
- **Verified facts**:
  - `weighted_nrmse_P` is non-negative and approaches 0 as `P_pred → P_true` (lower-is-better).
  - `spearman_rho_P` returns a value in `[-1, +1]`; +1 = perfect rank match, −1 = anti-rank, 0 = uncorrelated. Higher-is-better.
  - The two metrics live on different scales; the chosen weight `0.5` brings ρ from `[-1, +1]` to `[-0.5, +0.5]` so it doesn't dwarf NRMSE values typical for the project (~0.3 on val, ~3.0 on holdout per S5).
- **Project decision (binding for R6)**:
  - Composite: `composite = report/<split>/field_nrmse_w − 0.5 · report/<split>/spearman_rho_P`. Lower is better.
  - Default split prefix in [scripts/select_best_step.py](../../scripts/select_best_step.py) is `val` (matches `cli/train.py`'s val tag); the script accepts `--split-prefix val_real` for runs that emit a different prefix (e.g. legacy S5 JSON).
  - `cli/train.py` writes a `metrics.json` summary at the end of the report block so the selection helper can run without an MLflow round-trip; metric keys use the `report/<split>/<name>` convention so val, test, and holdout splits are uniformly accessible.
  - To make the composite computable on val, [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) `_eval_split` is extended to log `spearman_rho_P` and `bin_accuracy_P` (was previously only logged for test/holdout via the local helper); val now also gets the rich metric set via `val_tagged.update(_eval_split(…, tag="val"))`.

---

## R7 — Per-split plot suite + dummy split

- **Driving plan**: [.cursor/plans/per-split-plot-suite_e0f631cc.plan.md](../../.cursor/plans/per-split-plot-suite_e0f631cc.plan.md). Mimic the S5 main-experiments figure suite at the per-cell level, plus a new `dummy` split with one-hot packed probes.
- **Sources consulted**:
  - [scripts/run_real_augmented.py](../../scripts/run_real_augmented.py) `_emit_all_figures` — verified the worst-to-best ranking rule (one-third worst + one-third best + one-third middle, sorted ascending by per-sample R²) and the canonical run-level distribution figs (R² + bin accuracy).
  - [src/mpinv/analysis/plots/r2_distribution.py](../../src/mpinv/analysis/plots/r2_distribution.py) `build_metric_distribution_figure` — verified the existing layout (top: per-split histograms; bottom: cross-split violin plot) is the correct reuse target.
  - [src/mpinv/analysis/plots/dummy_probe.py](../../src/mpinv/analysis/plots/dummy_probe.py) — verified the `(pred_packed, active_indices)` API; reused unchanged by `build_split_report`.
  - [src/mpinv/analysis/plots/field_comparison.py](../../src/mpinv/analysis/plots/field_comparison.py) `build_field_comparison_grid_figure` — verified `n=8` worst-to-best multi-row layout signature.
  - [final_exps_13_05.txt](../../final_exps_13_05.txt) — verified S5 production defaults `--n-train-eval-samples 1024`, `--n-figure-grid-samples 8`, `--eval-batch-size 256` (now wired as the `report:` group defaults).
- **Verified facts**:
  - The per-sample metrics required for the violin-distribution plots ([per_sample_weighted_r2_P](../../src/mpinv/analysis/metrics/field_metrics.py), `per_sample_bin_accuracy_P`, `per_sample_spearman_rho_P`) already existed; `per_sample_weighted_nrmse_P` and `per_sample_packed_mse` are new but follow the same numpy-on-float64 contract.
  - The dummy split is interpreted **per packed slot** (`4 K = 140` samples for `L = 5`), not per complex coefficient (`2 K = 70`). Rationale: matches the existing `dummy_probe.py:build_dummy_probe_figure` API which takes a single `active_indices: list[int]`. Per-coefficient interpretation would require a 2-active-index variant of that plot.
  - `coef_histograms.pdf` is **degenerate** for the dummy split (target_packed is one-hot per row), so `build_split_report` skips it for that split. Other plots are emitted.
  - `pca_explained_variance.pdf` is emitted **only** for `val`, where the PCA pipeline (when used) lives. Other splits do not have a fitted PCA per their feature path.
  - Train-augmented is sub-sampled to `n_train_eval_samples=1024` rows by deterministic head slicing (matches S5; full 10 000 augmented rows are too large to forward + plot per cell).
- **Project decisions (binding for R7)**:
  - Per-cell scope: every cell of every multirun emits the full suite. Disk budget ~170 MB for Steps 1+2+3; acceptable.
  - Output layout per cell: `report/<split>/{coef_scatter,per_l_breakdown,field_comparison_grid,coef_histograms?}.pdf` plus `report/<split>/dummy_probe.pdf` for the dummy split, plus run-level `report/{r2,bin_accuracy,spearman,nrmse,coef_mse}_distribution.pdf`.
  - Composite-metric selection (`scripts/select_best_step.py`) is unchanged — it reads `metrics.json`, the new figures are auxiliary.
- **Use in framework**:
  - **NEW** [src/mpinv/data/dummy_probe.py](../../src/mpinv/data/dummy_probe.py) — `build_single_mode_probe(basis, l_max, amplitude=1.0)`.
  - **NEW** keys in [configs/data/real_augmented_l5.yaml](../../configs/data/real_augmented_l5.yaml): `include_dummy_probe`, `dummy_amplitude`.
  - **NEW** entries in [src/mpinv/data/real_augmented_pipeline.py](../../src/mpinv/data/real_augmented_pipeline.py) → `out["P_dummy"]`, `out["packed_dummy"]`, `out["dummy_active_indices"]`, `out["n_dummy"]`.
  - **NEW** function `build_split_report` in [src/mpinv/analysis/reports/run_report.py](../../src/mpinv/analysis/reports/run_report.py) plus the `_pick_grid_indices` helper.
  - **NEW** wrappers in [src/mpinv/analysis/plots/r2_distribution.py](../../src/mpinv/analysis/plots/r2_distribution.py): `build_spearman_distribution_figure`, `build_nrmse_distribution_figure`, `build_coef_mse_distribution_figure`.
  - **NEW** per-sample metrics in [src/mpinv/analysis/metrics/](../../src/mpinv/analysis/metrics/): `per_sample_weighted_nrmse_P` and `per_sample_packed_mse` (re-exported from the package `__init__`).
  - **NEW** config group [configs/report/default.yaml](../../configs/report/default.yaml) wired into [configs/train.yaml](../../configs/train.yaml) `defaults:`.
  - [src/mpinv/cli/train.py](../../src/mpinv/cli/train.py) §"9. End-of-run analysis report" rewritten as a per-split loop using `build_split_report`; the legacy single-split `build_run_report` call path is gone (the function is preserved for `scripts/run_real_augmented.py`'s `_figures_for`).
  - Tests: [tests/unit/test_dummy_probe_data.py](../../tests/unit/test_dummy_probe_data.py), [tests/unit/test_real_augmented_pipeline.py](../../tests/unit/test_real_augmented_pipeline.py), [tests/unit/test_split_report.py](../../tests/unit/test_split_report.py), [tests/unit/test_per_sample_metrics_and_violins.py](../../tests/unit/test_per_sample_metrics_and_violins.py) — 25 new tests, all green; full suite at 185 passed / 1 skipped.

---

## Phase 2 — Plan

The full numbered sub-task list lives in [.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md](../../.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md) (Часть Б, items B1–B7 + Часть В). Each yaml file in [paper/final_experiments/](.) maps to one experiment cell or sweep.

The R7 extension lives in [.cursor/plans/per-split-plot-suite_e0f631cc.plan.md](../../.cursor/plans/per-split-plot-suite_e0f631cc.plan.md).

## Phase 3 — Manifest summary

This file is the manifest. Every API touch in B1–B6 references one of the entries R1–R6 above; the per-split plot suite is backed by R7.
