# Final experiments — operational guide

Run order, invocation cheatsheet, and selection-helper usage for the four-step
plan derived from [paper/proposal.md](../proposal.md). Plan-of-record:
[.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md](../../.cursor/plans/final-experiments-from-proposal_6f3b24b3.plan.md).
Research manifest backing every API touch:
[paper/final_experiments/manifest.md](manifest.md).

> All runs use the real-augmented L=5 pipeline at
> [configs/data/real_augmented_l5.yaml](../../configs/data/real_augmented_l5.yaml)
> (200 sources / 180 train / 20 val_real / 100 holdout_real / 10 000 augmented).
> Selection metric (lower is better):
>   `composite = report/val/field_nrmse_w − 0.5 · report/val/spearman_rho_P`.

## Layout

| File | Contents |
|---|---|
| `manifest.md` | Phase 3 research manifest (R1..R6); cite this from the thesis. |
| `step0_base.yaml` | 1 cell — baseline (mlp_4x60 + coef_mse + raw_flat + no scheduler). |
| `step1_pure.yaml` | 2 cells — coef_mse, physics_power. |
| `step1_mixed.yaml` | 5 cells — physics_power_mixed × `coef_aux_weight ∈ {0.001, 0.01, 0.1, 1.0, 10.0}`. |
| `step1_rank.yaml` | 5 cells — physics_power_rank × `rank_bin_weight ∈ {0.001, 0.01, 0.1, 1.0, 10.0}`. |
| `step1_combined.yaml` | 9 cells — physics_power_combined × `{coef_aux × rank_bin} ∈ {0.01, 0.1, 1.0}²`. |
| `step2_features.yaml` | 7 cells — `features ∈ {raw_flat, raw_plus_sh, power_pca, power_pca_small, pca_cv, cv_only, subsample_stride4}`; loss = Step 1 winner (placeholder). |
| `step3_scheduling.yaml` | 6 cells — `backbone_policy × truncate_target_to_active_band`; model = multi_head_mlp_4x60; loss = Step 1 winner; features = Step 2 winner (placeholders). |

## Hydra invocation

The yamls are **not** under `configs/`, so we add this directory as an extra
search path via `--config-dir`. The primary search path stays at `configs/`
(the default of `mpinv-train`), so all standard groups (`data`, `features`,
`model`, `loss`, …) resolve normally.

### Single-cell (Step 0)

```bash
uv run mpinv-train --config-dir paper/final_experiments --config-name step0_base
```

### Multirun cell sets (Steps 1–3)

The yamls have `hydra.mode: MULTIRUN` and `hydra.sweeper.params: {…}`
baked in, so the `--multirun` flag is **not** required at the CLI; Hydra
will pick it up from the config.

```bash
# Step 1 — regularization (4 sub-sweeps, 21 cells total).
uv run mpinv-train --config-dir paper/final_experiments --config-name step1_pure
uv run mpinv-train --config-dir paper/final_experiments --config-name step1_mixed
uv run mpinv-train --config-dir paper/final_experiments --config-name step1_rank
uv run mpinv-train --config-dir paper/final_experiments --config-name step1_combined

# Step 2 — feature sweep (7 cells). Update step2_features.yaml first
# (see "Selection helper" below).
uv run mpinv-train --config-dir paper/final_experiments --config-name step2_features

# Step 3 — scheduling sweep (6 cells). Update step3_scheduling.yaml first.
uv run mpinv-train --config-dir paper/final_experiments --config-name step3_scheduling
```

Each multirun produces output under `multirun/<experiment_name>/<timestamp>/<job_num>_<override_dirname>/`,
with `metrics.json`, `report/`, `checkpoints/`, and (Step 3 only) `stage_reports.json`.

### Smoke-test mode (no real corpus on disk)

If `data/raw/real_antenna/` is empty, override `data.smoke_test=true` to
substitute synthetic colored α=2 samples for the holdout corpus.
Code-path check only — results from smoke mode are not meaningful.

```bash
uv run mpinv-train --config-dir paper/final_experiments --config-name step0_base \
    data.smoke_test=true
```

## Selection helper

After each multirun, pick the best cell by composite metric:

```bash
uv run python scripts/select_best_step.py \
    --metrics-glob 'multirun/final_step1_*/**/metrics.json' \
    --split-prefix val \
    --output paper/final_experiments/step1_winner.json
```

The output JSON contains the winner (the cell with the lowest composite),
the top-3 ranked cells, and metadata about the metric formula. Open the
winner's `metrics_path`, look at the sibling `.hydra/overrides.yaml` (Hydra
writes it next to every multirun job), and copy the relevant overrides into
`step2_features.yaml` (or `step3_scheduling.yaml`) before running the next
step.

Concrete edit pattern for Step 2:

1. Open `paper/final_experiments/step1_winner.json` and find
   `winner.metrics_path = "multirun/final_step1_rank/.../<job_num>_.../metrics.json"`.
2. Open `multirun/final_step1_rank/.../<job_num>_.../.hydra/overrides.yaml`
   and read the override list (e.g. `["loss.rank_bin_weight=0.1"]`).
3. In `step2_features.yaml`:
   - Set `defaults: - override /loss: physics_power_rank` (or whatever loss
     family won).
   - Add the resolved knobs at top level, e.g.
     ```yaml
     loss:
       rank_bin_weight: 0.1
     ```

Same pattern for Step 3, plus the features override.

## Output schema

Each cell writes:

- `metrics.json` — flat `{metric_key: float}` dict; consumed by
  `select_best_step.py`. Keys use the canonical `report/<split>/<name>`
  namespace. Splits emitted: `train_aug`, `val`, `test`, `holdout`, `dummy`
  (modulo splits that do not exist for a given pipeline; e.g. synthetic
  pipelines lack `holdout` unless `cfg.holdout` is set).
- `report/` — per-cell figure suite (built by
  [src/mpinv/analysis/reports/run_report.py:build_split_report](../../src/mpinv/analysis/reports/run_report.py)
  per split, plus run-level distribution plots):

  | Path | What |
  |---|---|
  | `report/<split>/coef_scatter.pdf` | predicted vs target packed coefficients (per split) |
  | `report/<split>/coef_histograms.pdf` | marginal histograms of target packed coefficients (skipped for `dummy` — degenerate one-hot) |
  | `report/<split>/per_l_breakdown.pdf` | per-l error decomposition |
  | `report/<split>/field_comparison_grid.pdf` | `n_grid_samples` (=8) rows ranked worst→best by per-sample sin-θ-weighted R² |
  | `report/<split>/dummy_probe.pdf` | only for `dummy`: \|pred\| per packed slot with the active slot highlighted |
  | `report/<split>/pca_explained_variance.pdf` | only for `val` when a PCA feature pipeline is used |
  | `report/r2_distribution.pdf` | run-level: per-split histograms (top) + cross-split violin (bottom) for per-sample R² |
  | `report/bin_accuracy_distribution.pdf` | same layout, hard rank-bin accuracy (`n_bins = 2 l_max + 1`) |
  | `report/spearman_distribution.pdf` | same layout, per-sample Spearman ρ |
  | `report/nrmse_distribution.pdf` | same layout, per-sample sin-θ-weighted NRMSE |
  | `report/coef_mse_distribution.pdf` | same layout, per-sample MSE in packed-coefficient space |

- `checkpoints/` — model + optimiser state every
  `callbacks.checkpoint_every_n_epochs` epochs (single-stage) or
  `checkpoints/stage_<k>/` (Step 3, staged).
- `stage_reports.json` — per-stage outcome list (Step 3 only).
- `.hydra/` — Hydra-internal: `config.yaml`, `overrides.yaml`,
  `hydra.yaml`. Read `overrides.yaml` to recover the cell's parameters.

The report suite is configured via the `report:` group ([configs/report/default.yaml](../../configs/report/default.yaml)):

| Knob | Default | What |
|---|---|---|
| `report.n_train_eval_samples` | 1024 | rows used for the `train_aug` split's report (subsample of the full augmented train) |
| `report.n_grid_samples` | 8 | rows in `field_comparison_grid.pdf` |
| `report.eval_batch_size` | 256 | chunk size for the predict + decode pass |

## Wall-time budget (informational)

35 cells total × 30 epochs × ~30 batches/epoch. Per-cell wall time on M-mac
M3 CPU is roughly 0.5–1.5 h; on a single GPU expect 5–15 min. Recommend
running Steps 1–3 on GPU and using `trainer.max_epochs=10` for a fast
preliminary screen of Step 1 if compute is tight.
