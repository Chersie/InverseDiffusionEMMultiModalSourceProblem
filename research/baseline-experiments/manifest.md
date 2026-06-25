# Research manifest — Baseline experiments block (`baseline-experiments`)

Task slug: `baseline-experiments`. Date: 2026-05-08.

This manifest is the Phase 3 record (per [RESEARCHER.md](../../RESEARCHER.md)) for the baseline-experiments block specified in [experiments/baseline/baseline_experiments.md](../../experiments/baseline/baseline_experiments.md). Every external claim that backs a feature, augmentation, dataset regime, or HPO pattern landing in `src/mpinv/**` for this block must trace back to one of the entries here. Honest gaps — i.e. things the literature does **not** answer — are recorded as such, not papered over.

---

## Phase 1 — Comprehension

**Task understood as**: implement a minimum useful subset of new infrastructure (two new feature extractors, four physically consistent augmentations, multi-split end-of-run evaluation, dataset-distribution-statistics CLI, a 2-layer MLP config grid) on top of the rebuilt `mpinv` framework, then run a structured experiment matrix (Stages S0/S1/S2/S3) across 5 model sizes, 3 losses, 6 feature pipelines, 5 augmentation conditions, and 4 generation regimes. Produce a written report at [experiments/baseline/baseline_experiments_report.md](../../experiments/baseline/baseline_experiments_report.md) of which combinations actually move the four canonical metrics across train / val / synthetic-test / real-antenna holdout.

**Technology / topic inventory**:

| Topic | Status | Resolved by |
|---|---|---|
| Phaseless spherical-harmonic recovery: ambiguity classification and standard generators | `[NEEDS RESEARCH]` | R1 |
| ML-based phaseless spherical / antenna inversion 2024–2026 | `[NEEDS RESEARCH]` | R2 |
| Mode-dropout / coefficient-masking regulariser cite | `[NEEDS RESEARCH]` | R3 |
| Phaseless SH sampling theorems | `[NEEDS RESEARCH]` | R4 |
| Real-antenna far-field datasets, public availability | `[NEEDS RESEARCH]` | R5 |
| Hydra 1.3 + Optuna 4.x + MLflow 3.x sweep pattern, 2026 | `[NEEDS RESEARCH]` | R6 |
| `(P, packed)` consistency under augmentation | `[KNOWN]` | inherited from [presentation/ch1_full.md](../../presentation/ch1_full.md) §1.5–1.7 |
| Existing framework integration points (`src/mpinv/data/synthetic_generator.py`, `src/mpinv/cli/_builders.py`, `src/mpinv/cli/train.py`) | `[KNOWN]` | inherited from [research/framework-rebuild/manifest.md](../framework-rebuild/manifest.md) |

---

## R1 — Bangun 2020: phaseless spherical-harmonic recovery

- **Citation**: A. Bangun, "Signal recovery on the sphere from compressive and phaseless measurements", PhD dissertation, RWTH Aachen University, 2020. doi:`10.18154/RWTH-2020-03041`.
- **Sources consulted**:
  - RWTH publications portal <https://publications.rwth-aachen.de/record/783873> (accessed 2026-05-08).
  - Bangun et al., "Sensing Matrix Design and Sparse Recovery on the Sphere and the Rotation Group" <https://arxiv.org/abs/1904.11596> (preprint version of dissertation results; accessed 2026-05-08).
- **Verified facts**:
  - Chapter 6 §6.2 establishes that scalar complex spherical-harmonic phaseless measurements admit the ambiguity `g_l^k = (-1)^k · conj(f_l^{-k})` beyond global phase, derived from the conjugation property `Y_l^k = (-1)^k · conj(Y_l^{-k})`. This is the *reflected-conjugate ambiguity* used in §1.7 of the chapter and in `src/mpinv/analysis/metrics/mode_metrics.py:reflected_conjugate_aware_loss`.
  - Proposition 6.2.1 exhibits explicit *sampling-pattern-dependent* ambiguities for which the SH sensing matrix becomes rank-deficient. The project's 1° equiangular grid (poles excluded) is **not** one of those patterns, so this class of ambiguity is *inactive* in our setup.
  - The synthetic-data experiments in the dissertation use white isotropic complex Gaussian SH coefficients over the active mode set, with no explicit `(l+1)^{-α}` colouring. There is no canonical `α` defined for "realistic antenna spectra" in the dissertation.
- **Honest gap**: the dissertation does **not** report a sampling theorem of the form "for bandlimit `L` you need `m ≥ f(L)` measurements for unique phaseless recovery up to trivial ambiguities". The only quantitative bounds available are the generic PhaseLift `m ≥ C₀ N log N` (R4a) and Bangun's compressive `m ≳ N^{1/6} s log³(s) log(N)` (R1, Theorem 4.4.1), neither of which is the same statement. The block does not cite a tighter bound.
- **Use in this block**:
  - The ambiguity-aware coefficient metric `report/coef_mse_amb_aware` (`src/mpinv/analysis/metrics/mode_metrics.py:31-60`) is the existing implementation of the §6.2 reflected-conjugate map.
  - The default synthetic generator regime `gaussian` (`src/mpinv/data/synthetic_generator.py:91-97`) matches the white-isotropic-complex-Gaussian baseline of Bangun's experiments. The block treats `α` for `colored` as a swept hyperparameter, not a literature-grounded default.

---

## R2 — Recent ML-based phaseless spherical / antenna inversion (2024–2026)

- **Sources consulted (all accessed 2026-05-08)**:
  - Schmid, Eibert et al., "Phaseless Spherical Near-Field Antenna Measurement using Wirtinger Flow with Masking", *Sensors* **25**(18):5637, 2025. <https://www.mdpi.com/1424-8220/25/18/5637>. doi:`10.3390/s25185637`.
  - Pacheco, Foged et al., "Phaseless near-field measurements via deep learning: a contemporary survey", *Measurement* (2025). <https://www.sciencedirect.com/science/article/pii/S0263224125001076> (paywalled abstract; verified title/abstract/DOI only).
  - PIEDNet preprint (2025): "Phase Information Embedded Deep Network for phaseless spherical near-field reconstruction". arXiv:`2502.09921` <https://arxiv.org/abs/2502.09921>.
  - APCAP 2024 conference index <https://ieeexplore.ieee.org/xpl/conhome/10827693/proceeding> for adjacent ML-vs-near-field papers.
- **Verified facts**:
  - Schmid et al. (2025) train on **simulated** AUTs (Standard Gain Horn, broadside slot, mmVAST array) using a Wirtinger-flow optimisation loop with a measurement mask; evaluate on simulated data with phase-noise injection; **no public real-antenna held-out dataset**. Inputs are amplitude-only spherical NF samples; the output is the recovered complex tangential field, not coefficients. Train sets are O(10^3–10^4) simulated patterns. No explicit augmentation beyond the random-phase initialisation that Wirtinger flow requires.
  - PIEDNet (2025) is a *near-field* phase-retrieval network on a different geometry; not a `P_UT(θ,φ) → (a^E, a^M)` mapping at fixed `L`. It does not use the project's grid (179×360, poles excluded). It is cited as adjacent prior art only.
  - Pacheco et al. (2025) is a survey, not a benchmark; provides taxonomy but no canonical reference dataset.
- **Honest gap (load-bearing)**: **no 2024–2026 ML paper performs the exact `P_UT → (a^E, a^M)` mapping at `L=15` on the same 360×179 grid with a real-antenna held-out test.** Therefore:
  - The baseline-experiments report cannot calibrate against a literature score.
  - The closest analogues (Schmid 2025, PIEDNet 2025) are on a different problem geometry and a different output space.
  - The block reports **relative improvements vs. the linear baseline**, not vs. any external number.
- **Use in this block**:
  - The "noisy-input / clean-target" dataset contract for `field_additive_noise` (additive Gaussian noise on `P` with unchanged `packed` targets) follows the same robustness-evaluation convention as Schmid et al. (2025) §5.
  - The recommendation in the report's Limitations section that this work is best framed as a *novel inverse-problem formulation*, not a reproduction of existing baselines, follows directly from this gap.

---

## R3 — Mode-dropout / coefficient-masking augmentation cite

- **Citation**: Liu, Wang et al., "DropAnSH-GS: Dropout-Augmented Spherical Harmonics for Gaussian Splatting", arXiv:`2602.20933`, February 2026.
- **Sources consulted (accessed 2026-05-08)**:
  - arXiv abstract page <https://arxiv.org/abs/2602.20933>.
  - Mihaylova et al., "Spectral Dropout for Improved Generalization in CNNs", *ICASSP* 2018, doi:`10.1109/ICASSP.2018.8462352` <https://ieeexplore.ieee.org/document/8462352> (older but cited as the original spectral-dropout precedent).
- **Verified facts**:
  - DropAnSH-GS applies independent Bernoulli dropout to high-degree SH coefficients during training to regularise the learned SH basis used by Gaussian Splatting renderers. It does *not* re-synthesise the rendered output through a forward operator after the drop; it relies on the renderer being differentiable and the dropped basis components being part of the learned representation.
  - Spectral dropout (Mihaylova 2018) drops 2-D-DFT coefficients during the forward pass and is the closest classical precedent for "dropout in a transform domain".
- **Honest gap**: neither paper drops *target* coefficients in a supervised regression setting where the input `P` is computed from the un-dropped target. **Mode-dropout in the sense of "set `a_lm = 0` for some `(l,m)` and re-synthesise `P`"** — which is what the legacy framework's `mode_dropout_prob` knob does (`src/mpinv/data/synthetic_generator.py:119-122`) — is a *distribution-shaping* knob, not an augmentation in the sense of "perturb a training pair while keeping it consistent". The S0–S3 stages of the baseline block treat `mode_dropout_prob` as a **generation knob**, not as an augmentation; the four augmentations implemented for those stages (`coef_phase_rotation`, `coef_additive_noise`, `field_additive_noise`, `field_phi_roll`) are documented to either re-synthesise `P` or explicitly declare an "input-noise vs. clean-target" contract.
- **Amendment 2026-05-13** (real-augmented sub-block, R7): the "honest gap" on a published "drop-and-resynthesise" augmentation **still stands**, but the project now implements the operation as a post-hoc augmentation (`CoefModeDropoutConfig` in `src/mpinv/data/augment.py`) anyway, because R7 needs it for the limited-data setting where there is no synthetic generator to push the dropout into. The augmentation re-synthesises `P` from the masked coefficients via the same `_synthesize` helper used by `coef_additive_noise`, so `(P, packed)` consistency is preserved. The choice is documented as a project-internal recipe, not a literature reproduction.
- **Use in this block**:
  - The S3 generation-regime ablation includes `sparse 10%` (`src/mpinv/data/synthetic_generator.py:101-104`) as one of the four regimes; this is the structural-sparsity regime, distinct from a per-step dropout augmentation.
  - The real-augmented sub-block (R7) uses `coef_mode_dropout` as one of three composed augmentations.
  - The report's "honest limitations" section records that no published "dropout on coefficients with re-synthesis" augmentation cite exists for our exact setting.

---

## R4 — Phaseless SH sampling theorems

### R4a — PhaseLift generic bound

- **Citation**: Candès, Strohmer, Voroninski, "PhaseLift: Exact and Stable Signal Recovery from Magnitude Measurements via Convex Programming", *Comm. Pure Appl. Math.* **66**(8):1241–1274, 2013. doi:`10.1002/cpa.21432`.
- **Source consulted**: arXiv preprint <https://arxiv.org/abs/1109.4499> (accessed 2026-05-08).
- **Verified fact**: for `N`-dimensional complex signals with `m` random Gaussian phaseless measurements, exact recovery up to global phase holds w.h.p. when `m ≥ C₀ N log N` for a fixed constant `C₀`. This is a *generic* bound; it does not exploit any spherical-harmonic structure.

### R4b — Bangun compressive bound

- **Citation**: Bangun 2020 (R1), Theorem 4.4.1.
- **Verified fact**: for `s`-sparse SH coefficient vectors at bandlimit `N`, the compressive phaseless bound is `m ≳ N^{1/6} · s · log³(s) · log(N)`. The exponent 1/6 in `N` is dictated by the sphere-specific Marcinkiewicz–Zygmund inequality, not the underlying phaseless geometry.

### R4c — Honest gap on bandlimit-`L`-specific bound

- For the project's specific setting (fixed `L=15`, full-density `K=255`, dense 360×179 measurements), neither R4a nor R4b applies tightly:
  - R4a is overkill (the project has `m = 64,440 >> N log N` measurements, so the *generic* bound is automatically satisfied).
  - R4b requires sparsity `s` which the project's gaussian regime does not have.
- **No known tighter bound** for "bandlimit-`L` phaseless dense SH measurement at `(360, 179)` grid up to global phase + reflected-conjugate". The block does not state one.

---

## R5 — Real-antenna far-field datasets

- **Sources consulted (all accessed 2026-05-08)**:
  - NIST Antenna Metrology page <https://www.nist.gov/programs-projects/antenna-metrology-and-calibration> (no dataset download link; only services).
  - IEEE AP-S benchmark / standards index <https://standards.ieee.org/ieee/1720/5938/> for Std 1720-2012 (recommended practice; not a dataset).
  - Antenna Magus product page <https://www.altair.com/feko-antenna-magus/> (commercial, paywalled).
  - Open Antenna Patterns (OAP) <https://www.openantennapatterns.org/> (search returned 0 results in 2026; site appears defunct).
- **Verified facts**:
  - There is **no** public spherical-NF real-antenna dataset on a 360×179 grid that ships with paired multipole-coefficient targets at `L=15`. NIST publishes calibration *services*, not a dataset. IEEE AP-S 1720-2012 is a measurement *standard*, not data.
  - Antenna Magus and CST Studio Suite ship libraries of **simulated** antennas; both are commercial.
- **Honest gap (binding for this block)**: the project's "real-antenna holdout" (`src/mpinv/data/real_antenna_loader.py`) is loaded from a corpus the user must supply at `data/raw/real_antenna/`. As of 2026-05-08 the directory is empty (verified by `Glob` over `data/raw/real_antenna/**/*.txt` → 0 files). The S1/S2/S3 stages will run with the holdout column **N/A** unless and until a corpus is supplied. The plan and report explicitly mark this.

---

## R6 — Hydra 1.3 + Optuna 4.x + MLflow 3.x sweep pattern

- **Source consulted (accessed 2026-05-08)**:
  - PyPI page for `hydra-optuna-mlflow-sweeper` v0.1.3, released 2026-05-01 <https://pypi.org/project/hydra-optuna-mlflow-sweeper/>.
  - `optuna-integration` 4.5.x docs <https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.MLflowCallback.html>.
- **Verified facts**:
  - The pattern `mlflow.start_run(nested=True, parent_run_id=...)` plus `optuna_integration.MLflowCallback(tracking_uri=..., metric_name=..., create_experiment=False, mlflow_kwargs={"nested": True})` is the canonical 2026 pattern for nested-run sweeps. The project's [src/mpinv/cli/sweep.py](../../src/mpinv/cli/sweep.py) already implements it (verified against R3 of [research/framework-rebuild/manifest.md](../framework-rebuild/manifest.md)).
  - `hydra-optuna-mlflow-sweeper` v0.1.3 is the same pattern packaged as a Hydra plugin; it is **not** a new requirement and not adopted here. We mention it only as confirmation that the in-tree pattern matches current 2026 community usage.
- **Use in this block**:
  - The S1/S2/S3 stages use `mpinv-train` directly with `--multirun` (Hydra 1.3 feature) for the 5×3×6 cartesian product, not the Optuna sweeper. Optuna is reserved for hyperparameter search, which this baseline block does not perform.

---

## R7 — Limited-real-data training via on-manifold augmentation (real-augmented sub-block)

- **Task slug extension**: `real-augmented`. Date: 2026-05-13. Sub-block of `baseline-experiments` (no separate manifest directory; entry recorded here).
- **Sources consulted (all accessed 2026-05-13)**:
  - Shorten & Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", *J. Big Data* 6:60, 2019. doi:`10.1186/s40537-019-0181-8`. Establishes the general claim that *physically consistent* augmentations of a labelled dataset can substitute for additional labelled samples up to the point where the augmentation distribution covers the test distribution.
  - Cohen, Geiger, Köhler, Welling, "Spherical CNNs", *ICLR* 2018. arXiv:`1801.10130`. Establishes that a `φ`-rotation of a spherical signal is an exact symmetry of the equiangular sphere grid (used by R-augmentation `field_phi_roll`).
  - Schmid et al. 2025 (R2): the only adjacent published recipe; it does **not** train on real-antenna data with augmentation, so calibration against it is not possible.
- **Verified facts**:
  - `field_phi_roll` is exact at the project's 1° azimuth grid because samples align with `φ_k = 2π k / n_phi` (already verified by `tests/unit/test_augment.py::test_field_phi_roll_consistent_with_coef_rotation`). Up to 360 distinct rolls per source sample are physically valid; with `≤ 100` source samples the augmentation can produce up to `36 000` consistent `(P, packed)` pairs without leaving the manifold.
  - `coef_mode_dropout` re-synthesises `P` from the masked coefficients, so the resulting `(P, packed)` pair is exact at the project's bandlimit (verified by `tests/unit/test_augment.py::test_coef_mode_dropout_resynthesis_consistent`). Composing it with `field_phi_roll` is well-defined because both preserve consistency.
  - `field_additive_noise(σ ≪ 1)` perturbs the model's input only; it does not generate distinct training pairs in target space. At `σ = 1e-8` it is effectively numerical noise — its role here is to tell the model "the training distribution has measurement noise floor", not to expand the dataset.
- **Honest gaps (load-bearing for this sub-block)**:
  1. **No published study calibrates the recipe `field_phi_roll ∘ coef_mode_dropout ∘ field_additive_noise(σ)` for limited real-antenna data.** The recipe is a project-internal experimental design choice (user-specified, 2026-05-13 chat). The block reports relative metrics only; we do not claim this recipe is optimal.
  2. **No published study quantifies how augmenting `N=100` real samples to `N_aug=10000` compares with training on `10000` synthetic samples on the same target distribution.** The S3/S4 results give an in-distribution synthetic baseline; the real-augmented sub-block gives an out-of-distribution real baseline. They are not directly comparable on a single metric, and the report says so.
  3. **The L=5 truncation introduces a fidelity loss on the real-antenna samples**: their measured `P` contains contributions from `l > 5` that are not representable at the project's bandlimit. The driver re-synthesises `P` from the truncated `(a^E, a^M)` at load time so `(P, packed)` is consistent on the L=5 manifold, but this means the model is trained on the **L=5-truncated** real antennas, not the originals. This is a stated, deliberate fidelity / runtime trade-off; switching to L=15 is a `--l-max 15` flag away.
  4. **Basis-convention mismatch in the legacy multipole files (resolved 2026-05-13)**. The legacy multipole files at `~/Desktop/diplom_dump/E+multip/Multipoles_in_plane/` and `~/Desktop/diplom/data/external/Rotation/Multipoles/` are written in a *different* VSH convention than the project: an extra `-(1j)^(l+1)` global phase per mode and a sign flip on the electric family relative to Jackson §9 / `mpinv.data._basis_cache`. Reusing those coefficients with the project's forward operator yielded `||P_meas|| / ||P_resyn|| ≈ 250` despite L=5 capturing >99% of multipole energy. The fix is to **discard the legacy multipole files** and re-derive `(a^E, a^M)` from the measured `(E_θ, E_φ)` via `mpinv.data.basis_decomposer.decompose_field_to_packed`, which is the analytic inverse of the project's forward operator (orthogonal projection under the standard `sin θ`-weighted inner product). The decomposer is verified by `tests/unit/test_basis_decomposer.py` (single-mode and random roundtrip on the tiny grid; full-grid relative error O(1e-4)). After re-derivation, the imported corpus has post-import `||P_meas - P_resyn||₂ / ||P_meas||₂` mean=0.012, median=0.004, max=0.081 across 396 samples, i.e. L=5 captures > 98.8% of energy on average for these antennas. **No external citation backs this convention reconciliation** — it is a project-internal verification anchored in the project's own basis definition (`src/mpinv/data/_basis_cache.py:133-171`) and the analytic inverse (`src/mpinv/data/basis_decomposer.py`).
- **Use in this block**:
  - The augmentation library gains `coef_mode_dropout` (and the manifest amendment in R3 documents the change of stance).
  - The driver `scripts/run_real_augmented.py` implements the load → split-by-sample-id → augment → train → eval pipeline. The 80/20 sample-id split is the standard practice anchor (Shorten & Khoshgoftaar 2019, §3.2 — "augmented copies of a held-out sample must not appear in the validation set").
  - The sub-block reports its results as a separate stage (S5, "real-augmented") in the project report; it is **not** mixed into the S1/S2/S3 synthetic stages because the data distribution and the question being asked are different.

---

## Hard-rule check (RESEARCHER.md "Hard rule on empty research")

Phase 3 produced new information for R1, R2, R3, R4, R5, R6, and R7 (the real-augmented amendment). All `[NEEDS RESEARCH]` items from Phase 1 are resolved before Phase 4 implementation begins. Honest gaps are recorded in R2 (no 2024–2026 baseline for the exact task), R3 (no published "drop-and-resynthesise" augmentation cite — implemented anyway as a project-internal recipe per R7), R4c (no bandlimit-`L`-specific phaseless SH sampling theorem), R5 (no public real-antenna holdout corpus on this grid), and R7 (no published calibration of the limited-real-data + on-manifold-augmentation recipe).

## Sources I consulted but did not use

- `pyshtools` — alternative SH library, ruled out for the same reason as in [research/framework-rebuild/manifest.md](../framework-rebuild/manifest.md).
- `e3nn` — equivariant neural networks; not used because the project's MLP family is the explicit scope of `experiments/baseline/baseline_experiments.md`.
- `wandb`, `clearml` — alternative trackers; out of scope.

## Anchor list — every external claim in this block traces to one of:

| Code / config / report claim | Manifest entry |
|---|---|
| Reflected-conjugate ambiguity in metric / aug discussion | R1 |
| Choice of `gaussian` as default generator regime; `α` swept not pinned | R1 |
| "No literature baseline for the exact task" caveat in report | R2 |
| Noisy-input / clean-target convention for `field_additive_noise` | R2 |
| Mode-dropout treated as generation knob, not augmentation | R3 |
| No tighter bandlimit-`L` sampling theorem stated | R4 |
| Real-antenna holdout marked N/A when absent | R5 |
| `mlflow nested + optuna_integration.MLflowCallback` pattern still current | R6 |
| `coef_mode_dropout` post-hoc augmentation (departure from R3 stance) | R3 (amended) + R7 |
| 80/20 sample-id split for limited-real-data + augment | R7 (Shorten & Khoshgoftaar 2019 §3.2) |
| `field_phi_roll` is exact on the 1° equiangular azimuth grid | R7 (Cohen et al. 2018) + existing test |
| Recipe `field_phi_roll ∘ coef_mode_dropout ∘ field_additive_noise(σ)` not calibrated | R7 (honest gap #1) |
| L=5 truncation + P re-synthesis at load time for consistency | R7 (honest gap #3) |
| Re-derivation of `(a^E, a^M)` from measured `E` for the legacy holdout | R7 (honest gap #4) |
| Basis-convention mismatch in legacy multipole files (resolved by re-derivation) | R7 (honest gap #4) |
