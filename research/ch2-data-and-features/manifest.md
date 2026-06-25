# Research manifest — Chapter 2 (data generation and featurisation)

Task slug: `ch2-data-and-features`. Date: 2026-06-01.

This manifest is the Phase 3 record of every external source and every project-internal artefact consulted in support of `presentation/paper_full.md`, Chapter 2. Every factual claim, numerical default, formula, citation, and pipeline description in that chapter must trace back to one of the entries below.

---

## Phase 1 — Comprehension

**Task understood as**: turn the Russian outline of Chapter 2 in `presentation/header.md` into a single English-prose Chapter 2 body, written one subsection at a time into `presentation/paper_full.md`, in the same academic register as `presentation/ch1_full.md`. The chapter covers (i) the synthetic data generator and its augmentations, (ii) the real-antenna data pipeline, (iii) the three input-mode regimes (power / magnitude / complex), (iv) randomised PCA, (v) two additional CV-style features (FFT radial spectrum, spherical-harmonic spectral power), and (vi) feature/target normalisation. The canonical thesis truncation is `L=5` (corrected in Chapter 1 in this same session); the experiments operate at that bandlimit.

**Topic inventory**:

| Topic | Status | Resolved by |
|---|---|---|
| Synthetic coefficient generation: modes and composable knobs | `[NEEDS RESEARCH]` (project-internal) | R1 |
| Augmentation primitives and their physical-consistency contracts | `[NEEDS RESEARCH]` (project-internal + lit) | R2 |
| Real-antenna file format, analytic decomposition, L=5 residuals | `[NEEDS RESEARCH]` (project-internal) | R3 |
| Real-augmented pipeline: split, augmentation chain, train/val/holdout | `[NEEDS RESEARCH]` (project-internal) | R3b |
| Feature pipelines (modes, raw_flat, PCA, FFT-radial, SH-power, composite, normalisers) | `[NEEDS RESEARCH]` (project-internal) | R4 |
| Randomised SVD theory and complexity | `[NEEDS RESEARCH]` (external) | R5 |
| scikit-learn `PCA(svd_solver="randomized")` and `StandardScaler` reference | `[KNOWN]` (sklearn docs; standard pattern) | R6 |
| Tensor-layout contract: `(B, n_theta, n_phi)` | `[NEEDS RESEARCH]` (project-internal) | R7 |
| Project conventions inherited from Chapter 1 (L=5, K=35, packed dim 140, grid 360×179) | `[KNOWN]` | inherited verbatim from `presentation/ch1_full.md` §1.1–1.2 |

---

## R1 — Synthetic coefficient generator

- **Sources consulted (read-only)**:
  - `src/mpinv/data/synthetic_generator.py` lines 1–175.
  - `src/mpinv/core/packing.py` lines 28–35 (`L_MAX=15` framework cap, `K_MODES=255` for that cap, `PACKED_DIM=1020`). The thesis-level bandlimit is `L=5` per Chapter 1 §1.1; the generator accepts `l_max` as a config field and the canonical experiments override the cap accordingly.
  - `src/mpinv/data/_basis_cache.py` lines 50–145 (basis construction `(K, 2, 2, n_theta, n_phi)`, `meshgrid` with `indexing="ij"`).
- **Verbatim facts extracted**:
  - The generator samples per-batch independent complex coefficients `(a^E, a^M)` via `np.random.default_rng(seed)`, synthesises the tangential field through a cached VSH basis, and forms the power pattern `P = (E.real**2 + E.imag**2).sum(axis=1)` (lines 164–167).
  - Four coefficient-sampling **modes**, defined by `SyntheticGeneratorConfig.mode ∈ {"gaussian", "uniform", "colored", "sparse"}` (lines 46, 91–115):
    - **gaussian** (line 91–97): `a^X = (N(0,1) + i·N(0,1)) / √2`, i.i.d. per `(l,m)` per family.
    - **uniform** (line 98–100): `Re a^X, Im a^X ∼ U[−1, 1]` independently — an axis-aligned box in `ℂ`, not a disk.
    - **sparse** (line 101–104): Bernoulli active mask with `sparse_active_fraction` (default `0.1`); active modes drawn as `N(0,1)+iN(0,1)` (without the `/√2` normalisation of the `gaussian` branch).
    - **colored** (line 105–115): start from `gaussian`, then multiply each `(l,m)` by `(l+1)^(-α)` with default `color_alpha = 1.0`.
  - Three composable post-processing knobs (lines 119–143):
    - `mode_dropout_prob` (default `0.0`): independent Bernoulli mask over both families.
    - `family_balance ∈ [0,1]` (default `0.5`): scales `a^E ← (1−b)·a^E·2`, `a^M ← b·a^M·2` so `b=0.5` gives equal weight; optional jitter via `family_balance_jitter`.
    - `coef_scale_log_uniform_range`: optional `(lo, hi)` → per-sample multiplicative scale `exp(U(lo, hi))`.
  - Final cast to `complex64` (line 144); pack via `pack_coefficients` → `float32` real vector of length `4K` (line 174).
- **Use in chapter**: §2.1 cites the four modes, the three knobs, and the einsum synthesis chain. The framework cap `L_MAX=15` is mentioned only to note that the thesis fixes the working truncation at `L=5` via Chapter 1.

---

## R2 — Augmentation primitives

- **Sources consulted (read-only)**:
  - `src/mpinv/data/augment.py` lines 1–295 (the module's own narrative docstring is unusually rich and explicitly states the physical-consistency contract of each augmentation).
- **Verbatim facts extracted** — five primitives:
  1. **`coef_phase_rotation`** (lines 49–121): `(a^E, a^M) → (e^{iα} a^E, e^{iα} a^M)` with `α ∼ U[0, 2π)`. `P = |E|² is invariant`, so `P` is left unchanged; only `packed` updates. The docstring explicitly cites `presentation/ch1_full.md §1.6` for the global U(1) invariance.
  2. **`coef_additive_noise`** (lines 57–155): `a ← a + σ_a · (N(0,1) + iN(0,1))/√2`, default `sigma = 0.05`; re-synthesises `P` from the perturbed coefficients to keep `(P, packed)` consistent.
  3. **`coef_mode_dropout`** (lines 69–185): independent Bernoulli keep mask per `(l, m)` per family per sample with `dropout_prob` (default `0.1`); re-synthesises `P`.
  4. **`field_additive_noise`** (lines 84–203): `σ_i = relative_sigma · max(P_i)` (default `relative_sigma = 0.02`), `P' = max(P + σ_i · N(0,1), 0)`, targets unchanged. This is the explicit **noisy-input / clean-target** robustness contract.
  5. **`field_phi_roll`** (lines 96–236): random integer roll `k ∼ U{0, …, n_phi−1}` along the azimuth axis; `P' = np.roll(P, k, axis=-1)`; `a'_{l,m} = e^{-imφ_k} a_{l,m}` with `φ_k = 2π k / n_phi`. The 1° azimuth grid samples `φ_k` exactly, so `(P, packed)` stay consistent up to discretisation error.
- **External-citation note** — the `field_additive_noise` docstring (line 21) attributes the noisy-input/clean-target setup to "Schmid et al. 2025" but does not give a full citation. A targeted web search did not uniquely identify the paper. The chapter therefore describes the noisy-input/clean-target setup as standard practice in robust inverse-problem learning without attempting a specific Schmid attribution; a directly relevant external reference is Lerma Pineda and Petersen, "Deep neural networks can stably solve high-dimensional, noisy, non-linear inverse problems", arXiv:2206.00934, 2022, which formalises "training on randomly perturbed data" as a stability mechanism for learned inverses (Theorem 4.3 of that preprint).
- **Use in chapter**: §2.1's augmentation paragraph lists all five primitives with exact maths and rationales; §2.2's pipeline names the composed chain `field_phi_roll → coef_mode_dropout → field_additive_noise`.

---

## R3 — Real-antenna data, file format, and L=5 decomposition

- **Sources consulted (read-only)**:
  - `src/mpinv/data/real_antenna_loader.py` lines 1–164.
  - `src/mpinv/data/basis_decomposer.py` lines 1–116.
  - `scripts/import_legacy_real_antenna.py` lines 1–256.
  - `data/raw/real_antenna/README.md` lines 1–18.
- **Verbatim facts extracted**:
  - File format (lines 5–11 of `real_antenna_loader.py`): seven columns per grid sample, `θ_deg, φ_deg, P, |E_θ|, arg(E_θ)_deg, |E_φ|, arg(E_φ)_deg`. Phases in **degrees in the file**, converted to **radians once in the loader** (lines 80–82).
  - Layout: rows stored outer-φ / inner-θ; loader reshapes `(n_phi, n_theta, 7)` and transposes to canonical `(n_theta, n_phi)` (lines 75–85). Loader verifies `n_phi · n_theta = 64 440` rows (lines 70–74).
  - Complex field: `E_θ = |E_θ| e^{i arg E_θ}`, `E_φ = |E_φ| e^{i arg E_φ}`; `E = stack((E_θ, E_φ), axis=0)` has shape `(2, n_theta, n_phi)` complex64.
  - Coefficient file (paired `Results_<id>.txt`): rows `Type l m Re Im` with `Type ∈ {E, M}`, packed via `pack_coefficients` of Chapter 1 §1.1 into a real vector of length `4K = 140` at `L = 5`.
  - Analytic VSH decomposition (lines 21–26 of `basis_decomposer.py`): area-weighted inner product on the same `(n_theta, n_phi)` grid as Chapter 1 §1.3; einsum `("kfctp,nctp,t->nkf", conj(basis), stack(E_θ, E_φ), w)` with `w = sin θ · dθ · dφ`. This is the discrete realisation of the Chapter 1 §1.3 projection.
  - Corpus size and residuals (`data/raw/real_antenna/README.md`):
    - Total paired files: **396** (line 7).
    - Bandlimit on import: **L = 5** (line 8).
    - Projection residuals at L=5 on the corpus (lines 15–16):
      - `||P_meas − P_resyn||_2 / ||P_meas||_2`: mean **0.0116**, median **0.0036**, max **0.0808**.
      - `||E − E_resyn||_2 / ||E||_2`: mean **0.0130**, median **0.0032**, max **0.1005**.
    - The 7-column files written by the importer store `P_resyn = |E_resyn|^2` from the bandlimited reconstruction (line 9 of README + `import_legacy_real_antenna.py` lines 174, 254).
- **Use in chapter**: §2.2 cites the file format, the analytic-decomposition path, the corpus count, and the L=5 residual numbers. The pipeline narrative summarises the load-and-truncate step without naming the experiment folder, per the user's instruction to generalise.

---

## R3b — Real-augmented split logic and training-pool augmentation

- **Sources consulted (read-only)**:
  - `scripts/run_real_augmented.py` lines 122–252 (CLI defaults) and lines 335–400 (the `_build_augmented` chain) and lines 507–538 (`_peek_split_ids` split logic).
- **Verbatim facts extracted**:
  - Split by sample ID, not by augmented copy:
    - Shuffle the full paired-file list with deterministic `shuffle_seed = 42` (line 132).
    - Take the first `n_source` IDs as the train+val pool (default `n_source = 100`); the first `n_train_sources` (default `80`) become train sources, the remainder become val (line 528–530).
    - Re-shuffle the tail with an independent `holdout_shuffle_seed = 314159` and take the first `n_holdout_samples` (default `100`) as the held-out test set (lines 533–537).
  - Augmentation **only on the training-source pool**, with `aug_seed = 4242` (line 134). Producing `n_augmented` samples (the canonical "best" pipeline uses 10 000 in production runs and 1 000 in lighter ablations) by:
    1. Source resampling with replacement: `i ∼ U{0, …, n_train_src − 1}`.
    2. `field_phi_roll` on the full batch.
    3. `coef_mode_dropout(p)` in chunks (default `aug_chunk_size = 500`).
    4. `field_additive_noise(relative_sigma)` on the full batch.
  - Validation and holdout receive no augmentation (lines 833–845).
  - Optional global scale: `P ← s · P`, `packed ← √s · packed` consistent with `P ∝ |E|²` (lines 143–152, 865–886). Production runs use `s = 10⁶` to lift O(10⁻⁶) magnitudes into a numerically convenient range.
- **Use in chapter**: §2.2 describes the split-by-ID strategy, the composed augmentation chain, and the leakage protections. No experiment folder is named in the prose, per the header instruction.

---

## R4 — Feature pipelines and normalisers

- **Sources consulted (read-only)**:
  - `src/mpinv/features/modes.py` lines 10–63 (input modes: power, magnitude, complex).
  - `src/mpinv/features/raw_flat.py` lines 1–82 (flat baseline, optional `log_input`, `StandardScaler`).
  - `src/mpinv/features/pca.py` lines 41–91 (`sklearn.decomposition.PCA(svd_solver="randomized")` with default `n_components=128`, `random_state=0`, `whiten=False`; and `IncrementalPCA` for streaming).
  - `src/mpinv/features/fft_radial.py` lines 44–94 (2D FFT over `(θ, φ)`, optional `sin θ` window, amplitude, radial binning, `log1p`).
  - `src/mpinv/features/sh_power.py` lines 49–91 (scalar SH projection of the `P` channel, sum of `|⟨P, Y_l^m⟩|²` per degree `l`, optional `log1p`).
  - `src/mpinv/features/composite.py` lines 47–84 (PCA + extractor concatenation, optional global `StandardScaler` post-concat).
  - `src/mpinv/features/power_pipeline.py` lines 32–99 (canonical PCA wrapper used through Hydra).
  - `src/mpinv/features/normalisers.py` lines 24–55 (`StandardScaler` with `eps = 1e-8`, `PassthroughScaler`).
  - `src/mpinv/features/registry.py` lines 7–18 (`FEATURE_EXTRACTORS` map).
  - `configs/features/*.yaml` (all six YAML files) for canonical defaults.
- **Verbatim facts extracted** (selected; others quoted inline in the chapter):
  - **Modes**: `InputMode.POWER` (1 channel, `(B, 1, n_θ, n_φ)`), `MAGNITUDE` (2 channels, `(|E_θ|, |E_φ|)`), `COMPLEX` (4 channels, `(Re E_θ, Im E_θ, Re E_φ, Im E_φ)`). POWER tolerates missing complex field; MAGNITUDE and COMPLEX raise `ValueError` if `E` is not provided.
  - **Randomised PCA**: defaults `n_components=128`, `whiten=False`, `random_state=0`; centering performed inside sklearn `PCA.fit`. Input width depends on input mode: `64 440` (POWER), `128 880` (MAGNITUDE), `257 760` (COMPLEX). Fit on the training partition only; transform-only on val/holdout.
  - **FFT radial spectrum**: optional `sin θ` pre-window (default on); 2D `fft2` over `(θ, φ)`; amplitude `|F|` (not power); equal-radius bins from `fftshift(fftfreq(·))`; default `n_bins = 32`; optional `log1p`. Output width `n_bins × C`.
  - **SH spectral power**: scalar SH basis `Y_l^m` constructed on `grid.theta_axis() × grid.phi_axis()`; area-weighted inner product; per-degree sum `out[:, c, l−1] = ∑_m |⟨ch_c, Y_l^m⟩|²`; optional `log1p`. Output width `L × C`.
  - **Normalisers**: `StandardScaler` only, fit on the training partition. The configured option `normalise_targets: true` is **not applied** by the implementation — packed coefficients reach the model unscaled. This is recorded as an honest gap in §2.6.
- **Use in chapter**: §§2.3–2.6 cite these files directly. The `pca_cv` and `raw_plus_sh` composite YAMLs are mentioned in §2.5 as the canonical CV-feature stacks (header §2.5).

---

## R5 — Randomised SVD theory (Halko, Martinsson, Tropp 2011)

- **Citation**: N. Halko, P.-G. Martinsson, J. A. Tropp, "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions", *SIAM Review*, vol. 53, no. 2, pp. 217–288, 2011. doi:`10.1137/090771806`. arXiv:`0909.4061` (preprint version).
- **Sources consulted**:
  - Author preprint at <https://tropp.caltech.edu/papers/HMT11-Finding-Structure.pdf>.
  - SIAM Review landing page <https://epubs.siam.org/doi/abs/10.1137/090771806>.
  - arXiv <https://arxiv.org/abs/0909.4061>.
- **Verbatim facts extracted** (from the SIAM Review abstract / introduction):
  - The paper presents a modular framework for randomised partial matrix decompositions. The randomised algorithms use random sampling to identify a subspace that captures most of the action of the input matrix, then deterministically reduce the matrix on that subspace.
  - For a dense `m × n` input matrix and target rank `k`, the randomised algorithm needs `O(mn log k)` flops, versus `O(mnk)` for classical algorithms targeting the same rank — i.e. **a factor `k / log k` speedup** in the model problem.
  - For matrices too large to fit in memory, the randomised algorithm needs only a constant number of passes over the data, versus `O(k)` passes for classical algorithms (single-pass variants exist).
  - Error analysis is provided in the paper but is not used in the chapter; the chapter cites only the complexity argument.
- **Use in chapter**: §2.4 cites Halko–Martinsson–Tropp 2011 for the rationale of using `sklearn.decomposition.PCA(svd_solver="randomized")` instead of a deterministic full SVD on the `(N, 64 440)` POWER feature matrix.

---

## R6 — scikit-learn implementation reference

- **Citation**: F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, É. Duchesnay, "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
- **Sources consulted**: scikit-learn user guide for `sklearn.decomposition.PCA` and `sklearn.preprocessing.StandardScaler` (current stable docs); JMLR PDF for the citation metadata.
- **Verbatim facts extracted**:
  - `sklearn.decomposition.PCA(n_components, svd_solver="randomized", random_state)` exposes randomised SVD as a drop-in solver; under the hood it calls the randomised algorithm of Halko–Martinsson–Tropp 2011 (R5).
  - `sklearn.preprocessing.StandardScaler` performs per-feature `(x − μ) / σ` standardisation with `with_mean=True, with_std=True` by default. The project wraps this in a `Normaliser` protocol with a small `eps = 1e-8` added to `σ` for numerical stability (`src/mpinv/features/normalisers.py` lines 24–36).
- **Use in chapter**: §2.4 names sklearn as the implementation; §2.6 quotes the StandardScaler formula. No separate citation is needed inline in the prose; the JMLR reference appears in the chapter-end references block.

---

## R7 — Project tensor-layout contract

- **Sources consulted**:
  - `docs/tensor_shapes.md` lines 1–60 (binding contract).
  - `src/mpinv/core/grid.py` lines 39–100 (default grid).
  - `src/mpinv/core/area_weights.py` lines 15–45 (area weights).
  - `src/mpinv/core/shapes.py` lines 21–68 (assertion helpers).
- **Verbatim facts extracted**:
  - `P`: `(B, n_theta, n_phi) = (B, 179, 360)` `float32`.
  - `E`: `(B, 2, n_theta, n_phi)` `complex64`; channels `(E_θ, E_φ)`.
  - VSH basis: `(K, 2, 2, n_theta, n_phi)` complex.
  - Area weight `μ(θ) = sin θ · dθ · dφ`, shape `(n_theta,)`; optional mean-normalisation broadcasts to `(n_theta, n_phi)` with global mean 1 for loss-scale stability.
  - The only layout boundary in the codebase is `src/mpinv/data/real_antenna_loader.py` lines 75–77, which transposes the seven-column file's `(n_phi, n_theta, 7)` row order into the canonical `(n_theta, n_phi)`. There is no `to_torch_layout` helper.
- **Note on `AGENTS.md` drift**: the repo-root `AGENTS.md` describes a different layout (`(B, n_phi, n_theta)` for the numpy einsum side, with a `to_torch_layout` boundary). The implemented contract is the one in `docs/tensor_shapes.md`, which is what the chapter follows. The discrepancy is documentation drift, not a bug; it is noted here so that the chapter does not propagate the wrong layout statement.
- **Use in chapter**: §§2.2–2.5 quote the `(B, n_theta, n_phi) = (B, 179, 360)` shape verbatim.

---

## Hard-rule check (RESEARCHER.md "Hard rule on empty research")

Phase 3 produced new information for R1, R2, R3, R3b, R4, R5, R6, R7. None of the entries are empty. The Schmid 2025 reference cited in `augment.py` (R2) could not be uniquely identified through web search; the chapter therefore describes the noisy-input/clean-target setup as standard practice without an attributed citation, and the gap is recorded explicitly above.

## Sources consulted but not used

- arXiv 2503.19468v1 (Noisier2Inverse, 2025) — self-supervised correlated-noise reconstruction; topic-adjacent but not what the chapter cites.
- arXiv 2510.12521 (learned regularisers with unknown noise covariance, 2025) — adjacent; not cited.
- arXiv 2408.08119 (joint parameterisation for inverse problems) — adjacent; not cited.
- arXiv:2206.00934 (Lerma Pineda and Petersen, "Deep neural networks can stably solve high-dimensional, noisy, non-linear inverse problems", 2022) — formalises noisy-input training as a stability mechanism; cited inline in §2.1 Primitive 4 for the Theorem 4.3 stability argument that underwrites the noisy-input/clean-target rationale.

---

## Post-review revision + aggressive trim (2026-06-01)

After the initial draft, Chapter 2 was reviewed by `presentation/ch2_review.md` (Judge Volodya, MAJOR REVISION verdict) and a second pass was applied to (i) fix the eight defect categories Volodya raised and (ii) aggressively trim non-operational prose so that the chapter reads as a methods section, not a textbook walkthrough.

**Trim philosophy.** The target reader wants to know *what data was used, how it was featurised, what normalisation was applied*. Everything that does not answer that question was cut: §§1.4–1.6 rehashes ("information collapse", "operational measurement regime"), justification-of-justification (per-knob Jackson asides, maxent interpretations of `gaussian`, philosophical "strict-symmetry vs perturbation" recaps), forward throat-clearing into Chapter 7, and explanatory paragraphs following self-explanatory diagrams. Equations, parameter defaults, and pipeline order were preserved without exception.

**Volodya defect categories addressed (eight):**

1. *Self-containedness gaps* — Unbound counts $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}, N_\mathrm{aug}$ in §2.2 are now explicitly deferred to "the experimental tables of Chapter 7" with a one-sentence pointer; the chapter-opener "fully specified" promise is softened to "pipelines are fully specified" with the same Chapter 7 deferral.
2. *Code-state references* — The snake_case augmentation chain `field_phi_roll → coef_mode_dropout → field_additive_noise` in §2.2 is replaced by "Primitive 5 (azimuthal roll) → Primitive 3 (coefficient mode dropout) → Primitive 4 (field additive noise)", referencing the §2.1 numbering. The unverified target-normalisation "flag" framing in §2.6 is replaced by a procedural statement of what is actually applied.
3. *Notation collisions* — `r_P, r_E` (§2.2 residuals) → $\rho_P, \rho_E$; `X` (§2.4 data matrix) → $\boldsymbol\Phi$ (resolves collision with $a^X_{lm}$ and §2.5.1 $\hat X$); `p` (§2.4 oversampling) → $q$ (resolves collision with §2.1 Bernoulli probability); `x` (§2.5.1 channel) → $f$ and `\hat X` → $\hat f$; `Y_l^m` (§2.5.2) → $Y_{lm}$ aligned with Chapter 1. A one-line $\hat\mu$ vs $\mu(\theta)$ disclaimer is added at the first appearance of $\hat\mu$ in §2.6.
4. *Strictness overreach* — "no $L=5$-truncated model can ever recover" (§2.2) weakened to "can reproduce above-bandlimit angular content in the resynthesised field"; the Lipschitz claim on §2.1 Primitive 4 is replaced with a motivation-only citation of Lerma Pineda & Petersen 2022; the "well-known pole-region bias" (§2.5.1) is replaced by a self-contained $\sin\theta$ derivation tying the FFT pre-window to the §1.3 area weight; the "informative directions" plausibility defect (§2.4) and the "useful when the downstream regressor is itself approximately linear" speculation (§2.5.1) are resolved by deletion.
5. *Unjustified literature attributions* — The Jackson Ch. 9 $(ka)^l$ aside in §2.1 Mode 3 is deleted, which also resolves the `a`-collision with the packed coefficient $a$. The remaining Jackson reference is then unused, so Jackson 1999 is removed from the chapter-end references block entirely.
6. *Mechanical errors* — "azimuthally invariant up to the sign of the radial bin" (§2.5.1) corrected to a precise statement that integer azimuthal rolls leave the radial-bin amplitudes unchanged because each Fourier component picks up a unit-modulus phase under a roll.
7. *Composite-stack naming* — The named stacks "PCA + CV" and "raw + SH" introduced in §2.5 are now carried through §2.6 so that the scaler discussion references concrete pipelines instead of "composite" in the abstract.
8. *Tensor-shape bijection drift* — The `(K, 2, 2, n_θ, n_φ)` basis-tensor mention is dropped as part of the §2.1 "Forward sampling chain" compression, removing the implicit bijection statement Volodya flagged.

**Operational content preserved.** All equations of the original draft remain (modes, knobs, augmentation primitives, decomposition quadrature, residual definitions, randomised-SVD construction, FFT and SH descriptors, StandardScaler formula, preprocessing diagram). All numerical defaults remain ($\sigma_a = 0.05$, $p = 0.1$, $\sigma_P = 0.02$, $\alpha = 1$, $p_\mathrm{active} = 0.1$, $r = 128$ canonical / $r = 64$ composite, $n_b = 32$ canonical / $n_b = 16$ composite, $\varepsilon = 10^{-8}$). All numbers for the real corpus remain ($\rho_P$ mean 1.16 %, median 0.36 %, max 8.08 %; $\rho_E$ mean 1.30 %, median 0.32 %, max 10.05 %). All composite-stack widths remain ($85$ for *PCA + CV* on POWER, $64{,}445$ for *raw + SH* on POWER).

---

## Second review pass (2026-06-01)

After the first revision, Chapter 2 was reviewed again by `presentation/ch2_review_1.md` (Judge Volodya, second MAJOR REVISION verdict). The second pass addresses the new defect categories listed below.

**Sourced facts added (six):**

1. *Residual computation method (§2.2, R3)* — The relative-residual numbers ($\rho_P$ and $\rho_E$ percentiles) are now explicitly sourced: full corpus of $396$ paired files at $L = 5$, on the same uniform 1° angular grid; $\|\cdot\|_2$ is the flat $\ell^2$ norm over the angular grid (no area weight); $P_\mathrm{meas}$ is the raw $|E_\theta|^2 + |E_\varphi|^2$ from the seven-column file and $P_\mathrm{resyn} = |E_\mathrm{resyn}|^2$ is the squared modulus of the bandlimited resynthesis. Backed by the `import_legacy_real_antenna` code path (`scripts/import_legacy_real_antenna.py` lines 230–246) and by `data/raw/real_antenna/README.md`.
2. *$\sin\theta$ pre-window default-on (§2.5.1, R4)* — Verified in `src/mpinv/features/fft_radial.py` line 27 (`sin_theta_window: bool = True`) and in `configs/features/pca_cv.yaml`, `configs/features/cv_only.yaml`. The chapter now says "default on; both the `pca_cv` and `cv_only` pipelines apply it".
3. *$\varphi$-grid endpoint convention (§2.1 Primitive 5, R7)* — Verified in `src/mpinv/core/grid.py` line 72: `np.linspace(0.0, 2*pi, n_phi, endpoint=False)`, so $\varphi_j = j\cdot 2\pi/n_\phi$ exactly. This makes the integer-roll consistency claim rigorous.
4. *Mode 4 missing $1/\sqrt 2$ (§2.1, R1)* — Verified in `src/mpinv/data/synthetic_generator.py` lines 101–104: Mode 4 sparse does *not* divide by $\sqrt 2$, unlike Modes 1 and 3. This is a real convention break in the code, now flagged in the chapter with $\mathbb E|a^X_{lm}|^2 = 2$ on active modes and $2\,p_\mathrm{active}$ unconditionally.
5. *Primitive 2 complex-Gaussian convention (§2.1, R2)* — Verified in `src/mpinv/data/augment.py` lines 146–151: $\varepsilon = (\varepsilon_R + i\varepsilon_I)/\sqrt 2$, $\mathbb E|\varepsilon|^2 = 1$. Same convention as Mode 1, now stated explicitly at first use.
6. *Augmentation-chain omission rationale (§2.2, R3b)* — Primitives 1 (global phase) and 2 (coefficient additive noise) are omitted from the real-data chain because (a) the bandlimited $P_\mathrm{resyn}$ is invariant under Primitive 1 and the corresponding $a$-rotation contributes no operationally new pair under coefficient-MSE training, and (b) the projection residuals of §2.2 already inject a measurement-side coefficient perturbation on the real corpus, so Primitive 2 would compound an effect already present in the data.

**Volodya defect categories addressed in this second pass (five):**

1. *Self-containedness gaps* — Residual sourcing (above); $\sin\theta$ window default; $\varphi$-grid endpoint convention; Primitive 2 $\varepsilon$ convention; post-hoc mode dropout vs Mode 4 motivation; Mode 4 motivation sentence added.
2. *Notation collisions* — Primitive 3's parameter renamed from $p$ to $p_\mathrm{drop}$ to share the post-hoc-knob symbol, with $p_\mathrm{drop} = 1 - p_\mathrm{active}$ relation noted; $n_\mathrm{ang}$ (§2.4) disambiguated from $N_\mathrm{ang}$ (§1.3) at first use; FFT formula rewritten with integer indices $(i, j)$ for clarity; array-shape convention $(2, 179, 360)$ in §2.2 cross-referenced to §1.2's $(360, 179, 2)$; `complex64` glossed as "single-precision complex floating-point (numpy `complex64`)".
3. *Coherence and depth — missing motivations* — Modes 1–3 motivations expanded one sentence each to name the prior they encode (order-agnostic / low-order-favouring); Mode 4 gets its own motivation (sparse-support prior) plus the normalisation-disparity flag; family-balance factor-of-2 motivated as the identity-preserving normalisation choice; post-hoc mode dropout knob motivated as the post-sampling counterpart of Mode 4 usable with any underlying mode.
4. *Strictness gaps* — Primitive 5 reframed as an exact $\mathrm{SO}(2)$ *equivariance* of $\mathcal A$ (not an extension of §1.7's ambiguity catalogue), with explicit vector-basis lift $\boldsymbol\Psi^X_{lm}(\theta, \varphi - \varphi_k) = e^{-im\varphi_k}\boldsymbol\Psi^X_{lm}(\theta, \varphi)$; the azimuthal-invariance argument in §2.5.1 strengthened to the pointwise-magnitude statement on $\hat f$ (Fourier components transform by a unit-modulus phase under a roll); the §2.2 "strict-symmetry first" rationale rewritten as a commute-up-to-grid statement (Primitive 5 commutes with Primitive 3 because the phase is per-mode and the mask is index-wise); the §2.3 "diagnostic upper bound" wording aligned with Chapter 1's "analytic gold-standard".
5. *Section structure* — The `### Composite feature stacks` heading in §2.5 (not in `header.md`) demoted to an unmarked tail paragraph framed as a *composition* of the previous features, with the labels *PCA + CV* and *raw + SH* explicitly flagged as this thesis's local shorthands reused in Chapter 7.
