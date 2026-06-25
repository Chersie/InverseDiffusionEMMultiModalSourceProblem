I will scope myself to Chapter 2 only (the block in `presentation/paper_full.md` from `# Chapter 2.` through `<!-- Sources backed by research/ch2-data-and-features/manifest.md (R1–R7). -->`), using `presentation/header.md` as the TOC reference and `presentation/ch1_full.md` only where Chapter 2 *explicitly* back-references it.

---

# Judge Volodya — Report on Chapter 2

## Coherence

### Per-section heading-vs-body table

| Section heading (English draft) | Outline match in `header.md` | Body delivers heading | Status |
|---|---|---|---|
| 2.1 Synthetic data generator | "2.1. Синтетический генератор данных" | covers forward sampling chain, four coefficient distributions, composable knobs, five augmentation primitives | PASS |
| 2.2 Real data used in training and validation | "2.2. Реальные данные, применямые в обучении и валидации" | covers file format, bandlimit, split, augmentation chain | PASS |
| 2.3 Input modes: power, ma и их информационное содержание" | defines the three modes; discusses information content vs. §§1.4–1.6 | PASS |
| 2.4 Randomised PCA as a dimensionality-reduction stage | "2.4. Рандомизированный PCA как метод снижения размерности: входная форма (N, 64440) и её сжатие" | discusses dimensionality, randomised SVD, fit-on-train, higher-dim modes | PASS |
| 2.5 Additional computer-vision features | "2.5. Дополнительные CV-признаки" | introduces structured descriptors, defines two of them | PASS |
| 2.5.1 Radial Fourier power spectrum | "2.5.1. Радиальный спектр мощности FFT" | windowed FFT, radial annular binning, log post-transform | PASS |
| 2.5.2 Spherical-harmonic spectral power | "2.5.2. Спектральная мощность сферических гармоник" | per-channel scalar-SH projection, per-degree energy | PASS |
| 2.6 Normalisation of features and target variables | "2.6. Нормализация признаков и целевых переменных" | feature StandardScaler; target normalisation reported as "configured, not applied" | PASS |

### Section ordering vs. `header.md`

Status: PASS. The English draft preserves the 2.1 → 2.2 → 2.3 → 2.4 → 2.5 (with 2.5.1, 2.5.2) → 2.6 order verbatim.

---

## Self-containedness

### Unresolved assumptions

| Quoted span | Classification |
|---|---|
| "the basis matrix of §1.3 is precomputed once as a complex tensor of shape $(K, 2, 2, n_\theta, n_\phi)$ (one slab per mode, per family, per tangential component)" (§2.1) | `argument needed` — §1.3 defined the basis as $\mathbf B \in \mathbb C^{70\times 128880}$; the chapter introduces a *different* layout in §2.1 and never states that the new five-axis tensor is a reshape of the same object. The reader must guess the bijection between the two shapes. |
| "Across the full real corpus the residuals are small but not negligible: $r_P$ has meat, or with what conventions. The reader has no way to relate the numbers back to any defined operator in the chapter. |
| "the first $n_\mathrm{src}$ identifiers are taken as the train-and-validation pool; the first $n_\mathrm{tr}$ of those identifiers form the training set ... and its first $n_\mathrm{ho}$ identifiers form the held-out evaluation set." (§2.2) | `definition needed` — $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}$ are introduced symbolically but never bound to numbers, nor explicitly deferred to a later chapter. Contradicts the chapter's own opening claim that "By the end of §2.2 the training distribution itself is fully specified". |
| "the training pool is expanded from $n_\mathrm{tr}$ source antennas to $N_\mathrm{aug}$ augmented training samples" (§2.2) | `definition needed` — $N_\mathrm{aug}$ is named once and never quantified or deferred. |
| "**field\_phi\_roll** → **coef\_mode\_dropout**(p) → **field\_additive\_noise**($\sigma_P$)." (§2.2) | `definition needed` — three ntifiers are introduced without an explicit mapping to the numbered Primitives 1–5 of §2.1. The body afterwards informally hints ("Azimuthal roll is a strict symmetry of $\mathcal A$ and applied first") but never states "field_phi_roll = Primitive 5" etc. |
| "an optional spherical-area window is applied to suppress the well-known pole-region bias of a flat 2-D FFT on an angular grid" (§2.5.1) | `external citation needed` — "well-known" is asserted without a reference, and "pole-region bias" is not defined or derived. |
| "useful when the downstream regressor is itself approximately linear in $P$ near the operating point" (§2.5.1) | `argument needed` — claims a property of "the downstream regressor" which has not yet been introduced anywhere in the in-scope text, and provides no support. |
| "The framework configures a target-side normalisation flag alongside the feature-side flag ... The implementation, however, does *not* act on that flag" (§2.6) | `external citation needed` — a claim about ththe implementation" (i.e. the code), which the reader does not have access to. The operational consequence ("the packed coefficient vector reaches the loss layer unscaled") is what the chapter actually needs; the appeal to a flag in the codebase is unverifiable. |
| "in qualitative agreement with the Jackson Ch. 9 convergence argument that higher-$l$ contributions decay with $(ka)^l$ for a source of finite extent $a$" (§2.1, Mode 3) | `external citation needed` — the chapter cites "Jackson Ch. 9" generically without page/section, and the symbol $a$ here ("finite extent $a$") collides with $a$ used everywhere else for the packed coefficient vector (see Notation below). |

### Forward references present in the chapter

| Forward reference | Status |
|---|---|
| "the empirical question of which prior trains the best learned regulariser is settled in Chapter 7" (§2.1) | correct deferral |
| "$\sigma_a = 0.05$, $p = 0.1$, $\sigma_P = 0.02$ ... are revisited in the experimental chapters where their final operg values are determined empirically" (§2.1) | correct deferral |
| "the other two modes appear in the experimental chapters as diagnostics" (§2.3) | correct deferral |
| "The choice beeen the two — and the choice of whether to compose with PCA at all — is an empirical question settled in Chapter 7." (§2.5) | correct deferral |
| "The contrast between the two losses on this point is one of the differences that the coefficient-vs-physics comparison of Chapter 5 is designed to expose2.6) | correct deferral |
| "the architecture chapters that follow" / "the learned regulariser of Chapter 4" (§2.6) | correct deferral |

No deferral was found that masquerades as a deferral while in fact dropping a load-bearing argument.

---

## Terminology and notation

### Non-standard or undefined terms
| Quoted span | Issue |
|---|---|
| "**field\_phi\_roll** → **coef\_mode\_dropout**(p) → **field\_additive\_noise**($\sigma_P$)" (§2.2) | snake_case identifiers used as terms-of-art without definition; the mapping to Primitives 1, 3, 4, 5 of §2.1 is left to the reader to reconstruct. |
| "the *PCA + Cck" / "the *raw + SH* stack" (§2.5) | composite-pipeline names are locally introduced but never recalled with the same naming convention in §2.6, which talks only about a generic "composite pipeline". |
| "azimuthally invariant up to the sign othe radial bin" (§2.5.1) | "sign of the radial bin" is not a defined concept — radial bins are indexed by non-negative intrs, so "sign" has no obvious referent. |
| "pole-region bias of a flat 2-D FFT on an angular grid" (§2.5.1) | term used in a load-bearing way (it motivates the $\sin\theta$ window) but neither defined nor cited. |
| "with conservative defaults" (§2.4) | unquantified qualifier about scikit-learn's randomised SVDnothing in the chapter pins what "conservative" means here. |

### Notation collisions and ambiguities

| Symbol | First use | Colliding use | Issue |
|---|---|---|---|
| $p$ | "with default active-mode fraction $p_\mathrm{active} = 0.1$" (§2.1, Mode 4); "For $p \in [0, 1]$" (§2.1, Primitive 3) | "a small oversampling $p$ (typically $p \in [5, 10]$)" (§2.4) | same letter  used for a Bernoulli probability in §2.1 and for an integer oversampling parameter in §2.4, with overlapping symbolic neighbourhoods. |
| $r$ | "the principal-component basis of rank $r$(§2.4) | "$r_P$ has mean $1.16\%$ ... $r_E$ has mean $1.30\%$" (§2.2) | $r$ is the PCA rank in §2.4 and a relative-residual fraction in §2.2; the latter uses a subscript but is the samtter. |
| $X$ | "$a^X_{lm}$" with $X \in \{E, M\}$ (§§2.1–2.2, 2.5.2) | "Let $X \in \mathbb{R}^{N \times n_\mathrm{ang}}$ stack the flat input vectors" (§2.4); "$\hat X(k_\theta, k_\varphi)$" the Fourier transform (§2.5.1) | the letter $X$ does at least three different jobs across the chapter (mode-family idata matrix, Fourier transform of one channel). No identification or warning. |
| $a$ | "a packed coefficient vector $a \in \mathbb R^{4K}$" (§2.1, §2.6, everywhere) | "for a source of finite extent $a$" (§2.1, Mode 3, in the parenthetical $(ka)^l$ remark) | the symbol $a$ is overloaded with "antenna radius"side a justification, with no explicit warning. |
| $Y_l^m$ vs. $Y_{lm}$ | §1.1 uses $Y_{lm}$ throughout | §2.5.2 uses $Y_l^m$ | the same scalar harmonics are written two different wayacross chapters with no statement of identification. |
| $\mathbf B$ vs. "complex tensor of shape $(K, 2, 2, n_\theta, n_\phi)$" | §1.3 defines $\mathbf B \in \mathbb C^{70\times 128880}$  §2.1 refers to "the basis matrix of §1.3 ... precomputed once as a complex tensor of shape $(K, 2, 2, n_\theta, n_\phi)$" | two different array shapes for the same object; the bijection (which is a reshape of $\mathbf B$) is not stated. |
| $\hat\mu$ vs. $\mu$ | $\mu(\theta) = \sin\theta\,\Delta\theta\,\Deltvarphi$ (area weight, §1.3, §2.2, §2.5, §2.6) | $\hat\mu \in \mathbb R^{d_\mathrm{feat}}$ (feature mean, §2.6) | the hat distinguishes them but they sit in the same paragraph in §2.6, where the area weight is also being invoked; the visual collision is real and not flagged. |

---

## Depth

**Chapter roleferred from `header.md`:** `implementation / experiment`. The outline puts §2.1, §2.2, §2.4, §2.5, §2.6 squarely at the construction level ("Способы генерации коэфанный PCA как метод снижения размерности"), and §2.2 in the outline even gestures at a *specific* experimental artefact ("нужно посмотреть в be reproducible.

**Depth assessment:** `too shallow` in two specific places, otherwise `matches role`.

| Quoted span | Defect | Recommendation |
|---|---|---|
| "By the end of §2.2 th training distribution itself is fully specified" (chapter opener) vs. §2.2 introducing $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}, N_\mathrm{aug}$ without binding any of them | the opening promises reproducibility; the body does not deliver four of the integers that govern that reproducibility, nor does it deer them with an explicit forward pointer. | `expand here` — bind the four counts (or, if they vary across experiments, defer to a named later table with a one-line pointer). |
| "By the end of §2.6 the input vector and the target vector reaching the model are likewise fully specified." (chapter opener) vs. §2.aving rank $r$, bin count $n_b$, and the choice among feature stacks as configurable | the dimensionality of the input vector $\tilde z$ depends on choices that are explicitly left open ("the choice ... is an empirical question settled in Chapter 7"); the chapter cannot simultaneously claim both. | `expand here` — soften the opener so that "fully specified" refers to th*pipeline*, not to a single instantiated input width, and clearly mark the configurable knobs that remain. |
| "The framework configures a target-side normalisation flag ... The implementation, however, does *not* act on that flag" (§2.6) | the chapter is at conceptual depth elsewhere; here it dives into the state of a configuration flag in the codebase, a level of detail the reader cannot check and that does not belong at the formulation layer. | `expand here` — restate the *procedural fact* ("the packed coefficient vector reaches the loss layer unscaled") and drop the "framework configures a flag" framing, OR add the licit reference to the configuration file (and consequence table) that pins it down. |

The remainder of the chapter (in particular §§2.1 and 2.4) is at appropriate depth for the role.

-

## Strictness gaps

| Quoted span | Status |
|---|---|
| "the well-known pole-region bias of a flat 2-D FFT on an angular grid" (§2.5.1) | plausible but unjustified — no citation, norivation, and "well-known" is doing the work of an argument. |
| "useful when the downstream regressor is itself approximately linear in $P$ near the operating point" (§2.5.1) | plausible but unjustified — asserts a property of a regressor that the chapter does not define, and never reconciles with the fact thahe same descriptor is used for nonlinear regressors. |
| "it Lipschitz-constrains the learned inverse on the small-noise sphere around each anchor, in the same spirit as the noisy-data training that Lerma Pineda and Petersen (arXiv:2206.00934, 2022, Theorem 4.3) prove stabilises learned inverses for noisy forward operators." (§2.1, Primitive 4) | overreach — the cited theorem applies to a specific class of inverse problems and architectures, and "Lipschitz-constrains the learned inverse on the sm-noise sphere around each anchor" is asserted as a *property of the augmentation*, not justified from the cited theorem's hypotheses. |
| "the heavier-tailed antennas retain a few percent of energy above the bandlimit that no $L=5$-truncated learned model can ever recover." (§2.2) | overreach — the impossibility ("can ever recover") is asserted without an argument: the cter has not established that the learned regulariser cannot also output coefficients that *anti-alias* above-bandlimit content into below-bandlimit shape, only that the resynthesised field is bandlimited. The intended weaker statement ("cannot recover above-bandlimit energy in the resynthesised field") is true; the stated stronger one is not justified in the chapter. |
| "A naive dense linear layer mapping the input to the $4K = 140$-dimensional packed coefficient vector already requires $n_\mathrm{ang}\cdot 4K \approx 9.0\times 10^6$ scalar weights — feasibl but not informative as a first stage when the angular structure of $P$ has far fewer than $64{,}440$ informative directions." (§2.4) | "not informative as a first stage" is a value judgment that motivates the PCA design; the underlying claim that "$P has far fewer than $64{,}440$ informative directions" is plausible but no spectrum diagnostic or rank estimate is provided to back it. |
| "the procedural effect of this stage is to push the per-dimension feature distribution close to zero mean and unit variance, which keeps the downstream regressor's initialisation scale meaningful" (§2.6) | invokes "the downstream regressr's initialisation scale" as if its semantics were established in scope; nothing in §§2.1–2.6 says how initialisation is done. |

---

## Verdict

**MAJOR REVISION.**

The chapter's spis well-aligned with the outline (Coherence is `PASS` end-to-end), the augmentation taxonomy of §2.1 is the strongest part of the chapter, and the §2.4 randomised-SVD treatment is correctly anchored in Halko–Martinsson–Tropp. But three categories of defects bite hard for an `implementation / experiment` chthat the chapter itself opens by promising will be fully specified:

1. **Self-containedness gap on quantitative parameters.** $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}, N_\mathrm{aug}$ are named in §2.2 and never bound or explicitly deferred, in direct contradiction of the opener "*By the end of §2.2 the training distribution itself is fully specified*".

2. **Code-state references the reader cannot verify.** The "framework configures a flag the implementation does not act on" framing in2.6 and the unbound snake_case identifiers in the §2.2 augmentation chain (`field_phi_roll`, `coef_mode_dropout`, `field_additive_noise`) both lean on artefacts outside the text. The readr does not have the framework.

3. **Notation hygiene.** $p$, $r$, $X$, and $a$ each carry at least two distinct meanings across the chapter without warning, and the `(K, 2, 2, n_\theta, n_\phi)` tensor is identified with §1.3's `B ∈ ℂ^{70×128880}` only by tacit reshape.

The "well-known pole-region bias" claim in §2.5.1 and the overreach in the §2.2 bandlimit paragraph and the §2.1 Primitive-4 Lipschitz claim are all fixable binserting a citation or weakening the statement, and should be fixed in the same revision pass.
