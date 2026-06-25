# Volodya's Report — Chapter 2

Scope under review: `presentation/paper_full.md`, §§2.1–2.6 (the block titled *"Chapter 2. Training-data generation. Feature representation and preprocessing"*).
Reference: `presentation/header.md`.
Out of scope: source code, configs, JSON results, chapters 4–7 except where the chapter itself forward-references them.

---

## Coherence

| § | Heading | Body match | Status |
|---|---|---|---|
| 2.1 | Synthetic data generator | Specifies the synthetic generator, 4 distribution modes, 3 composable knobs, 5 augmentation primitives. Header note also asks for "В чём смысл каждой генерации коэффициентов и вида аугментации" — only partly addressed (motivation for Mode 4 `sparse` is absent, motivation for the "Post-hoc mode dropout" knob is absent). | PARTIAL |
| 2.2 | Real data used in training and validation | Specifies file format, $L=5$ truncation residuals, split protocol, augmentation chain. Header note ("опation chain on the training pool"* sub-section uses **only three** of the five primitives introduced in §2.1 without saying why Primitives 1 and 2 are dropped. | PARTIAL |
| 2.3 | Input modes: power, magnitude, complex | Body matches heading. | PASS |
| 2.4 | Randomised PCA as a dimensionality-reduction stage | Body matches heading; the exact form $(N, 64440)$ promised in the header note is delivered. | PASS |
| 2.5 | Additional computer-vision features (with 2.5.1, 2.5.2) | Subsections match the header. The trailing sub-section **"Composite feature stacks"** is not present in `header.md` under §2.5 and is not a "CV feature" — it is a composition of all previous features. Mild drift. | PARTIAL |
| 2.6 | Normalisation of features and target variables | Body matches heading; target-normalisation paragraph correctly forward-references Chapter 5. | PASS |

Section ordering vs. `header.md`: PASS.

---

## Self-containedness

### Unresolved assumptions

- *"Across the full real corpus $\rho_P$ has mean $1.16\median $0.36\%$, and maximum $8.08\%$; $\rho_E$ has mean $1.30\%$, median $0.32\%$, and maximum $10.05\%$."* — **argument needed**. These specific percentages are load-bearing: they justify the sentence *"The typical antenna is well-described at $L = 5$"*, which is itself the justification for the §2.2 bandlimit choice. The chapter does not say how the numbers were computed, on which subset, or with which discretisation. A reader cannot verify them from the chapter alone.
- *"the same operation as Mode 4 above, applied **after** sampling rather than **as** the sampling distribution."* — **argument needed**. The distinction between "Post-hoc mode dropout" and Mode 4 `sparse` is stated, but the *purpose* of duplicating the operation as a knob is not. Why would a user combine "Post-hoc mode dropout" with a non-sparse mode rather than just switching to Mode 4?
- *"$\mathrm{SO}(2)$ azimuthal-rotation symmetry of $\mathcal A$"* (§2.1, Primitive 5) — **argument needed**. Chapter 1 §1.7 catalogues exactly al symmetries: the global phase $U(1)$ and the reflected-conjugate reflection. A third symmetry — $\mathrm{SO}(2)$ azimuthal equivariance of $\mathcal A$ — is now introduced without back-reference. The chapter sketches the action on scalar $Y_{lm}$ but does not prove that the **vector** spherical harmonics $\boldsymbol\Psi^X_{lm}$ inherit the $e^{-im\varphi_k}$ phase factor (the analogous lifting was carried out explicitly for reflected-conjugate in §1.7; here it is waved through).
- *"the discrete roll and the coefficient-domain phase rotation stay consistent"* — **argument needed**. The claim depends on the 1° grid sampling $\varphi_k = 2\pi k / n_\phi$ exactly at integer $k$. The chapter asserts this without showing that $\varphi$-bin centres (or edges?) land on these values exactly.
- *"$\varepsilon$ a complex standard normal"* (Primitive 2) — **definition needed**. There are two competing conventions for "complex standard normal" (per-component $\mathcal N(0, 1/2)$ vs. per-component $\mathcal the chapter does not fix one.
- *"with default colour exponent $\alpha = 1$"*, *"default active-mode fraction $p_\mathrm{active} = 0.1$"*, *"default parameters $\sigma_a = 0.05$, $p = 0.1$, $\sigma_P = 0.02$"*, *"canonical configuration uses $r = 128$ components"* — these are **engineering defaults**. The chapter says they are "revisited in Chapter 7", which is acceptable deferral. **Not flagged as defects.**
- *"an optional $\sin\theta$ pre-window"* (§2.5.1) — **definition needed**. Optional means it is sometimes off. The chapter does not say which is the default for the canonical pipeline, so the descriptor is not fully specified.
- *"`numpy.random.default_rng`"*, *"`sklearn.decomposition.PCA` with `svd_solver='randomized'`"*, *"complex64"* — implementation appeals. Acceptable for a pipeline-spec chapter because all three resolve to public, well-documented APIs.

### Forward references

- *"settled in Chapter 7"* (Mode selection) — correct deferral.
- *"reported in the experimental tables of Cha(counts $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}, N_\mathrm{aug}$, seeds) — correct deferral.
- *"revisited in Chapter 7"* (augmentation defaults) — correct deferral.
- *"the coefficient-MSE loss of §5.1"*, *"the physics-informed power loss of §5.2"*, *"$\|\cdot\|_w$-norm of §1.8"* — correct deferrals/back-references.
- *"model (Ch. 4)"* at the end of the pipeline diagram in §2.6 — correct deferral.

No forward reference in this chapter masquerades as missing load-bearing material.

---

## Terminology and notation

### Non-standard or undefined terms

- *"complex64"* (§2.2) — numpy dtype jargon, not a textbook term. Resolvable, but a reader from outside numerical Python will pause.
- *"PCA + CV"*, *"raw + SH"* (§2.5, *Composite feature stacks*) — local internal labels that may not survive into later chapters; the chapter introduces them as if they were canonical and uses them in a forward-pointing way ("are used in the experimental chapters").

### Notation collisions and ambiguitiesent Bernoulli parameters with conflicting semantics**:
  - Mode 4 `sparse`: *"$m^X_{lm}\sim\mathrm{Bern}(p_\mathrm{active})$"* — keep probability;
  - Knob "Post-hoc mode dropout": *"$d^X_{lm}\sim\mathrm{Bern}(1 - p_\mathrm{drop})$"* — drop probability;
  - Primitive 3: *"For $p\in[0,1]$ ... $d^X_{lm}\sim\mathrm{Bern}(1-p)$"* — drop probability under a third symbol.

  The relation $p_\mathrm{drop} = 1 - p_\mathrm{active}$ is neverd, and the reuse of $d^X_{lm}$ for two distinct masks ("Post-hoc mode dropout" knob and Primitive 3) is unmarked.

- **Mode 4 vs Mode 1 normalisation disparity**:
  - Mode 1: *"$\frac{1}{\sqrt 2}(\mathcal N(0,1) + i\,\mathcal N(0,1))$"*;
  - Mode 4: *"$a^X_{lm} = m^X_{lm}\cdot(\mathcal N(0,1) + i\,\mathcal N(0,1))$"* — **no $1/\sqrt 2$**.

  This is either an intentional nvention break (Mode 4 has $\mathbb E|a|^2 = 2 p_\mathrm{active}$ for active modes, not $p_\mathrm{active}$) or a typo. The chapter does not flag it.

- **Shape convention drift**: §1.2 esablishes *"an array of shape $(360, 179, 2)$"* for the complex field; §2.2 then writes *"an array of shape $(2, n_\theta, n_\phi) = (2, 179, 360)$ of complex64"* for the same object. The two conventions transpose the trailing axes; the chapter does nt say which is canonical or why the layout was permuted.

- **$N_{\mathrm{ang}}$ (§1.3, $= 128{,}880$) vs $n_\mathrm{ang}$ (§2.4, $= 64{,}440$)** — different objects under almost the same symbol; only the case and the inclusion/exclusion of the $C$ multiplier distinguish them. The chapter does not mark the distinction.

- **FFT formula in §2.5.1**: the sum *"$\sum_{\thearphi}\tilde f(\theta,\varphi)\,e^{-i 2\pi(k_\theta \theta/n_\theta + k_\varphi \varphi/n_\phi)}$"* uses the same symbols $\theta, \varphi$ as both angles (with values in degrees elsewhere in the chapter) and as integer sample indices. The reader has to silently retype them.

---

## Depth

Chapter role inferred from `header.md`: **implementation / specification** (data pipeline and featurisation pipeline; not a formulation chapter).

Depth assessment: **mostly matches role, with two patches of "too shallow"** relative to what the chapter's role demands.

- *"This is the unbiased baseline."* / *"is a heavy-tail-free baseline; it never produces extreme samples."* / *"Larger $\alpha$ pushes energy toward the lower-$l$ sector"* / [Mode 4: **no motivation sentence at all**] — header note for §2.1 explicitly demands "В чём смысл каждой генерации коэффициентов и вида аугментацor Modes 1–3 it is one-line and does not connect to the physical priors discussed in §1.8. **expand here.**

- *"Primitive 5 (azimuthal roll) → Primitive 3 (coefficient mode dropout) → Primitive 4 (field additive noise)."* — only three of thimitives are used in the production augmentation chain; the omission of Primitives 1 and 2 is unexplained. For an implementation chapter, this leaves the recipe under-specified. **expand here** (either justify the omission or add the missing primitives to the chain).

- *"the relative residuals $\rho_P$ ... has mean $1.16\%$, median $0.36\%$, and maximum $8.08\%$"* — for an implementation chapter quoting concrete percentages, the computation method and the corpus subset must be specified. **expand here.**

No "too detailed" defects; the chapter does not drain network architectures, training schedules, or hyperparameter tables, which is correct.

---

## Strictness gaps

- *"By construction the descriptor is azimuthally invariant: integer azimuthal rolls of the input leave the radial-bin amplitudes unchanged, since each Fourier component picks up a unit-modulus phase under a roll."* — invariance under integer azimuthal rol is plausible and the half-line of justification is on the right track, but the argument quietly ignores the radial binning step: the radius $\sqrt{k_\theta^2 + k_\varphi^2}$ depends on $k_\theta$, which is not the roll axis. The result is still true (the per-component magnitudes are roll-invariant pointwise, hence so are any of their aggregates), but the chapter does not actually say this.

- *"the rescaling is the identity"* at $b = 0.5$ in the Family-balance knob — true ($2(1-0.5) = 2 \cdot 0.5 = 1$), but the factor of 2 has no stated motivation. A reader would expect a convex combination $(1-b)\,a^E + b\,a^M$ or a multiplicative renormalisation; instead the chapter uses an asymmetric rescaling that is *neither* and never says why.

- *"The order places the strict-symmetry primitive first and e consistency-breaking noise last, so that the noise added by Primitive 4 is not erased by the re-synthesis step inside Primitive 3."* — the second half is correct, but the first half (trict-symmetry first") is not actually argued. If 5 is a strict symmetry, the placement is permutation-free with respect to coefficient operations followed by re-synthesis; putting it first is a convention, not a derived requirement. The chapter presents it as a justified choice.

- *"COMPLEX is the fully-measured regime of §1.3 on which the analytic inverse ... is available, included as the diagnostic upper bound from §1.3"* — Chapter 1 §1.3 indeed presents the analytic inverse as the golddard. Chapter 2 promotes it to *"diagnostic upper bound"*, which Chapter 1 does not literally say. Plausible inference, not a quotation. Minor.

---

## Verdict

**MAJOR REVISION.**

The chapter is well-organised and section-by-section matches `header.md`. The blocking issues are not a single fatal flaw; they are a cluster of self-containedness and notation gaps that together undermine the chapter's purpose as a reproducible specification.

The most consequential are: (a) the empirical residual statistics in §2.2 (*"$\rho_P$ has mean $1.16\%$, median $0.36\%$, and maximum $8.08\%$"*) are load-bearing for the $L=5$ bandlimit choice but are not sourced — the chapter does not swho computed them, on which subset, with which discretisation; (b) the augmentation chain in §2.2 uses only three of the five primitives introduced in §2.1, without explaining the omission of Primitives 1 and 2; (c) the *"$\mathrm{SO}(2)$ azimuthal-rotation symmetry of $\mathcal A$"* invoked in Primitive 5 is ahird symmetry of $\mathcal A$ beyond the two catalogued in §1.7, and is sketched only for scalar $Y_{lm}$ — the lift to the vector basis is not made explicit even though Chapter 1 did exly that lift for reflected-conjugate; (d) cumulative notation drift — three different Bernoulli parameters with conflicting semantics, Mode 4 silently dropping the $1/\sqrt 2$ scaling that Mode 1 uses, the field array shape transposed between §1.2 and §2.2, and the $N_\mathrm{ang}$/$n_\mathrm{ang}$ near-colln — each minor in isolation but corrosive in aggregate; (e) Mode 4 and the "Post-hoc mode dropout" knob enter with no stated motivation, in a chapter whose own TOC entry promises "В чёсл каждой генерации коэффициентов". Each of these is patchable with a few paragraphs and a few notational decisions, but the patches must land before the chapt
