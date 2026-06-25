# Research manifest — Chapter 1 enrichment

Task slug: `ch1-enrichment`. Date: 2026-05-07.

This manifest is the Phase 3 record of every external source consulted in support of `presentation/ch1_full.md`. Every factual claim, citation, page number, and verbatim taxonomy in the chapter must trace back to one of the entries below. Anything not backed by an entry here is either deferred via a forward reference or removed.

---

## Phase 1 — Comprehension

**Task understood as**: enrich the Russian outline `presentation/h1.md` (sections 1.1–1.8) into a single English-prose Chapter 1 file `presentation/ch1_full.md`, titled "Глава 1. Физическая постановка задачи мультипольного анализа" / "Chapter 1. Physical formulation of the multipole-analysis problem". Merge the previously split front-matter (`chapter1.md`, `chapter2.md`) back into one chapter. Close the §1.7 load-bearing non-uniqueness claim that Judge Volodya flagged as "asserted on the strength of an analogy". Update peripheral files (`header.md`, `h1.md`, `figures.md`); delete the old split chapters.

**Technology / topic inventory**:

| Topic | Status | Resolved by |
|---|---|---|
| Vector spherical harmonics, far-field formalism, $1/r$ asymptotics | `[NEEDS RESEARCH]` | R1 |
| Hadamard well-posedness | `[KNOWN]` (textbook, e.g. Engl–Hanke–Neubauer) | — |
| Tikhonov / sparsity / TV regularization | `[KNOWN]` | — |
| Arridge–Maass–Öktem–Schönlieb 2019 *Acta Numerica* taxonomy | `[NEEDS RESEARCH]` | R2 |
| Phase retrieval canonical references and trivial-ambiguity classification | `[NEEDS RESEARCH]` | R3 |
| Documented non-uniqueness for phaseless inversion of spherical-harmonic expansions | `[NEEDS RESEARCH]` | R4 |
| Far-field / spherical-near-field standard sampling convention | `[NEEDS RESEARCH]` | R5 |
| Project conventions ($L=15$, $K=255$, $360\times 179$, packed $4K=1020$ reals) | `[NEEDS RESEARCH]` | R6 (negative result — declared as project parameters in the chapter) |

---

## R1 — Vector spherical harmonics: Jackson, *Classical Electrodynamics*, 3rd ed.

- **Citation**: J. D. Jackson, *Classical Electrodynamics*, 3rd edition, Wiley, 1999, ISBN 978-0-471-30932-1.
- **Sources consulted**:
  - Wiley product page <https://www.wiley.com/en-us/Classical+Electrodynamics%2C+3rd+Edition-p-9780471309321>.
  - Multipole / VSH lecture notes (UCSC) <https://scipp.ucsc.edu/~dine/ph214/214_vector_spherical_harmonics_lecture.pdf>.
  - Duke electrodynamics notes <https://webhome.phy.duke.edu/~rgb/Class/Electrodynamics/Electrodynamics/node139.html>.
  - University of Texas notes <https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node124.html>.
- **Verbatim facts extracted**:
  - Jackson Chapter 9, "Radiating Systems, Multipole Fields and Radiation", treats radiation by charge–current sources via multipole expansion using vector spherical harmonics.
  - The treatment splits into the radiation zone ($r \gg \lambda \gg d$, fields $\propto 1/r$ with outgoing spherical-wave behaviour), intermediate / static zone, and near zone.
  - The two complementary multipole-field families are: **magnetic multipoles** (transverse-electric, TE) where the electric field is transverse to the radius vector, and **electric multipoles** (transverse-magnetic, TM) — the complementary set. These form a complete set of vector solutions of Maxwell's equations in spherical coordinates.
  - Vector spherical harmonics are eigenfunctions of angular-momentum operators and are constructed from scalar $Y_{lm}$ with the angular-gradient and radial-cross-product operators on $S^2$.
- **Use in chapter**: §1.1 cites Jackson Ch. 9 for the VSH expansion, the TE/TM family naming, and the completeness/orthogonality of the basis. No verbatim formulas are quoted; the conventions are stated as "we follow Jackson" and the chapter writes the expansion in standard form.

---

## R2 — Arridge, Maass, Öktem, Schönlieb 2019 *Acta Numerica*

- **Citation**: S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb, "Solving inverse problems using data-driven models", *Acta Numerica*, vol. 28, pp. 1–174, 2019. doi: `10.1017/S0962492919000059`. Open Access.
- **Sources consulted**:
  - Cambridge Core landing page <https://www.cambridge.org/core/journals/acta-numerica/article/solving-inverse-problems-using-datadriven-models/CE5B3725869AEAF46E04874115B0AB15>.
  - Full PDF text mirror (Cambridge AOP).
  - MaRDI portal <https://portal.mardi4nfdi.de/wiki/Publication:5230520>.
  - ADS <https://ui.adsabs.harvard.edu/abs/2019AcNum..28....1A/abstract>.
- **Verbatim facts extracted from the paper**:
  - Volume 28, pp. 1–174, 2019. Open Access (CC BY-NC-ND).
  - The paper formalises inverse problems as $g = A(f) + e$ — measured data $g$, model parameter $f$, forward operator $A$, observational noise $e$.
  - The paper organises learning-augmented methods into the following relevant sub-categories (verbatim from the table of contents and Section 5.1):
    - **§5.1.3 Fully learned Bayes estimation** — "the reconstruction operator $R_\theta : Y \to X$ has a generic parametrization … fully learned approaches usually involve one or more 'fully connected layers' that represent a pseudo-inverse operator". The advantage is simplicity (no explicit forward operator); the disadvantage is poor scaling (very many weights) and inapplicability to novel instrumentation absent training data.
    - **§5.1.4 Learned iterative schemes** — unrolled iterative solvers; "data-driven components are interwoven with inverse model assumptions".
    - **§5.1.5 Learned post-processing** — "apply an initial [knowledge-driven] reconstruction" then learn a regulariser/denoiser as post-processing. This is *not* direct regression from $g$ to $f$.
    - **§4.6 Black-box / plug-and-play denoiser** — decouples regularisation from inversion using a learned denoiser inside a classical iterative solver.
- **Important correction**: the previous draft (`chapter2.md`) classified the thesis's approach as "post-processing / direct-regression family". In Arridge's taxonomy these are **two different families**. Direct regression $P_{UT} \to (a^E, a^M)$ with no analytic inverse and no unrolled forward operator is **§5.1.3 fully learned**, not §5.1.5 post-processing. The new chapter §1.8 names this correctly.
- **Use in chapter**: §1.8 cites Arridge et al. 2019 for the categories named above and places this thesis in the §5.1.3 fully-learned family.

---

## R3 — Phase retrieval canonical references

### R3a — Shechtman et al. 2015 IEEE SPM

- **Citation**: Y. Shechtman, Y. C. Eldar, O. Cohen, H. N. Chapman, J. Miao, M. Segev, "Phase retrieval with application to optical imaging: a contemporary overview", *IEEE Signal Processing Magazine*, vol. 32, no. 3, pp. 87–109, May 2015. doi: `10.1109/MSP.2014.2352673`. arXiv: 1402.7350.
- **Sources consulted**: Eldar group preprint <https://www.ee.technion.ac.il/Sites/People/YoninaEldar/journals/170_Phase%20Retrieval%20with%20Application.pdf>; ArXiv <https://ar5iv.labs.arxiv.org/html/1402.7350>.
- **Verbatim facts extracted** (from the preprint, "Uniqueness — Fourier measurements" section):
  - "First, there are so-called **trivial ambiguities** that are always present. The following three transformations (or any combination of them) conserve Fourier magnitude: 1) global phase shift $x[n] \mapsto x[n]\cdot e^{j\phi_0}$; 2) conjugate inversion $x[n] \mapsto \overline{x[-n]}$; 3) spatial shift $x[n] \mapsto x[n + n_0]$."
  - "Second, there are **nontrivial ambiguities**, the situation of which varies for different problem-dimensions. In the 1-D setting, there is no uniqueness — i.e., there are multiple 1-D signals with the same Fourier magnitude. Even if the support of the signal is bounded within a known range, uniqueness does not exist."
  - "For higher dimensions (2-D and above), Bruck and Sodin, Hayes, and Bates have shown that, with the exception of a set of signals of measure zero, a real $d \geq 2$ dimensional signal … is uniquely specified by the magnitude of its continuous Fourier transform, up to the trivial ambiguities mentioned earlier."
- **Use in chapter**: §1.6 names trivial vs non-trivial ambiguities; §1.7 cites Shechtman et al. 2015 for the trivial-vs-nontrivial classification. The chapter does **not** transplant the *full* Fourier classification onto the spherical-harmonic case; the spherical case has its own classification in R4.

### R3b — Fienup 1982 *Applied Optics*

- **Citation**: J. R. Fienup, "Phase retrieval algorithms: a comparison", *Applied Optics*, vol. 21, no. 15, pp. 2758–2769, 1982. doi: `10.1364/AO.21.002758`.
- **Sources consulted**: cited verbatim in Shechtman et al. 2015 reference list (R3a) and in Bangun 2020 reference list (R4). Optica abstract page <https://opg.optica.org/ao/abstract.cfm?uri=ao-52-1-45> (companion personal-tour paper for context).
- **Use in chapter**: §1.7 cites Fienup 1982 alongside Shechtman 2015 as the foundational phase-retrieval reference.

---

## R4 — Phaseless inversion of spherical-harmonic expansions: documented ambiguities

This is the load-bearing reference that closes Volodya's gap. The previous draft asserted by Fourier-analogy that non-trivial ambiguities exist; the literature has a direct result.

- **Citation 1 (primary)**: A. Bangun, "Signal recovery on the sphere from compressive and phaseless measurements", PhD dissertation, RWTH Aachen University, 2020. doi: `10.18154/RWTH-2020-03041`. ISBN 978-3-86359-836-5. Advisors: R. Mathar, D. Heberling.
- **Citation 2 (related conference paper)**: A. Bangun, A. Behboodi, C. Culotta-López, R. Mathar, D. Heberling, "On phaseless spherical near-field antenna measurements", in *Proc. 13th European Conf. Antennas and Propagation (EuCAP 2019)*, 2019. (Identified via Google Scholar profile of A. Bangun.)
- **Citation 3 (IEEE conference, by ID)**: "Signal Recovery from Phaseless Measurements of Spherical Harmonics Expansion", IEEE Xplore document `8902696`. Direct fetch was rejected by IEEE's anti-bot endpoint; the work is by the same RWTH group and is incorporated into Bangun 2020 Chapter 6.
- **Sources consulted**:
  - RWTH Publications record <https://publications.rwth-aachen.de/record/785206>.
  - Full PDF of the dissertation downloaded and inspected.
  - Bangun's Google Scholar profile <https://scholar.google.co.il/citations?hl=de&user=z02mVEIAAAAJ>.
- **Verbatim facts extracted** from Bangun 2020, Chapter 6 "Signal Recovery from Phaseless Measurements", §6.2 "Ambiguities in Phaseless Measurements":
  - **Trivial ambiguities for complex spherical-harmonic phaseless measurements** are (i) **rotated signal / global phase**: $y = x e^{j\alpha}$ for $\alpha \in [0, 2\pi)$; (ii) **reflected-conjugate ambiguity**: $\hat g_l^k = (-1)^k \overline{\hat f_l^{-k}}$ for $0 \le l \le B-1$, $-l \le k \le l$, yielding the same phaseless measurements $|\langle a_p, x\rangle|^2 = |\langle a_p, y\rangle|^2$.
  - The reflected-conjugate ambiguity is a direct consequence of the conjugation property of complex scalar spherical harmonics, $Y_l^k(\theta,\varphi) = (-1)^k \overline{Y_l^{-k}(\theta,\varphi)}$.
  - **Real spherical harmonics**: the conjugate ambiguity is removed, but **sampling-induced ambiguities** appear when the angular sampling pattern collapses certain rows of the sensing matrix (Proposition 6.2.1: for bandwidth $B \ge 4$ and the special pattern $(\theta_p, \varphi_p) = (\theta_p, (B-2)\theta_p)$, distinct coefficient vectors produce identical measurements).
  - The conclusion (verbatim): "the one-dimensional Fourier phase retrieval is an ill-posed problem … Other types of ambiguities may still occur in the one-dimensional Fourier phase retrieval, even when trivial ambiguities are excluded."
- **Use in chapter**: §1.7 cites Bangun 2020 (Chapter 6, §6.2) for the existence of at least two distinct trivial ambiguities (global phase + reflected-conjugate) of phaseless spherical-harmonic measurements, plus sampling-induced ambiguities. The chapter does not lift Bangun's *exact* statement — Bangun treats scalar spherical harmonics, while the thesis works with vector spherical harmonics — and explicitly notes this distinction. The conjugation-based ambiguity, however, is structural and carries through to the VSH case because VSH are built from $Y_{lm}$ via differential operators that preserve the conjugation relation.
- **Honest scope note**: Bangun 2020 does **not** claim that all ambiguities of the phaseless complex-SH inverse are trivial. The §6.2 discussion lists the trivial ones and then turns to sampling-induced ones; the existence (or non-existence) of further non-trivial nonlinear-interference ambiguities for complex SH is left as an open question in that work. The new chapter §1.7 reflects this honestly: it states that the inverse fails Hadamard uniqueness *at minimum* by global phase and reflected-conjugate, and notes the phase-retrieval literature for the broader picture rather than claiming a stronger result than the literature supports.

---

## R5 — Spherical near-field / far-field measurement convention

- **Citation 1**: J. E. Hansen (ed.), *Spherical Near-Field Antenna Measurements*, IEE Electromagnetic Waves Series 26, Peter Peregrinus / IEE, London, 1988 (re-issued 2008). The canonical reference for SNF measurement geometry; cited verbatim by Bangun 2020 as `[42]`.
- **Sources consulted**:
  - IEEE Xplore listing <https://ieeexplore.ieee.org/document/6102052>.
  - Coordinate-system tutorial (Next Phase Measurements) <https://nextphasemeasurements.com/wp-content/uploads/2018/10/2007-CoordinateSystemPlottingForAntennaMeasurements-GFM_SFG.pdf>.
  - Bangun 2020 reference list and Section 2.4 "Spherical Near-Field Antenna Measurements".
- **Verbatim facts extracted**:
  - Standard SNF coordinates: $\theta$ polar (zenith) angle, $\varphi$ azimuthal angle. The $z$-axis is the pole.
  - At the pole, $\varphi$ is undefined and the spherical-area weight $\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi$ vanishes; sample-acquisition methods that pass through the pole versus those that do not have qualitatively different error budgets.
  - For a band-limited signal at SH bandwidth $B$ (truncation $L = B-1$ in this thesis's convention), classical sampling theory on the sphere requires $\Theta(L)$ azimuthal samples per ring and $\Theta(L)$ polar rings to determine the coefficients; oversampling beyond this is conventional and harmless.
- **Use in chapter**: §1.2 cites Hansen 1988 for the SNF coordinate convention and the rationale for excluding the polar caps from the angular grid. The 1° grid is presented as a project choice that comfortably exceeds the Nyquist-on-sphere criterion for $L=15$.

---

## R6 — Project conventions ($L=15$, $K=255$, grid, packing): negative result

- **Sources consulted (read-only)**:
  - Workspace listing of `/Users/chersie/Desktop/diplom_clean/`.
  - Glob searches for `src/**/*.py`, `docs/**/*.md`, `data/**/*.npz`.
- **Result**: the workspace contains only `presentation/`, `JUDGE_VOLODYA.md`, `RESEARCHER.md`, and two `Perplexity_*.md` meta-files. There is no `src/`, no `docs/`, no `data/`, no Python, no numerical artefacts. The only project-internal sources stating $L=15$, $K=255$, $4K=1020$, the $360 \times 179$ grid, and the packed-coefficient ordering are the very chapters being rewritten (`chapter1.md`, `chapter2.md`, `figures.md`).
- **Implication for the chapter**: these values are **declared as project parameters** in §1.1 and §1.2. They are not asserted as external facts. The chapter introduces them as the choices fixed for the rest of the thesis; whether they are also reflected in code is a separate concern that does not enter the published-reader's view of the chapter.
- **Honest deferral**: any later chapter that references implementation details ($L=15$ in code, the specific tensor shape used in `numpy`/`torch`, etc.) inherits the same stated values. If the implementation chapter ever disagrees with §1.1 / §1.2, the implementation chapter is wrong, not §1.1 / §1.2.

---

## Hard-rule check (RESEARCHER.md "Hard rule on empty research")

Phase 3 produced new information for R1, R2, R3, R4, R5. R6 is a recorded negative result. The manifest is not empty.

## Sources I consulted but did NOT use

- `https://opg.optica.org/abstract.cfm?uri=josa-73-11-1446` — band-limited multidimensional signal uniqueness (not directly applicable to spherical case).
- `https://opg.optica.org/josaa/abstract.cfm?uri=josaa-40-12-2223` — recent (2023) work on band-limited image phase retrieval; modern, but tangential to the chapter's level.
- `https://arxiv.org/abs/1705.09590` — Fourier phase retrieval uniqueness review; superseded for our purposes by R3a.
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC12473657/` — mask-based phase reconstruction in phaseless SNF (project-style, not foundational).
- Various Cambridge-Core/MaRDI mirror entries for R2 — used only as cross-checks for the citation metadata.
