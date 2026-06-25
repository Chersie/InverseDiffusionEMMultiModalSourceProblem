# Chapter 1. Physical formulation of the multipole-analysis problem

*Глава 1. Физическая постановка задачи мультипольного анализа.*

This chapter is the entire physical and mathematical front matter for the thesis. It introduces the radiating-system objects we work with, the analytic decomposition that defines the multipole signature, the practical measurement modality that prevents the analytic decomposition from being used directly, and the inverse problem that the rest of the thesis is dedicated to solving. By the end of the chapter the problem statement, the reasons it is hard, and the rationale for a learned-regularisation approach are all in place; the chapters that follow turn each of these into a concrete construction.

Throughout §§1.1–1.3 the field is treated as given and known exactly. Everything in those three sections is reversible: the multipole coefficients of a fully measured complex field can be recovered by a single linear projection. The irreversibility — the loss of phase, the collapse of information, and the resulting ill-posedness — appears for the first time in §1.4, is given a formal statement in §1.5, is quantified in §1.6, is shown to violate Hadamard uniqueness in §1.7, and is answered with a learned regulariser in §1.8.

## 1.1. Multipole expansion in electromagnetism: vector spherical harmonics and coefficients

A monochromatic, source-free electromagnetic field outside the smallest sphere enclosing its sources admits a unique expansion in **vector spherical harmonics** (VSH). The angular dependence of the radiated electric field separates into two orthogonal mode families, conventionally called *electric* (or *transverse magnetic*, TM) and *magnetic* (or *transverse electric*, TE) multipoles:

\[
\mathbf{E}(\theta,\varphi) \;=\; \sum_{l=1}^{\infty}\sum_{m=-l}^{l}
\Bigl[\, a^E_{lm}\,\boldsymbol{\Psi}^E_{lm}(\theta,\varphi) \;+\; a^M_{lm}\,\boldsymbol{\Psi}^M_{lm}(\theta,\varphi)\,\Bigr].
\]

Here $\boldsymbol{\Psi}^E_{lm}$ and $\boldsymbol{\Psi}^M_{lm}$ are the vector-valued angular mode functions built from the scalar spherical harmonics $Y_{lm}$ via the angular gradient and the radial-cross-product operators on $S^2$. The construction is standard: we follow Jackson, *Classical Electrodynamics*, 3rd ed., Chapter 9, both in convention and in normalisation. The complex weights $a^E_{lm}\in\mathbb{C}$ and $a^M_{lm}\in\mathbb{C}$ are the **multipole coefficients**, and they are the canonical compact representation of any radiating structure.

Two textbook properties of the expansion are used repeatedly in this thesis. **Completeness**: any square-integrable tangential vector field on $S^2$ can be reconstructed from its $\{a^E_{lm}, a^M_{lm}\}$ to arbitrary precision. **Orthogonality**: distinct $(l,m)$ modes integrate to zero against one another under the spherical inner product (made explicit in §1.3). Lower orders ($l=1, 2$) describe broad, dipolar and quadrupolar lobes; higher orders capture finer angular structure.

In practice the expansion must be truncated. The truncation order $L$ controls how much angular detail the model can represent. For each family there are
\[
K \;=\; \sum_{l=1}^{L}(2l+1) \;=\; L(L+2)
\]
distinct $(l,m)$ pairs, and the complex pair $(a^E_{lm}, a^M_{lm})$ contributes four real numbers per $(l,m)$. **In this thesis we fix $L=5$**, giving $K=35$ modes per family and a packed real coefficient vector of width $4K=140$ in the order $[\,\mathrm{Re}\,a^E,\,\mathrm{Im}\,a^E,\,\mathrm{Re}\,a^M,\,\mathrm{Im}\,a^M\,]$. The packing convention is fixed once here and used unchanged from Chapter 2 onward.

The compactness of this representation is the practical motivation for working in the multipole basis. A discretised radiation pattern at the angular resolution introduced in §1.2 carries on the order of $2.6\times 10^5$ real numbers; its multipole signature carries $140$. The compression ratio is roughly $1800{:}1$, and the compressed representation is *physical* — each coefficient corresponds to a definite TE or TM angular mode. Recovering $(a^E, a^M)$ from a measurement is therefore not data compression but identification of the radiator in a basis that is intrinsic to the radiation problem.

## 1.2. The radiation pattern as object of analysis: complex field on the $360 \times 179$ angular grid

The forward direction of the multipole map produces a far-field radiation pattern, by which we mean the angular distribution of the radiated electric field in the far zone, with the radial $1/r$ factor stripped off and the time-harmonic dependence suppressed. In the far zone the radial component of $\mathbf E$ vanishes, so the field is purely tangential to the sphere, and a basis of two complex scalars per direction suffices:

\[
E_{UT}(\theta,\varphi) \;=\;\bigl(E_\theta(\theta,\varphi),\; E_\varphi(\theta,\varphi)\bigr) \;\in\; \mathbb{C}^2,
\]

where the subscript "UT" denotes *under test* — the field whose multipole signature we wish to identify. The polar angle $\theta$ is the zenith and $\varphi$ is the azimuth, in the standard convention of Hansen, *Spherical Near-Field Antenna Measurements* (IEE Electromagnetic Waves Series 26, 1988), which is also the convention adopted by the antenna-measurement literature this thesis builds on.

We sample $E_{UT}$ on a uniform angular grid with one-degree resolution. The azimuth $\varphi$ runs from $0°$ to $359°$ in 360 steps; the polar angle $\theta$ runs from $1°$ to $179°$ in 179 steps. The two polar caps $\theta=0$ and $\theta=180°$ are excluded from the grid because the spherical area element $\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi$ vanishes there: a sample at the poles carries no integration weight, contributes no information to the spherical inner products of §1.3, and in the discrete representation introduces an artificial coordinate singularity since $\varphi$ is undefined at $\theta\in\{0,\pi\}$. Removing the poles costs nothing physically and removes the singularity by construction. The same exclusion is standard in spherical-harmonic measurement practice (Hansen 1988).

The full discretised field therefore lives in an array of shape $(360, 179, 2)$ with complex entries. Counting real degrees of freedom:

\[
\dim_{\mathbb R}(E_{UT}) \;=\; \underbrace{2}_{\text{complex} \to \text{real}} \times\; \underbrace{360}_{\varphi}\times\; \underbrace{179}_{\theta}\times\; \underbrace{2}_{(E_\theta,\,E_\varphi)} \;=\; 257{,}760.
\]

This is the raw data resolution against which the analytic inverse of §1.3 and the data-driven inverse of Chapters 2 onward are measured.

The choice of a one-degree grid is a deliberate over-sampling. For truncation $L=5$ the highest mode $\boldsymbol{\Psi}_{L,L}$ has $2L=10$ azimuthal nodes, well below the $360$ available azimuthal samples; standard sampling theory on the sphere requires roughly $2L+1$ azimuthal samples per ring and a comparable number of polar rings (Hansen 1988), so $360 \times 179$ is far above the bandlimit. Oversampling buys numerical headroom for the discrete inner products of §1.3 and matches the angular resolution of the held-out real-antenna measurement set described in §2.2.

## 1.3. Analytical decomposition: inner products with basis modes on the sphere

Given a fully measured complex field $E_{UT}$, the multipole coefficients can be recovered exactly (up to the truncation order $L$ and the discretisation grid) by direct projection. The vector spherical harmonics are orthogonal under the standard $L^2$ inner product on $S^2$ with the area-weighted measure $\mathrm{d}\Omega = \sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi$:

\[
\langle\,\boldsymbol{\Psi}^X_{lm},\, \boldsymbol{\Psi}^{X'}_{l'm'}\,\rangle_{S^2}
\;\equiv\;\iint_{S^2} \boldsymbol{\Psi}^X_{lm}(\theta,\varphi) \cdot \overline{\boldsymbol{\Psi}^{X'}_{l'm'}(\theta,\varphi)}\;\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\varphi
\;=\;\delta_{XX'}\,\delta_{ll'}\,\delta_{mm'},
\]

where $X, X' \in \{E, M\}$ index the mode family, the dot denotes the $\mathbb{C}^2$ inner product of the two tangential components, and the modes are normalised so that the inner product yields unity. Orthogonality reduces the inverse problem to one inner product per coefficient:

\[
a^E_{lm} \;=\; \langle\,E_{UT},\, \boldsymbol{\Psi}^E_{lm}\,\rangle_{S^2}, \qquad
a^M_{lm} \;=\; \langle\,E_{UT},\, \boldsymbol{\Psi}^M_{lm}\,\rangle_{S^2}.
\]

To make the discrete realisation explicit, we record the field as a length-$N_{\mathrm{ang}}$ vector with $N_{\mathrm{ang}} = 360\cdot 179\cdot 2 = 128{,}880$ complex entries — one entry per angular sample per tangential component — and we write $\mathrm{vec}(E_{UT})\in\mathbb{C}^{128880}$ for that flattening. Each VSH mode $\boldsymbol{\Psi}^X_{lm}$, evaluated on the same grid and folded by the same flattening, is a vector $\boldsymbol\psi^X_{lm}\in\mathbb{C}^{128880}$. Stack the $2K = 510$ basis vectors as the rows of a fixed matrix
\[
\mathbf{B} \;\in\; \mathbb{C}^{2K \times N_{\mathrm{ang}}} \;=\; \mathbb{C}^{70 \times 128880},
\]
**folding the area weight $\mu(\theta) = \sin\theta\,\Delta\theta\,\Delta\varphi$ into each entry** — that is, the row corresponding to $\boldsymbol\psi^X_{lm}$ at angular sample $(\theta_i, \varphi_j)$ and tangential component $c$ stores $\mu(\theta_i)\,\overline{\Psi^{X,c}_{lm}(\theta_i,\varphi_j)}$, with $\Delta\theta = \Delta\varphi = \pi/180$. (We reserve the letter $\mu$ for the scalar area weight to keep $w$ free for the area-weighted-norm subscript $\|\cdot\|_w$ used in §1.8.) The recovery is then a single matrix–vector product:
\[
\hat{a} \;=\; \mathbf{B}\,\mathrm{vec}(E_{UT}) \;\in\; \mathbb{C}^{2K} \;=\; \mathbb{C}^{70},
\]
and the entries of $\hat a$ are the $a^E_{lm}$ stacked above the $a^M_{lm}$ in the $(l,m)$ ordering fixed by the §1.1 packing convention. The dimensions match by construction: $70 \times 128880$ times $128880 \times 1$ gives $70 \times 1$.

This is the gold-standard non-iterative inverse: deterministic, linear, and exact in the noise-free, fully-measured complex case — to within the quadrature error of the discrete area-weighted Riemann sum. At the $360 \times 179$ grid for bandlimit $L=5$ that residual is many orders of magnitude below typical real-measurement noise, so we treat the recovery as numerically exact for the purposes of this thesis. No prior is needed, no iteration is run, and the answer is read off in a single linear pass. We treat this decomposition as the reference against which any learned model in this thesis is judged: a learned model that performs worse than the analytic inverse on the fully-measured complex regime is, in this thesis, regarded as failing the easiest available diagnostic.

The catch — anticipating §1.4 — is the words *fully measured complex*. The inner product needs both magnitude and phase of $E_\theta$ and $E_\varphi$, at every angular sample. In the application setting that motivates this thesis, only one real scalar per angular sample is observed.

## 1.4. The practical bottleneck: power-only measurement and phase loss

Far-field antenna characterisation in industrial and laboratory practice typically yields the **power pattern**:

\[
P(\theta,\varphi) \;=\; |E_\theta(\theta,\varphi)|^2 \;+\; |E_\varphi(\theta,\varphi)|^2.
\]

The reasons are practical rather than fundamental. Coherent measurement of complex amplitudes requires phase-stable instrumentation, narrowband sources, and a controlled reference path; the systems that do this exist but are expensive, slow, and standard only in metrology laboratories (Hansen 1988, Chapter 1). Power detectors, by contrast, are cheap, broadband, and yield $P(\theta,\varphi)$ with no phase reference. Most measured data — and almost all data collected outside dedicated near-field/far-field ranges — is therefore *phaseless*: a single real scalar per direction.

The numerical consequence is severe. At each grid point the complex pair $(E_\theta, E_\varphi)$ carries four real numbers; the corresponding $P$ carries one. Globally, the discretised radiation pattern shrinks from
\[
2 \times 360\times 179\times 2 \;=\; 257{,}760 \quad\text{real numbers (complex field)}
\]
to
\[
360\times 179 \;=\; 64{,}440 \quad\text{real numbers (power pattern)},
\]
a four-to-one collapse per angular sample, three-quarters of the angular information lost before the inverse problem even begins. The analytic decomposition of §1.3 is unavailable: the inner product $\langle E_{UT}, \boldsymbol{\Psi}^X_{lm}\rangle_{S^2}$ requires the complex field, not its squared modulus, and there is no single $E_{UT}$ to project onto the basis — every complex field whose dual-polarisation power matches $P$ is consistent with the measurement. The map $P \to (a^E, a^M)$ is therefore not merely harder than $E_{UT} \to (a^E, a^M)$: it is a structurally different problem, defined on a different input space, with a different and far less benign mathematical character. Section 1.5 makes this difference formal.

## 1.5. Formal statement: from the power pattern to the multipole coefficients

The inverse problem of this thesis is

\[
P_{UT} \;\longrightarrow\; (a^E, a^M),
\]

where the input $P_{UT}\in\mathbb{R}_{\ge 0}^{360\times 179}$ is the dual-polarisation power pattern of §1.4 on the angular grid of §1.2, and the target $(a^E, a^M)\in\mathbb{C}^{K}\times\mathbb{C}^{K}$ is the pair of multipole-coefficient vectors at $L=5$, $K=35$. Packed in the order fixed at the end of §1.1, the target lives in $\mathbb{R}^{4K}=\mathbb{R}^{140}$.

The forward problem — generating $P_{UT}$ from $(a^E, a^M)$ — admits a clean two-step factorisation. Let $\mathcal{S}$ denote the **synthesis operator**, the linear map that builds the complex field from its multipole coefficients via the truncated VSH expansion of §1.1:

\[
\mathcal{S}: (a^E, a^M)\;\longmapsto\; E_{UT}, \qquad
E_{UT}(\theta,\varphi) \;=\; \sum_{l=1}^{L}\sum_{m=-l}^{l}\bigl[a^E_{lm}\boldsymbol{\Psi}^E_{lm} + a^M_{lm}\boldsymbol{\Psi}^M_{lm}\bigr](\theta,\varphi).
\]

Let $|\cdot|^2$ denote the pointwise dual-polarisation power map:

\[
|\cdot|^2: \;E_{UT}(\theta,\varphi) = (E_\theta, E_\varphi)\;\longmapsto\;P(\theta,\varphi) \;=\; |E_\theta|^2 + |E_\varphi|^2.
\]

The composite **forward operator** is
\[
\mathcal{A} \;=\; |\cdot|^2 \circ \mathcal{S}\colon \;(a^E, a^M)\;\longmapsto\;P,
\]
and the inverse problem of this thesis is the construction of an $\mathcal{A}^{-1}$, or — when no such inverse exists in the usual sense — a sensible substitute for it.

The factorisation $\mathcal{A} = |\cdot|^2 \circ \mathcal{S}$ already exposes the difficulty. The synthesis operator $\mathcal{S}$ is linear, and on the angular grid of §1.2 its inversion is exactly the analytic decomposition of §1.3: by VSH orthogonality, $\mathcal{S}^{-1}$ exists and is the matrix $\mathbf B$ of §1.3 acting on the field. The pointwise power map $|\cdot|^2$, by contrast, is nonlinear and intrinsically many-to-one — at every angular sample it sends a complex pair $(E_\theta, E_\varphi)\in\mathbb{C}^2 \cong \mathbb R^4$ to its squared norm, collapsing four real degrees of freedom to one. Composition with $\mathcal{S}$ does not heal this collapse; the well-behaved linear factor cannot rescue the badly-behaved nonlinear one. Whether $\mathcal{A}$ as a whole is many-to-one — and if so, by how much — is exactly the question §§1.6–1.7 take up.

What can still be solved, regardless of how that question is answered, is the *regularised* problem: pick a single physically reasonable preimage out of the (possibly many) preimages that the forward operator admits. Sections 1.6 and 1.7 quantify "many"; §1.8 specifies what we mean by "physically reasonable" and how a learned model supplies it.

## 1.6. Information collapse from $E$ to $P$

The non-injectivity of $\mathcal{A}$ enters entirely through the pointwise power map $|\cdot|^2$. We therefore examine the $E\to P$ step in isolation, treating $E_{UT}$ as a free element of the synthesised-field space and asking: how many distinct $E_{UT}$'s give the same $P$?

### Per-sample collapse

At a single angular sample, the complex field carries the four real numbers
\[
\bigl(\mathrm{Re}\,E_\theta,\,\mathrm{Im}\,E_\theta,\,\mathrm{Re}\,E_\varphi,\,\mathrm{Im}\,E_\varphi\bigr)\in\mathbb{R}^4,
\]
and the power
\[
P \;=\; (\mathrm{Re}\,E_\theta)^2 + (\mathrm{Im}\,E_\theta)^2 + (\mathrm{Re}\,E_\varphi)^2 + (\mathrm{Im}\,E_\varphi)^2
\]
is the squared Euclidean norm of that 4-vector. The preimage of any value $P>0$ under this map is a 3-sphere $S^3$ of radius $\sqrt{P}$ in $\mathbb{R}^4$ — a continuous, three-parameter family of complex pairs, all consistent with the same observed power. Locally, the map is a *four-to-one collapse* in the dimensional sense.

### Global degree-of-freedom count

Aggregated over the $360\times 179$ angular grid, the complex field carries $\dim_\mathbb{R}(E_{UT}) = 4\times 360\times 179 = 257{,}760$ real degrees of freedom (cf. §1.2), while the power pattern carries only $\dim_\mathbb{R}(P) = 360\times 179 = 64{,}440$. Three quarters of the angular information — $193{,}320$ real numbers per pattern — has no representation in $P$.

A note on what this number does, and does not, imply. The forward operator $\mathcal S$ maps a $4K = 1020$-dimensional coefficient space into the $257{,}760$-dimensional field space, so the image of $\mathcal S$ is a low-dimensional submanifold of field space, *not* all of it. The $E\to P$ collapse therefore acts on a much smaller object than the full $4\times 360\times 179$ ambient; whether composing with $\mathcal S$ retains any non-injectivity *beyond global phase* is a genuine question, and §1.7 answers it (yes — at minimum by one further trivial ambiguity, the reflected-conjugate one). The dimension count of this paragraph establishes only that the field-space collapse is severe — it does not by itself prove anything about the multipole inverse.

### Two qualitatively different sources of degeneracy

Within the discarded information, two distinct kinds of ambiguity can be named at the conceptual level; §1.7 promotes the second from "named" to "argued".

The first is the **trivial global-phase symmetry**. For any global phase $\alpha\in\mathbb{R}$, multiplying the entire field by $\mathrm{e}^{i\alpha}$ leaves $P$ unchanged:
\[
P\bigl[\mathrm{e}^{i\alpha} E\bigr] \;=\; |\mathrm{e}^{i\alpha}E_\theta|^2 + |\mathrm{e}^{i\alpha}E_\varphi|^2 \;=\; |E_\theta|^2 + |E_\varphi|^2 \;=\; P[E].
\]
This $U(1)$ symmetry pulls back to coefficient space as a one-parameter orbit: $(a^E, a^M)$ and $(\mathrm{e}^{i\alpha}a^E, \mathrm{e}^{i\alpha}a^M)$ produce identical power patterns for every $\alpha$. The degeneracy is harmless: it is a single global phase, it is physically meaningless (electromagnetic measurements outside an interferometer are insensitive to it), and any sensible recovery procedure may freely fix it by convention.

The second is **further coefficient-space ambiguity beyond global phase**. The power map mixes contributions from different modes pointwise:
\[
P(\theta,\varphi) \;=\; \Bigl|\textstyle\sum_{l,m,X} a^X_{lm}\Psi^{X,\theta}_{lm}\Bigr|^2 + \Bigl|\textstyle\sum_{l,m,X} a^X_{lm}\Psi^{X,\varphi}_{lm}\Bigr|^2,
\]
which expands into a sum of cross-terms $a^X_{lm}\overline{a^{X'}_{l'm'}}$ across all mode pairs. This is the same algebraic structure as classical phase retrieval (Shechtman et al., *IEEE Signal Processing Magazine*, 2015; Fienup, *Applied Optics*, 1982): the squared modulus of a band-limited expansion. Phase retrieval over the Fourier basis is well known to admit, beyond the global-phase ambiguity, two further trivial ambiguities (conjugate inversion and spatial shift) and, in 1-D, additional non-trivial ones; in higher dimensions the Fourier case is generically unique up to the trivial set. The spherical-harmonic case has its own *partial* classification — at minimum one further trivial ambiguity, plus an explicit open question about further non-trivial ambiguities — summarised in §1.7. We do not transplant the Fourier classification onto the spherical case; we cite the spherical result directly.

## 1.7. Strict non-uniqueness: an infinite preimage from at least two distinct trivial ambiguities

We now turn the conceptual statements of §1.6 into precise statements about the inverse problem.

### Hadamard well-posedness

Following Hadamard's classical formulation, an inverse problem is *well-posed* when its solution exists, is unique, and depends continuously on the data; otherwise it is *ill-posed*. The three conditions are independent: a problem may admit a solution but not a unique one (uniqueness violation), or a unique solution that depends pathologically on the data (continuity violation). Both kinds of ill-posedness require regularisation, but for different reasons. The treatise of Arridge, Maass, Öktem and Schönlieb (*Acta Numerica*, vol. 28, pp. 1–174, 2019, doi:10.1017/S0962492919000059) gives the canonical modern taxonomy of data-driven approaches to ill-posed inverse problems and is the reference we use throughout for the data-driven side of the story (§1.8).

### The phaseless multipole inverse fails uniqueness — at least twice

The forward operator $\mathcal{A} = |\cdot|^2 \circ \mathcal{S}$ of §1.5 has *at least two distinct trivial ambiguities*, both established directly for spherical-harmonic phaseless inversion in the literature.

**(i) Global phase $U(1)$.** As shown in §1.6, every solution $(a^E, a^M)$ of $\mathcal{A}(a^E, a^M) = P_{UT}$ generates an entire one-parameter family $\{(\mathrm{e}^{i\alpha}a^E, \mathrm{e}^{i\alpha}a^M) : \alpha\in[0,2\pi)\}$ of solutions producing the same $P_{UT}$.

**(ii) Reflected-conjugate ambiguity.** Bangun (PhD dissertation, RWTH Aachen, 2020, "Signal recovery on the sphere from compressive and phaseless measurements", doi:10.18154/RWTH-2020-03041, Chapter 6, §6.2) establishes that scalar complex spherical-harmonic phaseless measurements admit a second trivial ambiguity beyond global phase: the coefficient vector
\[
\hat g_l^{\,k} \;=\; (-1)^k\,\overline{\hat f_l^{\,-k}}
\]
yields the same phaseless measurements as $\hat f_l^{\,k}$. This is a direct consequence of the conjugation property of complex scalar spherical harmonics, $Y_l^{k}(\theta,\varphi) = (-1)^k\,\overline{Y_l^{-k}(\theta,\varphi)}$.

The argument lifts to *vector* spherical harmonics by the same identity, with one extra step. The Jackson-convention VSH (Ch. 9) are built from the scalar $Y_{lm}$ via the angular-momentum operator $\mathbf L = -i\,\mathbf r \times \nabla$ and the radial-cross-product operator $\hat{\mathbf r}\times$; both are real differential operators on $S^2$ in the sense that they have real coefficients in any Cartesian chart, and therefore they commute with complex conjugation: $\overline{\mathbf L Y_{lm}} = \mathbf L\,\overline{Y_{lm}}$ and likewise for $\hat{\mathbf r}\times$. Composed with the scalar identity $Y_l^k = (-1)^k\,\overline{Y_l^{-k}}$, this yields the VSH conjugation rule
\[
\overline{\boldsymbol{\Psi}^X_{lm}(\theta,\varphi)} \;=\; (-1)^m\,\boldsymbol{\Psi}^X_{l,-m}(\theta,\varphi),
\qquad X\in\{E,M\}.
\]
Now define the **reflected-conjugate coefficient vector** by $a'^{\,X}_{lm} = (-1)^m\,\overline{a^X_{l,-m}}$ for $X \in \{E, M\}$, and compute the field it synthesises:
\[
E[a'](\theta,\varphi) \;=\; \sum_{l,m,X} a'^{\,X}_{lm}\,\boldsymbol\Psi^X_{lm}
\;=\; \sum_{l,m,X} (-1)^m\,\overline{a^X_{l,-m}}\,\boldsymbol\Psi^X_{lm}
\;=\; \overline{E[a]}(\theta,\varphi),
\]
where the last equality follows by re-indexing $m \to -m$ in the second sum and applying the VSH conjugation rule once. Hence $|E[a']|^2 = |\overline{E[a]}|^2 = |E[a]|^2$ pointwise on the sphere, i.e. $\mathcal A(a') = \mathcal A(a)$, so the reflected-conjugate transformation is a strict ambiguity of the VSH-power inverse $\mathcal A$ used in this thesis. Quotienting out the global-phase orbit therefore *does not* restore uniqueness: a second discrete reflection ambiguity remains, and the preimage $\mathcal{A}^{-1}(P_{UT})$ contains *at least two* topologically distinct components for every $P_{UT}$ in the image of $\mathcal{A}$.

**(iii) Sampling-induced ambiguity.** Bangun 2020, Proposition 6.2.1, exhibits explicit sampling patterns for which the spherical-harmonic sensing matrix becomes rank-deficient and additional, sampling-pattern-dependent ambiguities appear. Our $360\times 179$ uniform grid (§1.2) is not one of those patterns, so this class is not active in our setup; we mention it for completeness and as a reminder that ambiguity analysis is sensitive to grid choice.

**(iv) Open class: further non-trivial ambiguities.** Whether the phaseless VSH inverse admits, beyond (i)–(iii), additional discrete or continuous non-trivial ambiguities is, to our knowledge, an open question. The Fourier-domain analogue (Shechtman et al. 2015) admits non-trivial ambiguities in 1-D and is generically unique up to the trivial set in higher dimensions; whether the spherical case behaves more like the 1-D Fourier case (rich non-trivial ambiguities) or the higher-dimensional one (generic uniqueness modulo trivial) is not settled by the references we use. We treat this as a working hypothesis to be probed empirically in later chapters: if non-trivial ambiguities are abundant, a learned regulariser must select among them; if they are rare, the trivial ambiguities (i)–(ii) are enough to motivate the regulariser already.

### Verdict

The phaseless multipole inverse problem is, in the Hadamard sense, **ill-posed by uniqueness violation** — at minimum by the global-phase orbit and by the reflected-conjugate reflection, and possibly more. This is a structural fact about the problem, not a deficiency of any particular numerical method. No algorithm — analytic, iterative, or learned — can produce *the* preimage of $P_{UT}$, because there is no such preimage. The most that can be asked is that the algorithm produces *one* preimage, selected by an externally imposed criterion. The next section identifies that criterion.

## 1.8. Machine learning as regularisation: a learned prior on realistic coefficients

### The variational template

Throughout this section we identify the forward operator $\mathcal A$ of §1.5 with its real-vector form via the §1.1 packing: a vector $a \in \mathbb{R}^{4K}$ unpacks to $(a^E, a^M) \in \mathbb{C}^K \times \mathbb{C}^K$ in the order $[\mathrm{Re}\,a^E, \mathrm{Im}\,a^E, \mathrm{Re}\,a^M, \mathrm{Im}\,a^M]$, so that $\mathcal A(a)$ and $\mathcal A(a^E, a^M)$ denote the same mapping written in two notations.

The classical regularised solution of an ill-posed inverse problem is the minimiser of a sum of two terms — a *data-fidelity* term that pulls the solution toward the measurement, and a *regulariser* that pulls it toward the prior:

\[
\hat a \;=\;\arg\min_{a\in\mathbb{R}^{4K}} \;\;\underbrace{\bigl\|\mathcal{A}(a) \;-\; P_{UT}\bigr\|_{w}^{2}}_{\text{data fidelity}} \;+\; \lambda\,\underbrace{R(a)}_{\text{regulariser}}.
\]

The norm $\|\cdot\|_w$ is the spherical-area-weighted $L^2$ norm on the $(\theta,\varphi)$ grid — the subscript $w$ stands for "weighted" and the weight is the same area weight $\mu(\theta) = \sin\theta\,\Delta\theta\,\Delta\varphi$ that appeared inside the matrix $\mathbf B$ of §1.3. The regulariser $R$ encodes whatever prior information distinguishes the *desired* preimage from the rest of $\mathcal{A}^{-1}(P_{UT})$. The hyperparameter $\lambda$ trades fidelity against regularity. This template covers Tikhonov regularisation ($R(a) = \|a\|_2^2$), sparsity-promoting regularisation ($R(a) = \|a\|_1$), total-variation regularisation, and many others; it is the same template under which most classical inverse-problem solvers are derived (Arridge et al. 2019, §2.4–2.7).

### Why we expect the classical templates not to fit

Each classical hand-designed regulariser encodes an explicit prior over coefficient space, and each of those priors has a clear failure mode in the multipole setting at the conceptual level:

- **Tikhonov $\ell_2$ shrinkage** drives all coefficients toward zero with a single global magnitude scale, treating low-order and high-order modes as exchangeable. Real radiating structures generically do not have a uniform coefficient magnitude across $l$; whether the actual training distribution we use exhibits the heterogeneity that the chapter's argument assumes is an empirical question we defer to Chapter 2 (data and generation).
- **Sparsity** ($\ell_1$, $\ell_0$) promotes a few large coefficients with the rest exactly zero. Whether realistic radiation patterns are sparse in the VSH basis is, again, an empirical question deferred to Chapter 2.
- **Total variation** is defined for signals on a domain with a notion of neighbouring indices. The multipole index $(l,m)$ has no canonical such structure, so TV does not have a natural definition in coefficient space.

The above are *expectations*, not theorems. Whether the empirical coefficient distribution in this thesis genuinely defeats Tikhonov and sparsity is decided by the experiments of Chapter 7, where a Tikhonov-style $\ell_2$ baseline is compared head-to-head with the learned models of Chapters 4–5. The role of §1.8 is only to motivate trying a learned regulariser; the verdict belongs to the experimental chapters.

### Learned regularisation

The data-driven response to the mismatch between hand-designed regularisers and an unknown physical prior is to **learn the regulariser from examples**. Given a training set $\{(a^E_n, a^M_n, P_n)\}_{n=1}^N$ of physically plausible coefficient vectors and the power patterns they generate, a parametric model
\[
f_\eta \colon \mathbb{R}^{64{,}440} \to \mathbb{R}^{4K},\qquad \eta\in\mathbb{R}^{d_\eta},
\]
with parameter vector $\eta$ (we use $\eta$ rather than the conventional $\theta$ to avoid collision with the polar angle $\theta$, and we reserve the letter $w$ for the area-weighted-norm subscript $\|\cdot\|_w$ above) can be trained to map the power pattern directly to a preimage:
\[
f_\eta(P_{UT}) \;\approx\; (a^E, a^M)\;\in\;\mathcal{A}^{-1}(P_{UT}).
\]
The network does not, and cannot, recover *the* preimage — there is no such object — but it can be trained to consistently select the preimage closest to those in the training distribution. The training distribution itself is the regulariser: it specifies, implicitly and at high resolution, what counts as a realistic $(a^E, a^M)$. Where the classical Tikhonov regulariser is an explicit function of $a$, the learned regulariser is an implicit prior — the support and density of the empirical coefficient distribution — encoded in the parameters $\eta$.

### Where this approach sits in the Arridge et al. 2019 taxonomy

Arridge, Maass, Öktem and Schönlieb (2019) organise data-driven methods for inverse problems along several axes; the categories relevant here are:

- **Fully learned Bayes estimation** (Arridge et al. 2019, §5.1.3): the reconstruction operator $f_\eta \colon Y \to X$ has a generic parametrisation that does not include the forward operator $\mathcal A$ as an explicit component; the network learns the inverse end-to-end from supervised pairs $(P, a)$. Advantages: simplicity, no need for a differentiable forward operator at inference time, no inner loop. Disadvantages noted by Arridge et al.: poor scaling in very high-dimensional problems (many weights to learn) and limited transferability when the measurement instrument changes.
- **Learned post-processing** (§5.1.5): apply a knowledge-driven inversion first, then learn a regulariser/denoiser that polishes the result. This requires an analytic inverse to start from, which the phaseless setting of this thesis does not have.
- **Learned iterative schemes** (§5.1.4): unroll an iterative solver and train its components, embedding $\mathcal A$ and its adjoint inside the network.
- **Plug-and-play / black-box denoiser** (§4.6): swap a learned denoiser into a classical iterative solver as the proximal step of a regulariser.

The approach pursued in this thesis sits in the **fully learned** family of §5.1.3: a feed-forward network is trained to map the power pattern directly to the coefficient vector, with no inner optimisation loop and no explicit use of $\mathcal A$ at inference time. We adopt this because it is the simplest baseline and because it is the most operationally convenient given an abundant synthetic training distribution and a fast, differentiable forward operator (used at training time, not inference time, and only when the loss is the physics-aware loss of Chapter 5). Plug-and-play and unrolled-iteration variants are natural alternatives that we do not pursue here.

### Thesis statement

The remainder of this thesis is the empirical study of one such fully learned regulariser. We build a synthetic training distribution over $(a^E, a^M)$ that approximates the population of realistic radiators and choose feature representations of the input power pattern that preserve as much of the salvageable angular information as possible (Chapter 2); train a family of neural-network architectures spanning the design space from linear regression to physics-aware deep models (Chapter 4); compare two training objectives — direct coefficient regression versus a physics-informed loss evaluated through a differentiable forward operator (Chapter 5); evaluate the resulting models against a battery of quality metrics on held-out synthetic samples and real-antenna data (Chapters 6–7); and conclude with what the experiments tell us about how well a learned prior fills the information gap left by phaseless measurement.

The organising question of the thesis is, accordingly:

> *Given the measured power pattern of a radiator, how accurately and how robustly can a learned regulariser recover its multipole coefficients?*

Sections 1.1–1.7 provided the physical objects and the formal problem statement. This section provided the regularisation framework. Everything that follows is the construction, training, and evaluation of an answer.

---

## References used in this chapter

- J. D. Jackson, *Classical Electrodynamics*, 3rd edition, Wiley, 1999. Chapter 9 ("Radiating Systems, Multipole Fields and Radiation").
- J. E. Hansen (ed.), *Spherical Near-Field Antenna Measurements*, IEE Electromagnetic Waves Series 26, Peter Peregrinus / IEE, London, 1988.
- Y. Shechtman, Y. C. Eldar, O. Cohen, H. N. Chapman, J. Miao, M. Segev, "Phase retrieval with application to optical imaging: a contemporary overview", *IEEE Signal Processing Magazine*, vol. 32, no. 3, pp. 87–109, 2015. doi:10.1109/MSP.2014.2352673.
- J. R. Fienup, "Phase retrieval algorithms: a comparison", *Applied Optics*, vol. 21, no. 15, pp. 2758–2769, 1982.
- A. Bangun, "Signal recovery on the sphere from compressive and phaseless measurements", PhD dissertation, RWTH Aachen University, 2020. doi:10.18154/RWTH-2020-03041.
- S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb, "Solving inverse problems using data-driven models", *Acta Numerica*, vol. 28, pp. 1–174, 2019. doi:10.1017/S0962492919000059.

<!-- Sources backed by research/ch1-enrichment/manifest.md (R1–R6). -->

