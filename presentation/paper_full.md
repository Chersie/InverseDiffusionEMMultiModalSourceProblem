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

---

# Chapter 2. Training-data generation. Feature representation and preprocessing

This chapter specifies the data pipeline and the featurisation pipeline on which every learned regulariser in the thesis is trained. Sections 2.1–2.2 specify the data: §2.1 the parametric synthetic generator and its augmentation primitives, §2.2 the real-antenna corpus used for validation and held-out evaluation. Sections 2.3–2.6 specify the featurisation: §2.3 the three input modes, §2.4 the randomised-PCA reduction, §2.5 two structured angular descriptors, §2.6 the normalisation conventions. The experiment-specific integer counts and the choice of feature stack are configured per experiment and reported in the experimental tables of Chapter 7.

## 2.1. Synthetic data generator

The synthetic generator draws a packed coefficient vector $a \in \mathbb{R}^{4K}$ from a parametric distribution and applies the analytic forward operator $\mathcal S$ of §1.3 followed by the pointwise modulus-squared of §1.4:

\[
(a^E, a^M)\;\xrightarrow{\;\mathcal S\;}\;E_{UT}\;=\;\sum_{l=1}^{L}\sum_{m=-l}^{l}\bigl[a^E_{lm}\boldsymbol{\Psi}^E_{lm} + a^M_{lm}\boldsymbol{\Psi}^M_{lm}\bigr]\;\xrightarrow{\;|\cdot|^2\;}\;P\;=\;|E_\theta|^2+|E_\varphi|^2.
\]

The randomness for every batch is drawn from an explicitly seeded `numpy.random.default_rng`, so the output is reproducible. The only design freedom is the distribution from which $a$ is drawn, described next.

### Coefficient-distribution modes

Four modes are available, each defining a parametric family of distributions over $(a^E, a^M)$. Below, $K = L(L+2)$ is the number of $(l,m)$ pairs per family at truncation $L$, and draws are taken independently per sample, per family, and per coefficient unless stated otherwise.

**Mode 1 — `gaussian`**:
\[
a^E_{lm},\, a^M_{lm}\;\sim\;\frac{1}{\sqrt 2}\bigl(\mathcal N(0,1) + i\,\mathcal N(0,1)\bigr).
\]
The $1/\sqrt 2$ scaling fixes $\mathbb E|a^X_{lm}|^2 = 1$. Mode 1 encodes the maximum-entropy order-agnostic prior at fixed second moment and is the reference distribution against which the other three are compared.

**Mode 2 — `uniform`**:
\[
\mathrm{Re}\,a^X_{lm},\, \mathrm{Im}\,a^X_{lm}\;\sim\;\mathcal U([-1, 1])\quad\text{independently}.
\]
Mode 2 is the order-agnostic heavy-tail-free baseline; the bounded support guarantees no extreme samples and isolates amplitude-scale sensitivity from tail behaviour.

**Mode 3 — `colored`** (per-degree decay):
\[
a^E_{lm},\, a^M_{lm}\;\sim\;(l+1)^{-\alpha}\cdot\frac{1}{\sqrt 2}\bigl(\mathcal N(0,1) + i\,\mathcal N(0,1)\bigr),
\]
with default colour exponent $\alpha = 1$. Mode 3 encodes a low-order-favouring prior: the per-degree scale $(l+1)^{-\alpha}$ pushes mass toward the dipole/quadrupole sector, in qualitative agreement with the §1.8 expectation that realistic radiators concentrate angular energy in the lowest multipoles. $\alpha = 0$ recovers Mode 1.

**Mode 4 — `sparse`** (Bernoulli active-mode mask):
\[
m^X_{lm}\sim\mathrm{Bern}(p_\mathrm{active}),\qquad a^X_{lm} = m^X_{lm}\cdot\bigl(\mathcal N(0,1) + i\,\mathcal N(0,1)\bigr),
\]
with default active-mode fraction $p_\mathrm{active} = 0.1$. Mode 4 encodes a sparse-support prior for radiators dominated by a small number of strongly-excited multipoles. Note that the active branch uses the un-normalised complex Gaussian — *no* $1/\sqrt 2$ — so $\mathbb E|a^X_{lm}|^2 = 2$ conditional on $m^X_{lm} = 1$ and $2\,p_\mathrm{active}$ unconditionally; the factor-of-two relative to Mode 1 is a deliberate convention break in the generator.

The empirical question of which mode trains the best learned regulariser is settled in Chapter 7.

### Composable knobs

Three knobs act on top of any choice of mode and may be combined freely.

**Family balance.** A scalar $b \in [0, 1]$ rescales the two families before packing:
\[
a^E \leftarrow 2(1-b)\cdot a^E,\qquad a^M \leftarrow 2b\cdot a^M.
\]
The factor of $2$ is the normalisation choice that makes $b = 0.5$ identity-preserving ($2(1-0.5) = 2\cdot 0.5 = 1$): other natural choices, such as a convex combination $(1-b)\,a^E + b\,a^M$ or a unit-energy renormalisation, would have shifted the per-family amplitude scale at the default and required a corresponding rescaling of the per-mode amplitudes. $b\to 0$ gives a purely electric radiator, $b\to 1$ a purely magnetic one. Optional per-sample uniform jitter is supported.

**Per-sample log-uniform amplitude scale.** An optional knob multiplies each sample by $s$ drawn log-uniformly:
\[
\log s\;\sim\;\mathcal U([\log s_{\min},\, \log s_{\max}]),\qquad a \leftarrow s\cdot a,
\]
exposing the network to many decades of absolute input magnitude in a single training run.

**Post-hoc mode dropout.** Independently per sample, per family, per $(l,m)$, with drop probability $p_\mathrm{drop}$ a coefficient is set to zero:
\[
a^X_{lm} \leftarrow d^X_{lm}\cdot a^X_{lm},\qquad d^X_{lm}\sim\mathrm{Bern}(1 - p_\mathrm{drop}).
\]
The Bernoulli mask $d^X_{lm}$ takes the value $1$ with the *keep* probability $1 - p_\mathrm{drop}$ (the relation to Mode 4's keep probability is $p_\mathrm{drop} = 1 - p_\mathrm{active}$ if a one-to-one match between knob and mode is desired). Unlike Mode 4, the knob acts on already-sampled coefficients drawn from *any* mode, so a non-sparse mode (`gaussian`, `uniform`, `colored`) can be combined with a sparsified support pattern without changing the underlying amplitude distribution.

### Augmentation primitives

Five augmentation primitives act on a sample $(P, a)$. Two of the five (Primitives 1 and 5) are strict symmetries of $\mathcal A$ that leave the pair on the same orbit of an exact group action and carry zero label noise. The other three (Primitives 2, 3, 4) are physically motivated perturbations with default parameters $\sigma_a = 0.05$, $p_\mathrm{drop} = 0.1$, $\sigma_P = 0.02$, revisited in Chapter 7.

**Primitive 1 — global phase rotation.** For $\alpha\sim\mathcal U[0,2\pi)$,
\[
a^X_{lm}\;\leftarrow\;e^{i\alpha}\,a^X_{lm},\qquad P\;\text{unchanged}.
\]
The $U(1)$ symmetry of §1.6 ($|e^{i\alpha} E|^2 = |E|^2$) makes $P$ invariant; only the coefficients change. The orbit covers the global-phase ambiguity of §1.7(i).

**Primitive 2 — coefficient additive noise.** For $\sigma_a > 0$ and $\varepsilon = (\varepsilon_R + i\varepsilon_I)/\sqrt 2$ with $\varepsilon_R, \varepsilon_I \sim \mathcal N(0, 1)$ independent (so $\mathbb E|\varepsilon|^2 = 1$, the same convention as Mode 1),
\[
a^X_{lm}\;\leftarrow\;a^X_{lm} + \sigma_a\,\varepsilon,\qquad P\;\text{re-synthesised from the perturbed coefficients}.
\]
Re-synthesis through $\mathcal S$ keeps $(P, a)$ on the image of $\mathcal A$. The purpose is local smoothing of the coefficient-space training distribution.

**Primitive 3 — coefficient mode dropout.** With drop probability $p_\mathrm{drop} \in [0, 1]$,
\[
a^X_{lm}\;\leftarrow\;d^X_{lm}\,a^X_{lm},\qquad d^X_{lm}\sim\mathrm{Bern}(1 - p_\mathrm{drop}),
\]
followed by re-synthesis of $P$. This is the same Bernoulli mask as the post-hoc mode dropout knob of the previous subsection, called as a per-batch augmentation primitive rather than as a sampling-time step. The primitive injects support-pattern variability and applies equally to synthetic and real-antenna samples.

**Primitive 4 — field additive noise (noisy-input, clean-target).** With relative noise scale $\sigma_P > 0$ and $\varepsilon$ an i.i.d. real standard normal,
\[
P_{i}\;\leftarrow\;\max\!\bigl(P_{i} + \sigma_P\cdot\max_{(\theta,\varphi)} P_{i}\cdot\varepsilon,\;0\bigr),\qquad a\;\text{unchanged}.
\]
The per-sample maximum of $P$ scales the noise so that $\sigma_P$ describes a consistent signal-to-noise ratio across samples. Targets are not updated: the training pair is deliberately inconsistent, on the *noisy-input, clean-target* contract. The broader robustness argument of Lerma Pineda and Petersen (arXiv:2206.00934, 2022) motivates this contract.

**Primitive 5 — azimuthal roll.** For a random integer shift $k\in\{0,\dots,n_\phi-1\}$ and $\varphi_k = 2\pi k / n_\phi$,
\[
P(\theta, \varphi)\;\leftarrow\;P(\theta, \varphi - \varphi_k),\qquad a^X_{lm}\;\leftarrow\;e^{-im\varphi_k}\,a^X_{lm}.
\]
The transformation is an exact $\mathrm{SO}(2)$ *equivariance* of $\mathcal A$ — input and target transform jointly, so it is not a strict ambiguity in the sense of §1.7's catalogue (which listed only the global $U(1)$ phase and the reflected-conjugate reflection) and does not extend that catalogue. The lift to the *vector* spherical-harmonic basis is direct: $\boldsymbol\Psi^X_{lm}$ depends on the azimuthal coordinate only through the scalar factor $Y_{lm}(\theta,\varphi)\propto e^{im\varphi}$, with $\theta$-dependent tangential components unchanged, so
\[
\boldsymbol\Psi^X_{lm}(\theta,\,\varphi - \varphi_k)\;=\;e^{-im\varphi_k}\,\boldsymbol\Psi^X_{lm}(\theta,\varphi).
\]
Substituting into the synthesis sum gives $E(\theta,\varphi-\varphi_k) = \sum_{lm}(e^{-im\varphi_k}\,a^X_{lm})\,\boldsymbol\Psi^X_{lm}(\theta,\varphi)$, and squaring yields $P(\theta,\varphi-\varphi_k)$ from the phase-rotated coefficients exactly. The discrete realisation is consistent: because the $\varphi$-axis grid is $\varphi_j = j\cdot 2\pi/n_\phi$ for $j = 0,\dots,n_\phi-1$ (the open-interval convention used by the loader), an integer roll by $k$ maps $\varphi_j \to \varphi_{(j+k)\bmod n_\phi} = \varphi_j + \varphi_k$ exactly, so the integer-roll operation on $P$ and the coefficient-domain phase rotation $a^X_{lm}\mapsto e^{-im\varphi_k}a^X_{lm}$ produce the same $(P,a)$ pair pointwise on the grid.

## 2.2. Real data used in training and validation

### File format and the analytic decomposition at $L = 5$

Each real antenna is delivered as a pair of files. The **field file** records the far-field on the same uniform 1° angular grid as §1.2, with seven real columns per grid sample:

\[
\theta_{\mathrm{deg}},\quad \varphi_{\mathrm{deg}},\quad P,\quad |E_\theta|,\quad \arg(E_\theta)_{\mathrm{deg}},\quad |E_\varphi|,\quad \arg(E_\varphi)_{\mathrm{deg}}.
\]

Phases are stored in degrees and converted to radians once on load; thereafter every internal quantity is in radians. The complex tangential field is reconstructed pointwise as $E_\theta = |E_\theta|\,e^{i\arg E_\theta}$, similarly for $E_\varphi$, yielding an array of shape $(2, n_\theta, n_\phi) = (2, 179, 360)$ in single-precision complex floating-point (numpy `complex64`); the leading tangential-component axis is the in-memory convention of the loader and is the transpose of §1.2's $(360, 179, 2)$ ordering for the same field. The paired **coefficient file** lists rows $(\text{type},\,l,\,m,\,\mathrm{Re},\,\mathrm{Im})$ with $\text{type}\in\{E, M\}$, truncated at $L = 5$ on import and packed in the order of §1.1.

The coefficients are derived from the field by the discrete analytic decomposition of §1.3 applied at $L = 5$:

\[
\hat a^X_{lm}\;\approx\;\sum_{\theta_i,\varphi_j,c}\;\mu(\theta_i)\,E^c(\theta_i,\varphi_j)\,\overline{\Psi^{X,c}_{lm}(\theta_i,\varphi_j)},\qquad \mu(\theta_i) = \sin\theta_i\,\Delta\theta\,\Delta\varphi.
\]

### Bandlimit choice and projection residuals

Truncating at $L = 5$ implies that any angular content above the fifth multipole degree is discarded by construction; the post-truncation field $E_\mathrm{resyn}$ synthesised from $\hat a$ via $\mathcal S$ is bandlimited at $L = 5$, and the relative residuals

\[
\rho_P\;=\;\frac{\|P_\mathrm{meas} - P_\mathrm{resyn}\|_2}{\|P_\mathrm{meas}\|_2},\qquad \rho_E\;=\;\frac{\|E_\mathrm{meas} - E_\mathrm{resyn}\|_2}{\|E_\mathrm{meas}\|_2}
\]

quantify how much of the original measurement is lost. The residuals are computed over the full corpus of $396$ paired files at $L = 5$, on the same uniform 1° angular grid $(n_\theta, n_\phi) = (179, 360)$ used throughout the chapter; $\|\cdot\|_2$ denotes the flat $\ell^2$ norm over the angular grid (no area weight, since the quantity of interest is the per-sample relative reconstruction error rather than a spherical inner product). $P_\mathrm{meas}$ is the raw $|E_\theta|^2 + |E_\varphi|^2$ from the seven-column file, and $P_\mathrm{resyn} = |E_\mathrm{resyn}|^2$ is the squared modulus of the bandlimited field obtained by re-synthesising $\hat a$ through the §1.3 quadrature.

Across this corpus $\rho_P$ has mean $1.16\%$, median $0.36\%$, and maximum $8.08\%$; $\rho_E$ has mean $1.30\%$, median $0.32\%$, and maximum $10.05\%$. The typical antenna is well-described at $L = 5$; the heavier-tailed antennas retain a few percent of energy above the bandlimit that no $L = 5$-truncated model can reproduce in the resynthesised field. Every downstream training, validation, and held-out sample uses the bandlimited pair $(P_\mathrm{resyn}, \hat a)$.

### Split by sample identifier

The real corpus is partitioned by *sample identifier*, never by augmented copy, in two stages with two independent deterministic shuffle seeds. In the first stage the full list of paired files is shuffled with a fixed seed and the first $n_\mathrm{src}$ identifiers are taken as the **train-and-validation pool**; the first $n_\mathrm{tr}$ of those form the training set of real source antennas and the remaining $n_\mathrm{src} - n_\mathrm{tr}$ form the validation set. In the second stage the tail of the original shuffled list is reshuffled with a second, independent seed and its first $n_\mathrm{ho}$ identifiers form the **held-out** evaluation set. The integer counts $n_\mathrm{src}, n_\mathrm{tr}, n_\mathrm{ho}$, the augmented training count $N_\mathrm{aug}$, and the two seeds are configured per experiment and reported in the experimental tables of Chapter 7.

Three properties follow: validation identifiers and training identifiers are disjoint at the source level, so a single antenna never contributes both a training augmented copy and a validation example; the held-out identifiers are drawn from outside the train+validation pool with an independent shuffle, so the held-out set is statistically independent of the training-side shuffle; and only the training pool is augmented — validation and held-out sets are passed through the truncation step but otherwise left untouched.

### Augmentation chain on the training pool

The training pool is expanded from $n_\mathrm{tr}$ source antennas to $N_\mathrm{aug}$ augmented training samples by repeated application of three primitives from §2.1, composed in a fixed order:

\[
\text{Primitive 5 (azimuthal roll)}\;\to\;\text{Primitive 3 (coefficient mode dropout)}\;\to\;\text{Primitive 4 (field additive noise)}.
\]

For each augmented slot, a training-source identifier is drawn uniformly with replacement from the $n_\mathrm{tr}$ sources and the three primitives are applied in turn.

Primitives 1 and 2 are deliberately omitted from this chain. Primitive 1 (global phase) leaves the bandlimited real $P_\mathrm{resyn}$ pointwise invariant and only rotates the target $a$ by a uniform $U(1)$ phase, which produces no operationally new $(P, a)$ pair on top of Primitives 5/3/4 under the coefficient-MSE training of Chapter 5. Primitive 2 (coefficient additive noise) would re-synthesise $P$ from $a + \sigma_a\varepsilon$ and so densify the coefficient-space distribution, but the small real corpus already carries a measurement-side coefficient perturbation through the projection residuals of the previous subsection ($\rho_E$ mean $1.30\%$); adding a second perturbation would compound an effect already present in the data without expanding the operating envelope of the model.

The remaining order is fixed: Primitive 5 commutes with Primitive 3's dropout-and-resynthesise step up to integer-grid alignment (a phase $e^{-im\varphi_k}$ on coefficients followed by a Bernoulli mask factorises identically to the mask followed by the phase, because the mask is index-wise and the phase is per-mode), so applying Primitive 5 first or between is convention; Primitive 4 must come last because the noise it adds to $P$ has no corresponding update to $a$ and would be erased by the re-synthesis step inside Primitive 3 if placed earlier.

## 2.3. Input modes: power, magnitude, complex

Sections 2.1 and 2.2 produce, for every sample, a complex tangential field $E_{UT} \in \mathbb{C}^{2 \times n_\theta \times n_\phi}$ together with its packed coefficient vector $a \in \mathbb{R}^{4K}$. The model receives a channel selection derived from the field, called the *input mode*. Three modes are available:

\[
\begin{aligned}
\text{POWER:}\quad &x^\mathrm{pow}(\theta,\varphi) = |E_\theta|^2 + |E_\varphi|^2 \in \mathbb{R}, \quad C = 1,\\
\text{MAGNITUDE:}\quad &x^\mathrm{mag}(\theta,\varphi) = \bigl(|E_\theta|,\,|E_\varphi|\bigr) \in \mathbb{R}^2, \quad C = 2,\\
\text{COMPLEX:}\quad &x^\mathrm{cplx}(\theta,\varphi) = \bigl(\mathrm{Re}\,E_\theta,\,\mathrm{Im}\,E_\theta,\,\mathrm{Re}\,E_\varphi,\,\mathrm{Im}\,E_\varphi\bigr) \in \mathbb{R}^4, \quad C = 4.
\end{aligned}
\]

Aggregated over the $360 \times 179$ grid, the flat input vector has dimension

\[
\dim_\mathbb R\,x \;=\; C\cdot n_\theta\cdot n_\phi\;=\; \begin{cases} 64{,}440, & \text{POWER},\\ 128{,}880, & \text{MAGNITUDE},\\ 257{,}760, & \text{COMPLEX}.\end{cases}
\]

POWER is the operational measurement regime of §1.4 and the default for all production training. COMPLEX is the fully-measured regime of §1.3 on which the analytic inverse $\hat a = \mathbf B\,\mathrm{vec}(E_{UT})$ is available, included so that §1.3's analytic gold-standard can be exercised as a diagnostic on the fully-measured complex regime: a learned model that does worse than the analytic inverse on COMPLEX input fails the easiest available diagnostic. MAGNITUDE retains per-component amplitude but discards per-component phase, so the analytic inverse is unavailable on it as well, and the mode serves as a diagnostic intermediate between the other two. All three modes are well-defined for both the synthetic and the real-antenna sources.

## 2.4. Randomised PCA as a dimensionality-reduction stage

The flat POWER representation lives in $\mathbb{R}^{n_\mathrm{ang}}$ with $n_\mathrm{ang} = C\cdot n_\theta\cdot n_\phi = 1\cdot 179\cdot 360 = 64{,}440$. (Distinct from §1.3's $N_\mathrm{ang} = 2\cdot n_\theta\cdot n_\phi = 128{,}880$, which counted the flat length of the complex tangential field over both polarisation components: here $n_\mathrm{ang}$ is the channel-multiplied flat input width and depends on the §2.3 channel selection $C$.) Neighbouring $(\theta, \varphi)$ cells of a band-limited radiator carry almost the same numerical value, so the angular grid is highly redundant; a linear projection onto a compact principal-direction basis removes the leading correlations and produces a low-dimensional, decorrelated input for the downstream regressor.

Let $\boldsymbol\Phi \in \mathbb{R}^{N \times n_\mathrm{ang}}$ stack the flat input vectors of the $N$ training samples row-wise. After column-mean centring, $\boldsymbol\Phi_c = \boldsymbol\Phi - \mathbf{1}\hat\mu^\top$, the principal-component basis of rank $r$ is the matrix $V_r \in \mathbb{R}^{n_\mathrm{ang}\times r}$ whose columns are the leading $r$ right singular vectors of $\boldsymbol\Phi_c$. The projection

\[
Z\;=\;\boldsymbol\Phi_c\,V_r\;\in\;\mathbb{R}^{N\times r}
\]

is the reduced feature representation. The canonical configuration uses $r = 128$ components, a roughly $500{:}1$ compression on POWER input.

A deterministic full SVD costs $O(N\,n_\mathrm{ang}^2 + n_\mathrm{ang}^3)$ and is prohibitive at the project's scale. The randomised-SVD framework of Halko, Martinsson and Tropp (*SIAM Review* 53(2):217–288, 2011) sidesteps that cost: draw a Gaussian probe matrix $\Omega \in \mathbb{R}^{n_\mathrm{ang}\times (r + q)}$ with small oversampling $q$ (typically $q \in [5, 10]$), form the range sketch $Y = \boldsymbol\Phi_c\,\Omega$, orthonormalise it to obtain $Q$, and run a deterministic SVD on the much smaller matrix $Q^\top \boldsymbol\Phi_c$ to recover $V_r$. The cost is essentially linear in $n_\mathrm{ang}$ and the spectral-norm error matches the deterministic top-$r$ SVD to within a small multiplicative factor with overwhelming probability. The project uses the scikit-learn implementation (`sklearn.decomposition.PCA` with `svd_solver="randomized"`, Pedregosa et al. *JMLR* 12:2825–2830, 2011); centring is handled by the solver and $\hat\mu$ is computed on the training partition only.

The PCA basis is fitted on the training partition only; validation and held-out samples are transformed using the trained basis. For the MAGNITUDE and COMPLEX input modes of §2.3 the construction generalises without change, with $\boldsymbol\Phi$ widened to $C\cdot n_\theta\cdot n_\phi$ columns.

## 2.5. Additional computer-vision features

Two structured angular descriptors are computed alongside the PCA basis: a radial Fourier spectrum and a per-degree spherical-harmonic energy spectrum. Both are channel-aware and apply to any of the three input modes of §2.3.

### 2.5.1. Radial Fourier power spectrum

Let $f(\theta_i, \varphi_j)$ denote a single channel of the input on the angular grid, with $(\theta_i, \varphi_j)$ the grid points indexed by integers $i = 0, \dots, n_\theta - 1$ and $j = 0, \dots, n_\phi - 1$. The $(\theta, \varphi)$ grid is uniform in coordinates but not in solid angle, so a $\sin\theta$ pre-window approximates the solid-angle weight of §1.3 before the FFT (default on; both the `pca_cv` and `cv_only` feature pipelines apply it):

\[
\tilde f(\theta_i, \varphi_j)\;=\;\sin(\theta_i)\cdot f(\theta_i, \varphi_j).
\]

A two-dimensional discrete Fourier transform is then computed across both axes,

\[
\hat f(k_\theta, k_\varphi)\;=\;\sum_{i = 0}^{n_\theta - 1}\sum_{j = 0}^{n_\phi - 1}\tilde f(\theta_i, \varphi_j)\,e^{-i 2\pi(k_\theta\,i / n_\theta + k_\varphi\,j / n_\phi)},
\]

and shifted so that the zero-frequency mode is at the array centre. The spectrum is collapsed radially: the $(k_\theta, k_\varphi)$ plane is partitioned into $n_b$ equal-radius annuli $\{R_b\}_{b=0}^{n_b-1}$ and the descriptor records the mean amplitude in each annulus,

\[
f_b\;=\;\frac{1}{|R_b|}\sum_{(k_\theta, k_\varphi)\in R_b}\bigl|\hat f(k_\theta, k_\varphi)\bigr|,
\]

followed by a $\log(1 + f_b)$ post-transform to compress the dynamic range. The default number of bins is $n_b = 32$; the descriptor has total width $n_b\cdot C$. The descriptor is azimuthally invariant by a pointwise argument: under an integer roll of the input by $k$ bins along $\varphi$, every Fourier component transforms by a unit-modulus phase, $\hat f(k_\theta, k_\varphi)\to e^{-i 2\pi k_\varphi k/n_\phi}\hat f(k_\theta, k_\varphi)$, so $|\hat f(k_\theta, k_\varphi)|$ is invariant pointwise on the $(k_\theta, k_\varphi)$ plane; the radial-bin mean is a function of pointwise magnitudes alone and inherits the invariance trivially.

### 2.5.2. Spherical-harmonic spectral power

The second descriptor projects a single real channel onto the *scalar* spherical-harmonic basis $\{Y_{lm}\}$ and aggregates the resulting coefficients by degree $l$. This is *not* the analytic VSH inverse of §1.3: the basis is scalar (not vector), the input is one real channel (not the complex tangential field), and the output is a per-degree energy summary rather than a recovery of the multipole coefficients.

For a single channel $f(\theta,\varphi)$ the area-weighted projections

\[
c_{lm}\;=\;\sum_{\theta,\varphi}\;\mu(\theta)\,f(\theta,\varphi)\,\overline{Y_{lm}(\theta,\varphi)},\qquad \mu(\theta) = \sin\theta\,\Delta\theta\,\Delta\varphi,
\]

are computed for every $(l, m)$ with $1 \le l \le L$ and $-l \le m \le l$. The per-degree spectral power is the $m$-aggregated squared magnitude,

\[
s_l\;=\;\sum_{m=-l}^{l}\;|c_{lm}|^2,\qquad l = 1, \dots, L,
\]

with a $\log(1 + s_l)$ post-transform. The output is a vector of length $L \cdot C$ — at $L = 5$, that is $5$ scalars on POWER, $10$ on MAGNITUDE, $20$ on COMPLEX.

The PCA reduction of §2.4 and the two descriptors of this section are not new features but admit *compositions*. Two such compositions are used in the experimental chapters and are named here for reference: the *PCA + CV* stack concatenates a moderate-rank PCA reduction ($r = 64$ on POWER input), a shortened radial Fourier descriptor ($n_b = 16$ bins), and the spherical-harmonic spectral-power descriptor, giving a width of $64 + 16 + 5 = 85$ on POWER at $L = 5$; the *raw + SH* stack omits PCA and concatenates the flat input vector with the spherical-harmonic spectral-power descriptor, giving a width of $64{,}440 + 5 = 64{,}445$ on POWER. The labels *PCA + CV* and *raw + SH* are this thesis's compositional shorthands; they are reused in the experimental tables of Chapter 7 but have no broader convention behind them.

## 2.6. Normalisation of features and target variables

### Feature normalisation

The final stage in every feature pipeline of §§2.3–2.5 is a per-dimension standardisation. Let $Z \in \mathbb{R}^{N\times d_\mathrm{feat}}$ stack the feature vectors of the training partition row-wise, possibly the concatenation of several blocks in the *PCA + CV* or *raw + SH* stacks of §2.5. Column-wise mean $\hat\mu \in \mathbb{R}^{d_\mathrm{feat}}$ (the hat distinguishes the feature mean $\hat\mu$ from the spherical area weight $\mu(\theta)$ of §1.3) and standard deviation $\hat\sigma \in \mathbb{R}^{d_\mathrm{feat}}$ are estimated on the training partition only, and every subsequent feature vector is mapped by

\[
z\;\mapsto\;\frac{z - \hat\mu}{\hat\sigma + \varepsilon}, \qquad \varepsilon = 10^{-8}.
\]

The additive $\varepsilon$ guards against division by zero on degenerate feature dimensions. At inference the same $\hat\mu, \hat\sigma$ are reused on the validation and held-out partitions; for the named composite stacks of §2.5 the scaler applies to the concatenated feature vector.

### Target normalisation

The packed coefficient vector $a \in \mathbb{R}^{4K}$ reaches the loss layer with its natural amplitudes; no per-dimension standardisation is applied to the target in this thesis. Two implications follow that Chapter 5 must accommodate. *First*, the coefficient-MSE loss of §5.1 inherits the natural amplitudes of $a$; modes of larger magnitude contribute more strongly to the loss than modes of smaller magnitude, even when their relative prediction error is comparable. *Second*, the physics-informed power loss of §5.2 evaluates the mismatch in the $\|\cdot\|_w$-norm of §1.8 on the area-weighted angular grid and so does re-impose a natural absolute scale through the weight $\mu(\theta) = \sin\theta\,\Delta\theta\,\Delta\varphi$, even though the coefficient-space loss does not.

### The full preprocessing chain

Combining the constructions of §§2.3–2.6, the path from a raw sample $(E_{UT}, a)$ to the model input is

\[
(E_{UT}, a)\;\xrightarrow{\text{mode (§2.3)}}\;x\;\xrightarrow{\text{reduce / describe (§§2.4, 2.5)}}\;z\;\xrightarrow{\text{standardise (§2.6)}}\;\tilde z\;\xrightarrow{\text{model (Ch. 4)}}\;\hat a,
\]

with the target $a$ flowing unscaled to the loss head.

---

## References used in this chapter

- N. Halko, P.-G. Martinsson, J. A. Tropp, "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions", *SIAM Review*, vol. 53, no. 2, pp. 217–288, 2011. doi:10.1137/090771806. arXiv:0909.4061.
- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, É. Duchesnay, "Scikit-learn: Machine Learning in Python", *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
- A. F. Lerma Pineda, P. C. Petersen, "Deep neural networks can stably solve high-dimensional, noisy, non-linear inverse problems", arXiv:2206.00934, 2022.

<!-- Sources backed by research/ch2-data-and-features/manifest.md (R1–R7). -->
