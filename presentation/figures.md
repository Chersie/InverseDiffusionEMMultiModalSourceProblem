# Figure specifications for Chapter 1

This file is the production spec for the figures referenced in [ch1_full.md](ch1_full.md). Each entry fixes the figure's identity, caption, data source, placement, and style; the actual rendering scripts live in `scripts/` and are tracked separately from this document. Figures are numbered as `Fig 1.Y` matching the section in which they appear, consistent with the merged single-chapter TOC in [header.md](header.md). The previous numbering (Fig 2.x for what is now §1.7 and §1.8) has been retired.

The convention is that *placement* is the anchor sentence in [ch1_full.md](ch1_full.md); *data source* is the canonical artifact in the repository from which the figure must be regenerated (or, for schematics, "drawn from scratch"); *style* fixes layout, colormap, and aspect choices for visual consistency across the thesis.

---

## Fig 1.1 — Sample vector spherical harmonic mode on the unit sphere

- **Caption**: "Magnitude of the vector spherical harmonic $\boldsymbol{\Psi}^E_{2,1}$ on the unit sphere. The angular structure of a single VSH mode determines the angular fingerprint of one entry in the multipole expansion of §1.1."
- **Placement**: §1.1, immediately after the sentence introducing the explicit form of $\{\boldsymbol{\Psi}^E_{lm}, \boldsymbol{\Psi}^M_{lm}\}$.
- **Data source**: cached basis tensor of shape `(K, 64440, 2)`. For this figure, slice the entry corresponding to $(l=2, m=1)$, electric family; reshape to `(360, 179, 2)`; compute pointwise norm $\sqrt{|\Psi_\theta|^2 + |\Psi_\varphi|^2}$.
- **Visualisation**: 3-D mercator-deformed sphere with the magnitude rendered as surface colour, viewing angle chosen to expose both the polar and equatorial structure. One panel.
- **Style**: perceptually uniform colormap (e.g. `viridis`), no axes labels on the sphere, scale bar on the side, square aspect, vector format (PDF/SVG).
- **Status**: not yet rendered.

## Fig 1.2 — One real antenna's complex field components and its power pattern

- **Caption**: "A single real-antenna far-field pattern from the held-out test set. Top row: real and imaginary parts of $E_\theta$ and $E_\varphi$ on the $360\times 179$ angular grid (four panels). Bottom row: the corresponding power pattern $P = |E_\theta|^2 + |E_\varphi|^2$ (one panel). The four complex components carry $4\times 360\times 179$ real numbers; the power pattern carries one quarter of that."
- **Placement**: §1.2, after the angular-grid definition and before the degree-of-freedom count.
- **Data source**: a single file from the real-antenna test set. Each file has a seven-column layout: `theta_deg, phi_deg, power, |E_theta|, arg(E_theta), |E_phi|, arg(E_phi)`. Reconstruct the complex components as $E_\theta = |E_\theta|\,\mathrm{e}^{i\arg(E_\theta)}$ and likewise for $E_\varphi$. The exact file path resolution and loader contract are defined by the data layer of the project (see Chapter 2 of the thesis).
- **Visualisation**: five equal-aspect $(\varphi, \theta)$ heatmaps on a single canvas, two-by-three grid with the bottom-right cell empty. Diverging colormap (e.g. `RdBu_r`) for the real/imaginary panels, sequential colormap (`viridis`) for the power panel. Shared colour scales within the real/imag group; separate scale for power.
- **Style**: $\theta$ on the vertical axis (downward), $\varphi$ on the horizontal axis. Tick marks every $30°$. One shared $\sin\theta$-aware aspect-ratio choice across all panels.
- **Status**: not yet rendered.

## Fig 1.4 — Anchor figure: complex field versus power pattern, side by side

- **Caption**: "The information collapse of §1.4 made visual. Left: the complex field $E_{UT}=(E_\theta, E_\varphi)$ for one real antenna, encoded as a $4\times$-channel heatmap stack ($4\times 360\times 179 = 257{,}760$ real numbers). Right: the corresponding power pattern $P = |E_\theta|^2 + |E_\varphi|^2$ on the same grid ($360\times 179 = 64{,}440$ real numbers). The right panel is what a phaseless antenna characterisation actually measures."
- **Placement**: §1.4 anchor figure, immediately after the global-DoF accounting paragraph.
- **Data source**: same antenna as Fig 1.2 (so the two figures cross-reference). Reuse the loader path and the seven-column file format.
- **Visualisation**: two large square panels side by side. Left panel: a stacked composite of the four real-valued components, either as a quadtych or as a single RGBA composite encoding $(\mathrm{Re}\,E_\theta, \mathrm{Im}\,E_\theta, \mathrm{Re}\,E_\varphi, \mathrm{Im}\,E_\varphi)$. Right panel: the scalar power pattern. Annotate each panel with the real-DoF count.
- **Style**: matching axes and aspect with Fig 1.2; horizontally compressed caption to keep the figure dominant on the page. This is the *one* figure of Chapter 1 that the reader will remember and that should appear in a defence-talk slide unaltered.
- **Status**: not yet rendered.

## Fig 1.7 — Two distinct coefficient sets, one identical power pattern

- **Caption**: "Demonstration of the non-uniqueness of $\mathcal{A}^{-1}$ at the trivial-ambiguity level. Coefficient sets $(a^E_1, a^M_1)$ and $(a^E_2, a^M_2)$ — top row, packed-coefficient bar plots — produce visually identical power patterns $P_1, P_2$ — bottom row, two heatmaps. The two coefficient configurations are related by the reflected-conjugate trivial ambiguity established in §1.7 (Bangun 2020, §6.2): $\hat g_l^k = (-1)^k\,\overline{\hat f_l^{-k}}$. They differ by more than the global $U(1)$ phase."
- **Placement**: §1.7, at the end of the *(ii) Reflected-conjugate ambiguity* paragraph. This is the visceral demonstration of the non-uniqueness argument.
- **Data source**: a small synthesis script (to be added under `scripts/figures/fig_1_7_nonuniqueness.py`) that uses the project's data generator to (i) sample a baseline coefficient vector $(a^E_1, a^M_1)$, (ii) construct $(a^E_2, a^M_2)$ by applying the reflected-conjugate map of §1.7 (ii) to the VSH coefficients of $(a^E_1, a^M_1)$, and (iii) verify numerically that $\|P_1 - P_2\|_w < 10^{-6}\|P_1\|_w$. The construction is *not* an open numerical search; it is a deterministic application of the analytically-known ambiguity. The script must be reproducible from a fixed RNG seed for $(a^E_1, a^M_1)$ and must record both coefficient vectors as figure metadata.
- **Visualisation**: $2\times 2$ panel: top row two bar plots (packed coefficient vectors of length $4K=1020$, with the four blocks $\mathrm{Re}\,a^E$, $\mathrm{Im}\,a^E$, $\mathrm{Re}\,a^M$, $\mathrm{Im}\,a^M$ visually demarcated); bottom row two power-pattern heatmaps with shared colour scale and a residual heatmap inset showing $P_1 - P_2$ at $10^4\times$ exaggeration to confirm the match is genuine.
- **Style**: shared $y$-axis on the bar plots; shared colour scale on the heatmaps; the residual inset uses a diverging map to show the noise-floor difference.
- **Status**: not yet rendered. Construction is now mechanical (closed-form ambiguity), not a numerical search.

## Fig 1.8 — Cartoon: solution-set landscape with and without learned prior

- **Caption**: "Schematic of the regularised inverse problem of §1.8. The fidelity term $\|\mathcal{A}(a) - P_{UT}\|_w^2$ vanishes on a (typically multi-component) admissible set in coefficient space (orange surface, with two components representing the trivial-ambiguity reflections of §1.7). Classical regularisers (Tikhonov, sparsity) impose explicit, hand-designed priors that may or may not intersect this admissible set near a physically realistic point. A learned regulariser, parametrised by the network weights $w$ and trained on the empirical distribution of realistic $(a^E, a^M)$ vectors (blue cloud), concentrates the selected solution on the realistic part of the admissible set."
- **Placement**: §1.8 closing figure, after the paragraph identifying the fully-learned family.
- **Data source**: schematic; no real data. Drawn from scratch in a vector tool (TikZ from a build script, or Inkscape SVG checked into `assets/figures/`).
- **Visualisation**: a 2-D projection of an abstract coefficient space, with the admissible set drawn as a curve (with two components to evoke the trivial-ambiguity multiplicity), a Tikhonov-favoured point (closest to origin), a sparsity-favoured point (on a coordinate axis), a learned-prior cloud of training samples, and the selected solution at the intersection of the cloud with the admissible set. Legend, no numerical axes.
- **Style**: deliberately schematic — round, smooth curves; no quantitative axes; minimal labels that match the symbols of §1.8 verbatim ($\hat a$, $\mathcal{A}^{-1}(P_{UT})$, $f_w(P_{UT})$, $R(a)$).
- **Status**: not yet rendered.

---

## Cross-cutting style rules

These rules apply to every figure in this list:

- **Vector format** (PDF or SVG) for any figure that contains text, axes, or schematic elements; rasterised PNG only for dense $360\times 179$ heatmaps and only when explicitly necessary.
- **Single colormap family per semantic type**: sequential `viridis` for non-negative scalars (power, magnitudes), diverging `RdBu_r` for signed quantities (real/imaginary parts, residuals).
- **Spherical-area awareness**: any heatmap on the $(\varphi, \theta)$ grid uses an aspect ratio that does not visually exaggerate the polar regions; a $\sin\theta$-corrected aspect ratio is preferred.
- **One idea per figure**. The four-panel composites (Fig 1.2, Fig 1.4) violate this nominally but are constructed to make the *single* qualitative point of "complex versus power" — the panels are not independent contributions.
- **Numerical traceability**: each figure's rendering script logs the data file(s) it consumed and writes them into the figure metadata, so the figure is reproducible from a clean checkout.

## Implementation notes

The Fig 1.7 construction has been simplified relative to earlier drafts: thanks to the closed-form reflected-conjugate ambiguity (Bangun 2020, §6.2), the second coefficient vector is no longer the result of a constrained numerical search but a deterministic, analytically-verifiable transformation of the first. The script need only sample one baseline and apply the map; correctness is then a one-line numerical check that $\|P_1 - P_2\|_w$ is at the floating-point noise floor.

The other four figures are mechanical to render once the data source is loaded: Figs 1.1, 1.2, 1.4 are direct visualisations of existing artifacts; Fig 1.8 is a schematic that requires only design, not data.

<!-- All citations in this file are backed by research/ch1-enrichment/manifest.md (R1–R6). -->
