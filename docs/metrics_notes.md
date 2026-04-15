# Metrics Ideas for Radiation Pattern Evaluation

## Raw Notes (verbatim)

> Metrics over P (ideas to inspiration):
>
> - metrics of geologic problems (our field differences should be evaluated similar to the
>   fluctuations on earth's surface or in geographic tasks of predicting wind or other weather
>   conditions)
> - Locations of local and global maximums and their differences in predicted vs true
> - our coordinate system is more dense near the poles. There is a trick from gamedev to generate
>   any sphere mesh uniformly, with no respect to poles
> - we can score difference in log scale, or use other functions here
> - Since we want to be more perceptive to field's overall condition, we may want to base our
>   metric upon not only predicted function, but over its derivative as well. Even though we don't
>   know the actual derivative, we can estimate it using methods of estimating derivatives of
>   functions by their values

---

## Idea → Implementation Mapping

### Idea 1 — Geoscience / weather-style field metrics
**Implemented as:** `weighted_mse`, `weighted_rel_l2` in `models/evaluation/metrics.py`

The equirectangular 360×179 grid places 360 points at every latitude ring including the poles,
but a 1°×1° cell at θ=1° covers sin(1°) ≈ 1.7% of the area of a cell at θ=90°. Un-weighted
MSE is therefore dominated by polar noise with negligible physical area.

Fix: weight every grid point by sin(θ) (already stored in `data/ml/features/basis_L*.npz`)
and normalize by Σ sin(θ). This is the standard area-element weighting used in geoscience and
spherical harmonic analysis.

### Idea 2 — Locations of local and global maxima
**Implemented as:** `beam_pointing_error_deg` in `models/evaluation/metrics.py`

The global argmax of P_true and P_pred gives the main-beam peak direction (θ*, φ*) in degrees.
The great-circle angular distance between the two peaks is computed via the spherical law of cosines:

```
d = arccos( sin θ_true · sin θ_pred · cos(Δφ) + cos θ_true · cos θ_pred )
```

This is the single most physically important metric: is the predicted beam pointing at the
right direction in space?

### Idea 3 — Pole-density bias (gamedev uniform sphere trick)
**Implemented via:** sin(θ) weighting in all area-weighted metrics (same as Idea 1)

The gamedev trick (Fibonacci lattice / equal-area projection) solves the same problem that
sin(θ) weighting solves analytically: points near the equator represent more solid angle than
points near the poles, and we correct for that. The Fibonacci lattice would give a truly
uniform point distribution, but our grid is fixed by the physics library. The analytic
weighting achieves the same unbiased integration result without changing the grid.

### Idea 4 — Log-scale scoring
**Implemented as:** `db_rmse` in `models/evaluation/metrics.py`

Radiation patterns are physically meaningful in dB (decibels). Linear MSE is dominated by
the main beam peak. Converting to dB:

```
P_dB(i) = 10 · log₁₀( P(i) / max(P) + ε )
db_rmse = sqrt( mean( (P_dB_pred - P_dB_true)² ) )
```

This gives equal weight to side-lobe accuracy, which is often what matters in real antenna
characterization. A floor of −40 dB is applied before RMSE to avoid numeric instability in
near-zero regions.

### Idea 5 — Derivative-based metric
**Implemented as:** `gradient_mae` in `models/evaluation/metrics.py`

Even without knowing the analytic derivative, we can estimate ∂P/∂θ and ∂P/∂φ numerically
on the 2-D sphere grid using `numpy.gradient` (central finite differences). The gradient
magnitude |∇P| captures the "texture" of the field: sharp beam edges, correct sidelobe
transitions. We report the mean absolute error of the gradient magnitude:

```
gradient_mae = mean( | |∇P_pred| - |∇P_true| | )
```

### Bonus — Fractions Skill Score (FSS)
**Implemented as:** `fss` in `models/evaluation/metrics.py`

Borrowed from weather forecast verification (Roberts & Lean 2008). Applies a −3 dB threshold
(half-power beamwidth boundary), then slides a spatial window and compares the fraction of
above-threshold pixels in true vs predicted. Rewards predictions that are "in the right
neighborhood" even if not pixel-perfect. Avoids the "double penalty" of classical metrics
where a slightly displaced beam is penalized both for missing and for false alarm.

---

## Metric Summary Table

| Name | Key | Unit | Physical meaning |
|---|---|---|---|
| Area-weighted MSE | `weighted_mse` | power² | Pole-corrected pixel error |
| Area-weighted rel L2 | `weighted_rel_l2` | dimensionless | Pole-corrected shape error |
| dB-scale RMSE | `db_rmse` | dB | Sidelobe-sensitive error |
| Main-beam pointing error | `beam_pointing_error_deg` | degrees | Beam direction accuracy |
| Gradient MAE | `gradient_mae` | power/deg | Pattern sharpness fidelity |
| Fractions Skill Score | `fss` | 0–1 | Spatial beam shape overlap |
| Linear MSE | `p_mse` | power² | Raw pixel error (baseline) |
| Linear MAE | `p_mae` | power | Raw pixel error (baseline) |
| Relative L2 | `p_rel_l2_mean` | dimensionless | Shape error (baseline) |

Implementation: [`models/evaluation/metrics.py`](../models/evaluation/metrics.py)
Used in: [`models/training/baseline_pipeline.py`](../models/training/baseline_pipeline.py)
