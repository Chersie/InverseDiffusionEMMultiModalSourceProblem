# Numerical Shapes Reference

This document defines key fields, vectors, and matrices used in the pipeline, with canonical shapes and meanings.

## Physical Glossary

This glossary maps physical terms to the numerical objects used in this project.

- `E_UT` (field under test)
  - Meaning: the far-field you want to analyze/decompose (measured or synthesized).
  - In this codebase: `a_study` in `src/pipeline/decompose_fields.py`.
  - Shape: `(360, 179, 2)` with component convention:
    - `[..., 0]` -> `E_theta` (complex amplitude)
    - `[..., 1]` -> `E_phi` (complex amplitude)
  - Physical interpretation: a complex vector field sampled on the angular grid `(phi, theta)`.

- `aE[l][m]`, `aM[l][m]` (multipole coefficients)
  - Meaning: expansion weights of the field in multipole basis functions.
  - `aE[l][m]`: electric-type mode weight.
  - `aM[l][m]`: magnetic-type mode weight.
  - Indices:
    - `l` -> multipole order.
    - `m` -> azimuthal index for the given order.
  - In this codebase:
    - synthesized coefficients: `a_e[l][m]`, `a_m[l][m]` in `src/pipeline/generate_fields.py`;
    - recovered/projection coefficients: `e_j`, `m_j` written to `Results_*.txt` in `src/pipeline/decompose_fields.py`.
  - Physical interpretation: "how much of each multipole mode is present in the field."

- `P_UT` and `E_i * (E_i*)`
  - Meaning: power/intensity from complex field amplitudes.
  - For one component `E_i`: `E_i * (E_i*) = |E_i|^2`.
  - For this two-component far field:
    - `P_UT` is proportional to `|E_theta|^2 + |E_phi|^2`.
  - In this codebase:
    - `power = np.sum(np.abs(amplitude) ** 2, axis=-1)` in `src/pipeline/generate_fields.py`;
    - serialized as column 3 of `Fields.txt`.

- `E_basis(l,m)` and decomposition idea
  - The library files `E_l*_m*.txt`, `M_l*_m*.txt` contain basis field patterns for each `(l,m)`.
  - Decomposition computes inner products between `E_UT` and each basis mode over angle with spherical weight `sin(theta) dtheta dphi`.
  - The resulting complex numbers are the estimated mode coefficients in `Results_*.txt`.

## Core Constants

- `ANGLE_STEP_DEG = 1`
- `DEFAULT_MAXORDER = 15`
- `LIBRARY_HEADER_LINES = 43`

Derived:

- `size_phi = 360 / ANGLE_STEP_DEG = 360`
- `size_theta = 180 / ANGLE_STEP_DEG - 1 = 179`
- `extrasize = size_phi * size_theta = 64440`
- Number of `(l, m)` pairs up to order `L`: `L * (L + 2)` -> for `L=15`, `255`
- Number of projection lines in `Results_*.txt`: `2 * 255 = 510` (`E` and `M`)

## Canonical Data Objects (`src/`)

| Name | Shape | Type | Meaning | Source |
|---|---:|---|---|---|
| `GridShape.size_phi` | scalar (`360`) | int | azimuth grid count | `src/pipeline/generate_fields.py` |
| `GridShape.size_theta` | scalar (`179`) | int | polar grid count (no poles) | `src/pipeline/generate_fields.py` |
| `theta_2d` | `(360, 179)` | float64 | polar angles in radians | `src/pipeline/generate_fields.py` |
| `phi_2d` | `(360, 179)` | float64 | azimuth angles in radians | `src/pipeline/generate_fields.py` |
| `ls_e_re`, `ls_e_im`, `ls_m_re`, `ls_m_im` | `(31, 31)` | float64 | Latin-square coefficient tables (`2L+1`, `L=15`) | `src/pipeline/generate_fields.py` |
| `a_e[l][m]`, `a_m[l][m]` | dict-of-dict | complex | electric/magnetic multipole coefficients | `src/pipeline/generate_fields.py` |
| `amplitude` | `(360, 179, 2)` | complex128 | synthesized field components | `src/pipeline/generate_fields.py` |
| `power` | `(360, 179)` | float64 | per-point power: `sum(abs(amplitude)^2)` | `src/pipeline/generate_fields.py` |
| `a_study` | `(360, 179, 2)` | complex128 | field under test from `Fields.txt` | `src/pipeline/decompose_fields.py` |
| `amp` | `(360, 179, 2)` | complex128 | one library mode (`E_lm` or `M_lm`) | `src/pipeline/decompose_fields.py` |
| `theta` (library) | `(360, 179)` | float64 | per-sample theta (degrees) from library file | `src/pipeline/decompose_fields.py` |
| `e_j`, `m_j` | scalar | complex128 | projection coefficients for one `(l,m)` | `src/pipeline/decompose_fields.py` |
| `d_omega` | scalar | float64 | angular cell factor `(step*pi/180)^2` | `src/pipeline/decompose_fields.py` |

Component convention for the last axis of 3D field tensors:

- `[..., 0]` -> `E_theta` amplitude
- `[..., 1]` -> `E_phi` amplitude

## Legacy Objects (still relevant)

| Name | Shape | Type | Meaning | Source |
|---|---:|---|---|---|
| `theta`, `phi` (legacy library grid) | `(361, 181)` | float64 | grid including poles (`0..180`, `0..360`) from older pre-generated libraries | historical artifacts |
| `field_for_multipole(...)` output | `(n_phi, n_theta, 2)` | complex128 | mode field components (`theta`, `phi`) | `Chersie/MPField_Spherical_Fast.py` |
| `A_study`, `A_temp` | `(360, 179, 2)` | complex128 | decomposition tensors | `NaiveSolution/3 FieldsToMultipoles.py` |
| `aE`, `aM` | nested dict | complex | multipole coefficients from `Results_*.txt` | `NaiveSolution/4 Plot3DMultipoles.py` |

## File-Level Vector/Matrix Layouts

### `Fields.txt` layout

Each row has 7 columns:

1. `theta_deg`
2. `phi_deg`
3. `power`
4. `abs(E_theta)`
5. `arg(E_theta)` (rad)
6. `abs(E_phi)`
7. `arg(E_phi)` (rad)

Expected rows for canonical decomposition grid: `360 * 179 = 64440`.

Row ordering:

- `phi` outer loop (`0..359`)
- `theta` inner loop (`1..179`)

### Library file layout (`E_l*_m*.txt`, `M_l*_m*.txt`)

- Header lines: `43` (skipped)
- Data columns align with `Fields.txt` style (`theta`, `phi`, power, amplitudes/phases)
- Canonical layout: `360 * 179 = 64440` rows (`phi=0..359`, `theta=1..179`).
- Legacy compatibility: `src` decomposition also accepts old `361 * 181 = 65341` grids and automatically maps to the canonical subset (`phi=0..359`, `theta=1..179`).
- Angle units: canonical libraries store `theta/phi` in degrees; decomposition auto-detects and converts older radian-based files.

### `Results_*.txt` layout

Each row:

- `<mode> <l> <m> <real> <imag>`
- where `<mode>` is `E` or `M`

For `L=15`: 510 rows total.

## Shape Notes and Mismatch Awareness

There are two library grid conventions in artifacts:

1. **Canonical grid**: `(360, 179)` without poles (used by `Fields.txt`, `src` decomposition, and current fast library generation).
2. **Legacy grid**: `(361, 181)` including poles (present in older generated library files).

During decomposition, `src` reads the canonical rows directly when the library is canonical.
For legacy libraries, it maps each canonical point to the legacy row with `idx = (size_theta + 2) * j + (i + 1)` and skips pole rows.

Validation helper:

- Run `python src/cli/validate_grid_files.py` to verify row counts and row ordering for `Fields.txt` and a sample library mode file.

## Suggested Rule for Future Modules

- If a function outputs field tensors for decomposition/training, prefer shape `(n_phi, n_theta, 2)` with explicit axis order `(phi, theta, component)`.
- Always document:
  - angular units (`deg` vs `rad`)
  - whether poles are included
  - row order used when serialized to disk

## ML Baseline Shapes

Baseline workflow (`P_UT -> coeffs -> E^ -> P^`) uses the following tensors:

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `X_power` | `(N, 64440)` | float32 | flattened `P_UT` features (`360*179`) |
| `Y_coeff` | `(N, 4*K)` | float32 | packed targets: `[Re(a_E), Im(a_E), Re(a_M), Im(a_M)]` |
| `K` | scalar (`L*(L+2)`) | int | number of `(l,m)` modes per E/M family |
| `a_E`, `a_M` | `(N, K)` | complex64 | unpacked predicted/true multipole coefficients |
| `basis e_theta/e_phi/m_theta/m_phi` | `(K, 64440)` | complex64 | cached canonical multipole basis |
| `P_hat` | `(N_test, 64440)` | float32 | predicted power from reconstructed `E^` |

Packing order for one sample:

1. iterate modes in deterministic order: `l=1..L`, `m=-l..l`;
2. concatenate E-family then M-family;
3. each family stored as real block then imag block.

Core metrics:

- coefficient-space sanity: `MSE(y_true, y_pred)`
- physics-space quality: `MSE(P, P^)`, `MAE(P, P^)`, mean relative `L2(P, P^)`
