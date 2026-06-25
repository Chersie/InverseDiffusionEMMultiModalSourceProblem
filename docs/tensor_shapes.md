# Tensor shapes & layouts

This document is the binding contract for tensor layouts everywhere in `mpinv`. If a function disagrees with this document, the function is wrong.

## Canonical grid

| symbol | meaning | default |
|---|---|---|
| `n_phi` | azimuthal samples | 360 |
| `n_theta` | polar samples (poles excluded) | 179 |
| `theta_axis` | radians, samples 1°..179° at 1° spacing | — |
| `phi_axis` | radians, samples 0°..359° at 1° spacing | — |

The single source of truth is [`mpinv.core.grid.GridSpec`](../src/mpinv/core/grid.py). Tests use a tiny grid (e.g. `n_phi=12, n_theta=8`).

## Power pattern

`P` has shape `(B, n_theta, n_phi)`, `float32`. Always non-negative.

## Complex field

`E` has shape `(B, 2 channels, n_theta, n_phi)`, `complex64`. Channel 0 is `E_theta`, channel 1 is `E_phi`.

## Packed coefficients

Shape `(B, 4 K)`, `float32`. The trailing axis is laid out as four equal blocks `[Re a^E, Im a^E, Re a^M, Im a^M]`. `K = L (L + 2)`; for `L = 15` (the project default) `K = 255` and `4 K = 1020`.

Within each block, modes are ordered by `iter_modes(L)`: `l` ascending, `m` from `-l` to `+l`. The functions [`pack_coefficients`](../src/mpinv/core/packing.py) and [`unpack_coefficients`](../src/mpinv/core/packing.py) are the only canonical conversions.

## VSH basis tensor

Shape `(K, 2 family, 2 component, n_theta, n_phi)`, `complex64`.

- `family = 0` is electric (TM, spheroidal); `family = 1` is magnetic (TE, toroidal).
- `component = 0` is the θ-component; `component = 1` is the φ-component.

See [`mpinv.data._basis_cache`](../src/mpinv/data/_basis_cache.py).

## Phase units

Inside the framework, phases are **radians**. The only place where degrees appear is in the seven-column real-antenna file format, and the conversion happens exactly once in [`mpinv.data.real_antenna_loader._parse_feature_file`](../src/mpinv/data/real_antenna_loader.py).

## Where transposes happen

The synthetic generator and the differentiable decoder both use the canonical `(B, ..., n_theta, n_phi)` layout natively. The seven-column real-antenna files store rows as `(phi outer, theta inner)`; the loader transposes on read. There are no other layout boundaries.
