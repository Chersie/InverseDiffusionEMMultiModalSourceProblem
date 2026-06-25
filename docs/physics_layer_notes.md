# Physics layer notes

The differentiable VSH decoder ([src/mpinv/losses/differentiable_field.py](../src/mpinv/losses/differentiable_field.py)) is the framework's single most error-prone module. This document captures the design and the regression checks pinning it.

## API

```
DifferentiableMultipoleField(grid: GridSpec, l_max: int, basis: VSHBasis | None = None)
forward(packed: Tensor[B, 4 K]) -> Tensor[B, n_theta, n_phi]      # power pattern
forward(packed, return_field=True) -> (P, E_complex)               # complex field too
```

Internally the layer holds the VSH basis tensor as two buffers (real and imaginary parts) of shape `(K, 2 family, 2 component, n_theta, n_phi)`. The forward pass is two einsums per family per channel and a sum-of-squares — no FFT, no torch-harmonics, no transposes.

## Why we do not use `torch-harmonics` for the production path

1. `InverseRealVectorSHT` returns a **real** tangential field. We need a **complex** field whose magnitude squared gives the power pattern (R1 in [research/framework-rebuild/manifest.md](../research/framework-rebuild/manifest.md)).
2. The library's `equiangular` grid is Clenshaw–Curtiss with poles included; the project grid excludes the poles.
3. The legacy adapter on top of `torch-harmonics` had two bugs that this project must not re-introduce:
   - **Wrong (l, m) indexing**: legacy used `th_l_idx = l - 1` — incorrect because `lmax` is non-inclusive; the correct index for `l` is `l` itself.
   - **Wrong m-handling**: legacy added `+m` and `−m` contributions to the same slot in the SHT grid, double-counting because the library handles the negative-m mirror internally via Hermitian symmetry.

We use the einsum-based decoder instead, which sidesteps all three issues. `torch-harmonics 0.6.5` (the last pure-Python wheel) remains pinned as a dev/test cross-check.

## The four regression tests

[tests/unit/test_differentiable_field.py](../tests/unit/test_differentiable_field.py) enforces:

1. **Reciprocity** — the numpy einsum forward in [`mpinv.data.synthetic_generator`](../src/mpinv/data/synthetic_generator.py) and the torch einsum forward in [`mpinv.losses.differentiable_field`](../src/mpinv/losses/differentiable_field.py) agree pointwise on `P` to within float32 round-off.
2. **Gradient flow at the project grid** — at `(n_theta=179, n_phi=360, L=15)` a single forward+backward pass produces non-zero gradients on the input packed coefficients. This closes the open question raised in the legacy [gradient_flow_investigation_results.md](/Users/chersie/Desktop/diplom/gradient_flow_investigation_results.md).
3. **Reflected-conjugate ambiguity** — for any sampled `a`, the analytic map `a'_{l,m} = (-1)^m * conj(a_{l,-m})` (presentation/ch1_full.md §1.7) yields a power pattern that matches `P[a]` to noise floor.
4. **Single-mode injection** — for every `(l, m, family)` at the test truncation, setting that one coefficient to `1+0j` and running the decoder reproduces the data generator's einsum result.

The `mpinv-validate-physics` CLI runs an abridged version (1 + 2) outside pytest, so the result can be pasted into experiment notes or logged as an MLflow artifact.

## Closed mathematical bug, for the record

The polar derivative recursion in [`mpinv.data._basis_cache._dtheta_sph_harm`](../src/mpinv/data/_basis_cache.py) is

\[
\partial_\theta Y_l^m = m \cot(\theta) Y_l^m + \sqrt{(l - m)(l + m + 1)}\, e^{-i\varphi}\, Y_l^{m+1}.
\]

The second term vanishes only when `m + 1 > l`. An earlier version of this file used `abs(m) + 1 > l` as the bound check, which incorrectly skipped the second term for `m = -1, l = 1` (and similar negative-m cases) and produced a basis that violated the VSH conjugation rule
\[
\overline{\boldsymbol\Psi^X_{l,m}} = (-1)^m \boldsymbol\Psi^X_{l,-m}.
\]
Fixed in commit (this file). The reflected-conjugate test catches any future regression of the same kind.
