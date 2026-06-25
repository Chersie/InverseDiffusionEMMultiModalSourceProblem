# mpinv

A research framework for **phaseless multipole-coefficient inversion**: learn the inverse of the operator
\[
\mathcal A = |\cdot|^2 \circ \mathcal S \colon (a^E, a^M) \in \mathbb C^K \times \mathbb C^K \longrightarrow P \in \mathbb R_{\ge 0}^{360 \times 179}
\]
from synthetic and real-antenna far-field power patterns. Truncation `L=15`, `K=255` modes per family, packed coefficient vector of width `4K=1020` in the order `[Re a^E, Im a^E, Re a^M, Im a^M]`. The full physical setting is in [presentation/ch1_full.md](presentation/ch1_full.md).

## Layout

```
configs/        # Hydra YAMLs (defaults composed here)
src/mpinv/
  core/         # tensor-shape contracts, packing, math primitives
  data/         # synthetic generator, real-antenna loader, memmap dataset
  features/     # PCA, FFT-radial, HOG, SH-power, normalisers
  models/       # registry, base, linear baselines, MLP variants
  losses/       # registry, coef-MSE, physics-power, differentiable VSH decoder
  training/     # Trainer, optimiser builder, AMP, sanity
  callbacks/    # logging, activation stats, checkpoint, early stop, grad clip
  tracking/     # MLflow sink, dataset logger, params, artifacts
  analysis/     # plots, reports, metrics
  cli/          # train, evaluate, sweep, generate_data, validate_physics, report
tests/          # unit + integration
scripts/
notebooks/
research/       # research manifests (Phase 3 source-of-truth)
docs/
```

## Quick start

```bash
uv venv --python 3.12
uv sync --extra dev --extra cv
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000 &
uv run mpinv-train trainer=fast_dev_run
```

See [docs/architecture.md](docs/architecture.md) for the design and [docs/config_cookbook.md](docs/config_cookbook.md) for Hydra recipes.

## Conventions

- **Canonical tensor layout** for the angular grid is `(B, n_theta=179, n_phi=360)` everywhere `torch-harmonics` is touched, and `(B, n_phi=360, n_theta=179)` only at the synthetic-generator boundary (numpy/einsum side). The boundary is a single function `to_torch_layout(...)` with an in-line shape assertion.
- **Phase units** are radians inside the package, with conversion happening exactly once in [src/mpinv/data/real_antenna_loader.py](src/mpinv/data/real_antenna_loader.py).
- **Single source of truth** for every registry. No duplicate `get_model_registry` like the legacy.
- **No silent reshape** inside losses. Shape assertions, not bilinear resize fallbacks.

## License

MIT.
