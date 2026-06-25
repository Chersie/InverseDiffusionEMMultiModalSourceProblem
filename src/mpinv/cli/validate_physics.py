"""``mpinv-validate-physics``: standalone validator for the differentiable VSH decoder.

Runs the same checks as ``tests/unit/test_differentiable_field.py`` but as a CLI so we
can paste the report into docs and into MLflow as an artifact.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import build_basis
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)
from mpinv.losses.differentiable_field import DifferentiableMultipoleField

logger = logging.getLogger(__name__)


def _reciprocity(
    decoder: DifferentiableMultipoleField, gen: SyntheticGenerator, n: int = 4
) -> float:
    rng = np.random.default_rng(0)
    P_np, packed = gen.generate_batch(n, rng)
    P_th = decoder(torch.from_numpy(packed)).detach().numpy()
    return float(np.abs(P_th - P_np).max() / max(np.abs(P_np).max(), 1e-12))


def _gradient_flow(decoder: DifferentiableMultipoleField, n: int = 2) -> bool:
    K = decoder.K
    packed = torch.randn(n, 4 * K, requires_grad=True)
    P = decoder(packed)
    P.pow(2).sum().backward()
    return packed.grad.abs().sum().item() > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the differentiable VSH decoder.")
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--n-theta", type=int, default=12)
    parser.add_argument("--theta-start", type=float, default=7.5)
    parser.add_argument("--theta-end", type=float, default=172.5)
    parser.add_argument("--l-max", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    grid = GridSpec(
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        theta_start_deg=args.theta_start,
        theta_end_deg=args.theta_end,
    )
    basis = build_basis(grid, l_max=args.l_max)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)
    gen = SyntheticGenerator(SyntheticGeneratorConfig(grid=grid, l_max=args.l_max), basis=basis)

    rec = _reciprocity(decoder, gen)
    grad_ok = _gradient_flow(decoder)

    logger.info("reciprocity max-rel-err = %.3e (target < 1e-4)", rec)
    logger.info("gradient flow OK = %s", grad_ok)
    if rec > 1e-4 or not grad_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
