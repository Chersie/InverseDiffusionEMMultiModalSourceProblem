"""``mpinv-report``: build the standard per-run report from saved model + data.

Loads a checkpoint plus the synthetic-validation arrays (or any held-out arrays
provided), runs inference, and produces the figures + metrics defined in
:mod:`mpinv.analysis.reports.run_report`.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from mpinv.analysis.reports.run_report import RunArtifacts, build_run_report
from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import build_basis
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.models.mlp import MLP, MLPConfig

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a per-run analysis report.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="report")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--n-theta", type=int, default=12)
    parser.add_argument("--theta-start", type=float, default=7.5)
    parser.add_argument("--theta-end", type=float, default=172.5)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    grid = GridSpec(
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        theta_start_deg=args.theta_start,
        theta_end_deg=args.theta_end,
    )
    basis = build_basis(grid, args.l_max)
    gen = SyntheticGenerator(
        cfg=SyntheticGeneratorConfig(grid=grid, l_max=args.l_max),
        basis=basis,
    )

    rng = np.random.default_rng(0)
    P, packed = gen.generate_batch(args.n, rng)
    feat = PowerPCAPipeline(cfg=PowerPCAPipelineConfig(pca_components=args.pca_components))
    feat.fit(P_train=P)
    z = feat.transform(P=P)

    K = args.l_max * (args.l_max + 2)
    model = MLP(
        MLPConfig(
            input_dim=feat.feature_dim,
            output_dim=4 * K,
            hidden_size=args.hidden_size,
            n_hidden_layers=args.n_hidden_layers,
        )
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(z).float()).numpy()

    decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)
    with torch.no_grad():
        P_pred = decoder(torch.from_numpy(pred).float()).numpy()

    art = RunArtifacts(
        pred_packed=pred,
        target_packed=packed,
        P_pred=P_pred,
        P_true=P,
        l_max=args.l_max,
        pca_explained_variance_ratio=feat.explained_variance_ratio_,
    )
    metrics = build_run_report(art, output_dir=args.output_dir)
    for k, v in metrics.items():
        logger.info("%s = %.6f", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
