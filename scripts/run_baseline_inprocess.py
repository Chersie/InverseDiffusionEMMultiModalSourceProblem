"""In-process baseline runner for S1, S2, S3.

Generates the synthetic train / val / synthetic-test arrays **once** per
configuration of (generator regime, augmentation, n_train, n_val, n_test) and
reuses them across every model x loss x feature cell. This is dramatically
faster than the subprocess-based runner (`run_baseline_S1.py`) because the
costly L=15 VSH einsum synthesis is amortised across cells.

Usage:

    uv run python scripts/run_baseline_inprocess.py \\
        --stage S1 \\
        --output experiments/baseline/S1_results.json \\
        --models linear,mlp_2x64,mlp_2x256 \\
        --losses coef_mse,physics_power \\
        --features power_pca,raw_flat,cv_only \\
        --seeds 0,1 \\
        --max-epochs 20 \\
        --n-train 2048 --n-val 512 --n-test 512

S2 mode picks the top-K cells from an S1 results file and replays each with
each augmentation. S3 picks the top cell from S2 and replays across regimes.

This driver intentionally does **not** use MLflow; it writes a plain JSON of
per-cell metrics so the report builder can pick them up. To get MLflow logging
use the subprocess driver `run_baseline_S1.py`.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mpinv.analysis.metrics.coefficient_metrics import packed_mse, packed_r2
from mpinv.analysis.metrics.field_metrics import weighted_mse_P, weighted_nrmse_P
from mpinv.analysis.metrics.mode_metrics import reflected_conjugate_aware_loss
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import _ArrayDataset, build_physics_power_loss
from mpinv.core.grid import GRID_DEFAULT
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.augment import apply_augmentation, build_augmentation
from mpinv.data.real_antenna_loader import RealAntennaLoaderConfig, load_real_antenna
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)
from mpinv.features.composite import CompositeFeaturesConfig, CompositePipeline
from mpinv.features.fft_radial import FFTRadial, FFTRadialConfig
from mpinv.features.modes import InputMode
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig
from mpinv.features.raw_flat import RawFlattenPipeline, RawFlattenPipelineConfig
from mpinv.features.sh_power import SHPower, SHPowerConfig
from mpinv.features.subsample import SubsampleGridPipeline, SubsampleGridPipelineConfig
from mpinv.losses.coef_mse import CoefMSE, CoefMSEConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.physics_power import PhysicsPowerLossConfig
from mpinv.models.linear_baselines import LinearBaseline, LinearBaselineConfig
from mpinv.models.mlp import MLP, MLPConfig
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["S1", "S2", "S3"])
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--models", default="linear,mlp_2x64,mlp_2x256", type=str)
    p.add_argument("--losses", default="coef_mse,physics_power", type=str)
    p.add_argument("--features", default="power_pca,raw_flat,cv_only", type=str)
    p.add_argument("--seeds", default="0,1", type=str)
    p.add_argument("--max-epochs", default=20, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--n-train", default=2048, type=int)
    p.add_argument("--n-val", default=512, type=int)
    p.add_argument("--n-test", default=512, type=int)
    p.add_argument("--seed-train", default=1234, type=int)
    p.add_argument("--seed-val", default=5678, type=int)
    p.add_argument("--seed-test", default=9012, type=int)
    p.add_argument("--l-max", default=15, type=int)
    p.add_argument("--regime", default="gaussian", type=str)
    p.add_argument("--color-alpha", default=1.0, type=float)
    p.add_argument("--sparse-active-fraction", default=0.1, type=float)
    p.add_argument("--augmentation", default=None, type=str,
                   help="JSON string for augmentation spec, or 'none'.")
    p.add_argument("--holdout-root", default="data/raw/real_antenna", type=str)
    p.add_argument("--holdout-feature-subdir", default="E_in_plane", type=str)
    # S2/S3 inputs
    p.add_argument("--top-cells", default=None, type=str,
                   help="JSON list of cell dicts (model, loss, features) for S2/S3.")
    p.add_argument("--augmentations", default=None, type=str,
                   help="Comma-separated list of augmentation specs for S2 (each as a name).")
    return p.parse_args()


def _build_generator(args: argparse.Namespace) -> SyntheticGenerator:
    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)
    cfg = SyntheticGeneratorConfig(
        grid=grid,
        l_max=args.l_max,
        mode=args.regime if args.regime != "colored_a1" and args.regime != "colored_a2" else "colored",
        color_alpha=args.color_alpha,
        sparse_active_fraction=args.sparse_active_fraction,
    )
    return SyntheticGenerator(cfg=cfg, basis=basis)


def _build_features(name: str, grid, l_max: int):
    if name == "power_pca":
        return PowerPCAPipeline(PowerPCAPipelineConfig(pca_components=128))
    if name == "power_pca_small":
        return PowerPCAPipeline(PowerPCAPipelineConfig(pca_components=32))
    if name == "raw_flat":
        return RawFlattenPipeline(RawFlattenPipelineConfig())
    if name == "subsample_stride4":
        return SubsampleGridPipeline(
            SubsampleGridPipelineConfig(theta_stride=4, phi_stride=4)
        )
    if name == "cv_only":
        return CompositePipeline(
            cfg=CompositeFeaturesConfig(skip_pca=True),
            extractors=[
                FFTRadial(FFTRadialConfig(n_bins=16)),
                SHPower(SHPowerConfig(l_max=l_max), grid=grid),
            ],
        )
    if name == "pca_cv":
        return CompositePipeline(
            cfg=CompositeFeaturesConfig(
                pca=PowerPCAPipelineConfig(pca_components=256)
            ),
            extractors=[
                FFTRadial(FFTRadialConfig(n_bins=64)),
                SHPower(SHPowerConfig(l_max=l_max), grid=grid),
            ],
        )
    if name == "raw_plus_sh":
        return CompositePipeline(
            cfg=CompositeFeaturesConfig(skip_pca=True),
            extractors=[
                RawFlattenPipeline(RawFlattenPipelineConfig()),
                SHPower(SHPowerConfig(l_max=l_max), grid=grid),
            ],
        )
    if name == "subsample_stridt_plus_sh":
        pass
    raise ValueError(f"unknown feature pipeline: {name!r}")


def _build_model(name: str, input_dim: int, output_dim: int):
    if name == "linear":
        return LinearBaseline(LinearBaselineConfig(input_dim=input_dim, output_dim=output_dim))
    sizes = {"mlp_2x16": 16, "mlp_2x32": 32, "mlp_2x64": 64, "mlp_2x256": 256, "mlp_2x512": 512, "mlp_3x200": 200, "mlp_5x200": 200}
    if name in sizes:
        return MLP(MLPConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=sizes[name],
            n_hidden_layers=2,
            architecture="flat",
        ))
    raise ValueError(f"unknown model: {name!r}")


def _build_loss(name: str, grid, l_max: int):
    if name == "coef_mse":
        return CoefMSE(CoefMSEConfig()), "coef"
    if name == "physics_power":
        return build_physics_power_loss(grid=grid, l_max=l_max), "physics"
    if name == "physics_power_mixed":
        return build_physics_power_loss(grid=grid, l_max=l_max, coef_aux_weight=0.1), "physics"
    if name == "physics_power_rank":
        # physics_power MSE with the rank-bin regulariser at lambda=0.1.
        return (
            build_physics_power_loss(grid=grid, l_max=l_max, rank_bin_weight=0.1),
            "physics",
        )
    if name == "rank_bin_p":
        # Pure rank-bin loss on P; ignores absolute amplitude, focuses
        # entirely on per-pixel rank order.
        from mpinv.losses.rank_bin import RankBinPLoss, RankBinPLossConfig

        return (
            RankBinPLoss(cfg=RankBinPLossConfig(), grid=grid, l_max=l_max),
            "physics",
        )
    raise ValueError(f"unknown loss: {name!r}")


def _eval_split(model, decoder, z, packed, P, l_max, grid, tag) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(z).float()).cpu().numpy()
        P_pred = decoder(torch.from_numpy(preds).float()).cpu().numpy()
    return {
        f"report/{tag}/coef_mse": packed_mse(preds, packed),
        f"report/{tag}/coef_r2": packed_r2(preds, packed),
        f"report/{tag}/coef_mse_amb_aware": reflected_conjugate_aware_loss(preds, packed, l_max),
        f"report/{tag}/field_mse_w": weighted_mse_P(P_pred, P, grid=grid),
        f"report/{tag}/field_nrmse_w": weighted_nrmse_P(P_pred, P, grid=grid),
    }


def run_cell(
    P_train, packed_train, P_val, packed_val, P_test, packed_test,
    holdout_data,
    feat_name: str, model_name: str, loss_name: str,
    seed: int, grid, l_max: int, basis,
    max_epochs: int, batch_size: int,
    aug_name: str = "none",
    feature_cache: dict | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    started = time.time()
    cache_key = (feat_name,)
    if feature_cache is not None and cache_key in feature_cache:
        feat, z_train, z_val, z_test = feature_cache[cache_key]
    else:
        feat = _build_features(feat_name, grid=grid, l_max=l_max)
        feat.fit(P_train=P_train)
        z_train = feat.transform(P=P_train)
        z_val = feat.transform(P=P_val)
        z_test = feat.transform(P=P_test) if P_test is not None else None
        if feature_cache is not None:
            feature_cache[cache_key] = (feat, z_train, z_val, z_test)

    K = l_max * (l_max + 2)
    input_dim = feat.feature_dim
    model = _build_model(model_name, input_dim=input_dim, output_dim=4 * K)
    loss_fn, loss_kind = _build_loss(loss_name, grid=grid, l_max=l_max)
    optimiser = build_optimiser(model, OptimiserConfig(name="adamw", lr=1e-3))

    train_ds = _ArrayDataset(z_train, packed_train, P_train)
    val_ds = _ArrayDataset(z_val, packed_val, P_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = Trainer(TrainerConfig(max_epochs=max_epochs, log_every_n_steps=200))
    callbacks = [
        LoggingCallback(log_every_n_steps=200),
        ValidationCallback(every_n_epochs=1),
        GradClipCallback(max_norm=1.0),
        EarlyStoppingCallback(patience=10),
    ]
    ctx = trainer.fit(
        model=model, train_loader=train_loader, loss_fn=loss_fn,
        optimiser=optimiser, loss_kind=loss_kind, val_loader=val_loader,
        callbacks=callbacks,
    )

    decoder = DifferentiableMultipoleField(grid=grid, l_max=l_max, basis=basis)
    metrics: dict[str, float] = {}
    metrics.update(_eval_split(model, decoder, z_val, packed_val, P_val, l_max, grid, "val"))
    if P_test is not None:
        metrics.update(_eval_split(model, decoder, z_test, packed_test, P_test, l_max, grid, "test"))
    if holdout_data is not None:
        z_holdout = feat.transform(P=holdout_data["P"])
        metrics.update(_eval_split(
            model, decoder, z_holdout, holdout_data["packed"], holdout_data["P"],
            l_max, grid, "holdout",
        ))

    return {
        "ok": True,
        "elapsed_s": time.time() - started,
        "feat_dim": int(input_dim),
        "metrics": metrics,
        "last_train_loss": float(ctx.last_loss),
    }


def load_holdout(args: argparse.Namespace, grid, l_max: int):
    root = Path(args.holdout_root)
    if not root.exists():
        return None
    cfg = RealAntennaLoaderConfig(
        root=str(root),
        feature_subdir=args.holdout_feature_subdir,
        grid=grid,
        l_max=l_max,
    )
    samples = load_real_antenna(cfg)
    if not samples:
        return None
    return {
        "P": np.stack([s.P for s in samples], axis=0),
        "packed": np.stack([s.packed for s in samples], axis=0),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    grid = GRID_DEFAULT

    logger.info("loading basis (grid=%s, l_max=%d) and pre-generating data", grid, args.l_max)
    gen = _build_generator(args)

    rng_train = np.random.default_rng(args.seed_train)
    rng_val = np.random.default_rng(args.seed_val)
    rng_test = np.random.default_rng(args.seed_test)
    P_train, packed_train = gen.generate_batch(args.n_train, rng_train)
    P_val, packed_val = gen.generate_batch(args.n_val, rng_val)
    if args.n_test > 0:
        P_test, packed_test = gen.generate_batch(args.n_test, rng_test)
    else:
        P_test, packed_test = None, None

    aug_name = "none"
    if args.augmentation:
        spec = json.loads(args.augmentation) if args.augmentation.startswith("{") else {
            "name": args.augmentation
        }
        aug_cfg = build_augmentation(spec)
        if aug_cfg is not None:
            aug_name = spec["name"]
            logger.info("applying augmentation %s to train split", aug_name)
            P_train, packed_train = apply_augmentation(
                P_train, packed_train, cfg=aug_cfg,
                rng=np.random.default_rng(4242),
                basis=gen.basis, l_max=args.l_max,
            )

    holdout_data = load_holdout(args, grid, args.l_max)
    if holdout_data is None:
        logger.warning("real-antenna holdout not found at %s; holdout metrics omitted",
                       args.holdout_root)
    else:
        logger.info("holdout corpus: %d samples", holdout_data["P"].shape[0])

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    losses = [l.strip() for l in args.losses.split(",") if l.strip()]
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cells = list(product(features, models, losses, seeds))
    logger.info("%d cells to run for stage %s (regime=%s, aug=%s)",
                len(cells), args.stage, args.regime, aug_name)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    feature_cache: dict[tuple, Any] = {}
    results: list[dict[str, Any]] = []

    for i, (feat_name, model_name, loss_name, seed) in enumerate(cells, 1):
        logger.info("[%d/%d] cell: feat=%s model=%s loss=%s seed=%d",
                    i, len(cells), feat_name, model_name, loss_name, seed)
        try:
            rec = run_cell(
                P_train, packed_train, P_val, packed_val, P_test, packed_test,
                holdout_data,
                feat_name=feat_name, model_name=model_name, loss_name=loss_name,
                seed=seed, grid=grid, l_max=args.l_max, basis=gen.basis,
                max_epochs=args.max_epochs, batch_size=args.batch_size,
                aug_name=aug_name, feature_cache=feature_cache,
            )
        except Exception as exc:
            logger.exception("cell failed: %s", exc)
            rec = {
                "ok": False, "elapsed_s": 0.0, "metrics": {},
                "error": str(exc),
            }
        rec.update({
            "model": model_name, "loss": loss_name, "features": feat_name,
            "seed": seed, "regime": args.regime, "augmentation": aug_name,
        })
        results.append(rec)
        out.write_text(json.dumps(results, indent=2))
        if rec.get("ok"):
            mab = rec["metrics"].get("report/val/coef_mse_amb_aware", float("nan"))
            logger.info("  -> val/coef_mse_amb_aware=%.6f elapsed=%.1fs", mab, rec["elapsed_s"])
        gc.collect()

    n_ok = sum(1 for r in results if r["ok"])
    logger.info("%s done: %d / %d cells ok", args.stage, n_ok, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
