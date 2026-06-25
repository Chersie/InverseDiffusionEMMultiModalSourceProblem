"""S3 generation-regime ablation, in-process.

Picks the top-1 cell from S2 (augmentation included) by
``report/val/coef_mse_amb_aware`` and replays it across the four generation
regimes: gaussian, colored alpha=1, colored alpha=2, sparse 10%.

Each regime regenerates train/val/test data; the augmentation (if any) is
re-applied to the train split only.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_inprocess import (  # type: ignore[import-not-found]
    _build_features,
    _build_loss,
    _build_model,
    _eval_split,
)

from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import _ArrayDataset
from mpinv.core.grid import GRID_DEFAULT
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.augment import apply_augmentation, build_augmentation
from mpinv.data.real_antenna_loader import RealAntennaLoaderConfig, load_real_antenna
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


REGIMES = {
    "gaussian": dict(mode="gaussian"),
    "colored_a1": dict(mode="colored", color_alpha=1.0),
    "colored_a2": dict(mode="colored", color_alpha=2.0),
    "sparse_p10": dict(mode="sparse", sparse_active_fraction=0.1),
}


AUG_BUILDERS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "coef_phase_rotation": {"name": "coef_phase_rotation"},
    "coef_additive_noise": {"name": "coef_additive_noise", "sigma": 0.05},
    "field_additive_noise": {"name": "field_additive_noise", "relative_sigma": 0.02},
    "field_phi_roll": {"name": "field_phi_roll"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s2-results", required=True, type=str)
    p.add_argument("--metric", default="report/val/coef_mse_amb_aware", type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--seeds", default="0,1", type=str)
    p.add_argument("--max-epochs", default=30, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--n-train", default=4096, type=int)
    p.add_argument("--n-val", default=1024, type=int)
    p.add_argument("--n-test", default=1024, type=int)
    p.add_argument("--seed-train", default=1234, type=int)
    p.add_argument("--seed-val", default=5678, type=int)
    p.add_argument("--seed-test", default=9012, type=int)
    p.add_argument("--l-max", default=5, type=int)
    p.add_argument("--holdout-root", default="data/raw/real_antenna", type=str)
    # Optional overrides on the cell auto-picked from S2 results.
    p.add_argument("--override-model", default=None, type=str,
                   help="If set, replace the picked cell's `model` with this name.")
    p.add_argument("--override-loss", default=None, type=str,
                   help="If set, replace the picked cell's `loss` with this name.")
    p.add_argument("--override-features", default=None, type=str,
                   help="If set, replace the picked cell's `features` with this name.")
    p.add_argument("--override-augmentation", default=None, type=str,
                   help="If set, replace the picked cell's `augmentation` with this name "
                        "(use 'none' to disable).")
    return p.parse_args()


def pick_top_1(s2: list[dict], metric_key: str) -> dict:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in s2:
        if not r.get("ok"):
            continue
        v = r.get("metrics", {}).get(metric_key)
        if v is None:
            continue
        grouped[(r["model"], r["loss"], r["features"], r["augmentation"])].append(float(v))
    if not grouped:
        raise RuntimeError("no valid S2 cells")
    best_key, best_vals = min(grouped.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    return {
        "model": best_key[0], "loss": best_key[1], "features": best_key[2],
        "augmentation": best_key[3],
        "mean_metric": sum(best_vals) / len(best_vals),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    s2 = json.loads(Path(args.s2_results).read_text())
    cell = pick_top_1(s2, args.metric)
    logger.info("S3 best-from-S2 cell: %s", cell)
    if args.override_model is not None:
        cell["model"] = args.override_model
    if args.override_loss is not None:
        cell["loss"] = args.override_loss
    if args.override_features is not None:
        cell["features"] = args.override_features
    if args.override_augmentation is not None:
        cell["augmentation"] = args.override_augmentation
    if any(getattr(args, k) is not None for k in
           ("override_model", "override_loss", "override_features", "override_augmentation")):
        logger.info("S3 cell after overrides: %s", cell)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)

    holdout_data = None
    if Path(args.holdout_root).exists():
        try:
            cfg = RealAntennaLoaderConfig(root=args.holdout_root, grid=grid, l_max=args.l_max)
            samples = load_real_antenna(cfg)
            if samples:
                holdout_data = {
                    "P": np.stack([s.P for s in samples], axis=0),
                    "packed": np.stack([s.packed for s in samples], axis=0),
                }
        except Exception as exc:
            logger.warning("holdout load failed: %s", exc)
    if holdout_data is None:
        logger.warning("real-antenna holdout not available; holdout metrics omitted")

    aug_spec = AUG_BUILDERS.get(cell["augmentation"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(REGIMES) * len(seeds)
    i = 0
    for regime_name, regime_kwargs in REGIMES.items():
        logger.info("=== regime: %s (%s) ===", regime_name, regime_kwargs)
        gen_cfg = SyntheticGeneratorConfig(
            grid=grid, l_max=args.l_max, **regime_kwargs
        )
        gen = SyntheticGenerator(cfg=gen_cfg, basis=basis)
        rng_train = np.random.default_rng(args.seed_train)
        rng_val = np.random.default_rng(args.seed_val)
        rng_test = np.random.default_rng(args.seed_test)
        P_train_base, packed_train_base = gen.generate_batch(args.n_train, rng_train)
        P_val, packed_val = gen.generate_batch(args.n_val, rng_val)
        P_test, packed_test = gen.generate_batch(args.n_test, rng_test)

        for seed in seeds:
            i += 1
            logger.info("[%d/%d] regime=%s seed=%d", i, total, regime_name, seed)
            P_t = P_train_base.copy()
            pk_t = packed_train_base.copy()
            if aug_spec is not None:
                aug_cfg = build_augmentation(aug_spec)
                P_t, pk_t = apply_augmentation(
                    P_t, pk_t, cfg=aug_cfg,
                    rng=np.random.default_rng(4242 + seed),
                    basis=basis, l_max=args.l_max,
                )

            torch.manual_seed(seed)
            np.random.seed(seed)
            feat = _build_features(cell["features"], grid=grid, l_max=args.l_max)
            feat.fit(P_train=P_t)
            z_train = feat.transform(P=P_t)
            z_val = feat.transform(P=P_val)
            z_test = feat.transform(P=P_test)

            K = args.l_max * (args.l_max + 2)
            model = _build_model(cell["model"], input_dim=feat.feature_dim, output_dim=4 * K)
            loss_fn, loss_kind = _build_loss(cell["loss"], grid=grid, l_max=args.l_max)
            optimiser = build_optimiser(model, OptimiserConfig(name="adamw", lr=1e-3))
            train_ds = _ArrayDataset(z_train, pk_t, P_t)
            val_ds = _ArrayDataset(z_val, packed_val, P_val)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            trainer = Trainer(TrainerConfig(max_epochs=args.max_epochs, log_every_n_steps=200))
            callbacks = [
                LoggingCallback(log_every_n_steps=200),
                ValidationCallback(every_n_epochs=1),
                GradClipCallback(max_norm=1.0),
                EarlyStoppingCallback(patience=10),
            ]
            started = time.time()
            try:
                trainer.fit(
                    model=model, train_loader=train_loader, loss_fn=loss_fn,
                    optimiser=optimiser, loss_kind=loss_kind, val_loader=val_loader,
                    callbacks=callbacks,
                )
                decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)
                metrics = _eval_split(model, decoder, z_val, packed_val, P_val, args.l_max, grid, "val")
                metrics.update(_eval_split(model, decoder, z_test, packed_test, P_test, args.l_max, grid, "test"))
                if holdout_data is not None:
                    z_h = feat.transform(P=holdout_data["P"])
                    metrics.update(_eval_split(
                        model, decoder, z_h, holdout_data["packed"], holdout_data["P"],
                        args.l_max, grid, "holdout",
                    ))
                rec = {
                    "ok": True, "elapsed_s": time.time() - started, "metrics": metrics,
                    **{k: cell[k] for k in ("model", "loss", "features", "augmentation")},
                    "regime": regime_name, "seed": seed,
                }
            except Exception as exc:
                logger.exception("cell failed: %s", exc)
                rec = {
                    "ok": False, "elapsed_s": time.time() - started,
                    **{k: cell[k] for k in ("model", "loss", "features", "augmentation")},
                    "regime": regime_name, "seed": seed,
                    "error": str(exc),
                }
            results.append(rec)
            out_path.write_text(json.dumps(results, indent=2))
            if rec.get("ok"):
                m = rec["metrics"].get("report/val/coef_mse_amb_aware", float("nan"))
                logger.info("  -> val/amb=%.6f elapsed=%.1fs", m, rec["elapsed_s"])
            del P_t, pk_t
            gc.collect()
        del gen, P_train_base, packed_train_base, P_val, packed_val, P_test, packed_test
        gc.collect()
    n_ok = sum(1 for r in results if r["ok"])
    logger.info("S3 done: %d / %d cells ok", n_ok, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
