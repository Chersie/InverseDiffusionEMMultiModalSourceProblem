"""S2 augmentation ablation, in-process.

Picks the top-K cells from S1 by ``report/val/coef_mse_amb_aware`` and runs each
cell against five augmentation conditions: ``none``, ``coef_phase_rotation``,
``coef_additive_noise(sigma=0.05)``, ``field_additive_noise(rel_sigma=0.02)``,
``field_phi_roll``. The base ``(P_train, packed_train)`` is generated **once**
and copies are augmented per condition; ``(P_val, packed_val)`` and
``(P_test, packed_test)`` are not augmented (clean targets).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_inprocess import (  # type: ignore[import-not-found]
    _build_features,
    _build_loss,
    _build_model,
    _eval_split,
)

from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

AUG_SPECS = [
    ("none", None),
    ("coef_phase_rotation", {"name": "coef_phase_rotation"}),
    ("coef_additive_noise", {"name": "coef_additive_noise", "sigma": 0.05}),
    ("field_additive_noise", {"name": "field_additive_noise", "relative_sigma": 0.02}),
    ("field_phi_roll", {"name": "field_phi_roll"}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-results", required=True, type=str)
    p.add_argument("--top-k", default=3, type=int)
    p.add_argument("--metric", default="report/val/coef_mse_amb_aware", type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--seeds", default="0", type=str)
    p.add_argument("--max-epochs", default=15, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--n-train", default=2048, type=int)
    p.add_argument("--n-val", default=512, type=int)
    p.add_argument("--n-test", default=512, type=int)
    p.add_argument("--seed-train", default=1234, type=int)
    p.add_argument("--seed-val", default=5678, type=int)
    p.add_argument("--seed-test", default=9012, type=int)
    p.add_argument("--l-max", default=15, type=int)
    p.add_argument("--regime", default="gaussian", type=str)
    p.add_argument("--holdout-root", default="data/raw/real_antenna", type=str)
    return p.parse_args()


def pick_top_k(s1_results: list[dict], k: int, metric_key: str) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in s1_results:
        if not r.get("ok"):
            continue
        v = r.get("metrics", {}).get(metric_key)
        if v is None:
            continue
        grouped[(r["model"], r["loss"], r["features"])].append(float(v))
    means = sorted(
        ((key, sum(vs) / len(vs)) for key, vs in grouped.items() if vs),
        key=lambda kv: kv[1],
    )
    return [
        {"model": k_[0], "loss": k_[1], "features": k_[2], "mean_metric": v}
        for k_, v in means[:k]
    ]


def run_one(
    P_train, packed_train, P_val, packed_val, P_test, packed_test,
    holdout_data, feat_name, model_name, loss_name, seed,
    grid, l_max, basis, max_epochs, batch_size,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    feat = _build_features(feat_name, grid=grid, l_max=l_max)
    feat.fit(P_train=P_train)
    z_train = feat.transform(P=P_train)
    z_val = feat.transform(P=P_val)
    z_test = feat.transform(P=P_test) if P_test is not None else None
    K = l_max * (l_max + 2)
    model = _build_model(model_name, input_dim=feat.feature_dim, output_dim=4 * K)
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
    trainer.fit(
        model=model, train_loader=train_loader, loss_fn=loss_fn,
        optimiser=optimiser, loss_kind=loss_kind, val_loader=val_loader,
        callbacks=callbacks,
    )
    decoder = DifferentiableMultipoleField(grid=grid, l_max=l_max, basis=basis)
    metrics = {}
    metrics.update(_eval_split(model, decoder, z_val, packed_val, P_val, l_max, grid, "val"))
    if P_test is not None:
        metrics.update(_eval_split(model, decoder, z_test, packed_test, P_test, l_max, grid, "test"))
    if holdout_data is not None:
        z_h = feat.transform(P=holdout_data["P"])
        metrics.update(_eval_split(
            model, decoder, z_h, holdout_data["packed"], holdout_data["P"], l_max, grid, "holdout",
        ))
    return metrics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    s1 = json.loads(Path(args.s1_results).read_text())
    top = pick_top_k(s1, args.top_k, args.metric)
    if not top:
        raise SystemExit("no valid S1 cells")
    logger.info("S2 top %d cells:", len(top))
    for t in top:
        logger.info("  %s/%s/%s -> %s = %.6f", t["model"], t["loss"], t["features"],
                    args.metric, t["mean_metric"])

    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)
    gen_cfg = SyntheticGeneratorConfig(grid=grid, l_max=args.l_max, mode=args.regime)
    gen = SyntheticGenerator(cfg=gen_cfg, basis=basis)

    logger.info("generating base data (n_train=%d, n_val=%d, n_test=%d)",
                args.n_train, args.n_val, args.n_test)
    rng_train = np.random.default_rng(args.seed_train)
    rng_val = np.random.default_rng(args.seed_val)
    rng_test = np.random.default_rng(args.seed_test)
    P_train_base, packed_train_base = gen.generate_batch(args.n_train, rng_train)
    P_val, packed_val = gen.generate_batch(args.n_val, rng_val)
    P_test, packed_test = (gen.generate_batch(args.n_test, rng_test)
                           if args.n_test > 0 else (None, None))

    holdout_data = None
    if Path(args.holdout_root).exists():
        try:
            cfg = RealAntennaLoaderConfig(
                root=args.holdout_root, grid=grid, l_max=args.l_max,
            )
            samples = load_real_antenna(cfg)
            if samples:
                holdout_data = {
                    "P": np.stack([s.P for s in samples], axis=0),
                    "packed": np.stack([s.packed for s in samples], axis=0),
                }
                logger.info("holdout: %d samples", holdout_data["P"].shape[0])
        except Exception as exc:
            logger.warning("holdout load failed: %s", exc)
    if holdout_data is None:
        logger.warning("real-antenna holdout not available; holdout metrics will be omitted")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    cells = list(product(top, AUG_SPECS, seeds))
    total = len(cells)
    logger.info("S2 will run %d cells (%d top * %d augs * %d seeds)",
                total, len(top), len(AUG_SPECS), len(seeds))

    for i, (cell, (aug_name, aug_spec), seed) in enumerate(cells, 1):
        logger.info("[%d/%d] cell=%s/%s/%s aug=%s seed=%d",
                    i, total, cell["model"], cell["loss"], cell["features"], aug_name, seed)
        # Apply augmentation to a fresh copy of the train arrays
        P_t = P_train_base.copy()
        pk_t = packed_train_base.copy()
        if aug_spec is not None:
            aug_cfg = build_augmentation(aug_spec)
            P_t, pk_t = apply_augmentation(
                P_t, pk_t, cfg=aug_cfg,
                rng=np.random.default_rng(4242 + seed),
                basis=basis, l_max=args.l_max,
            )
        started = time.time()
        try:
            metrics = run_one(
                P_t, pk_t, P_val, packed_val, P_test, packed_test,
                holdout_data,
                feat_name=cell["features"], model_name=cell["model"], loss_name=cell["loss"],
                seed=seed, grid=grid, l_max=args.l_max, basis=basis,
                max_epochs=args.max_epochs, batch_size=args.batch_size,
            )
            rec = {
                "ok": True, "elapsed_s": time.time() - started, "metrics": metrics,
                "model": cell["model"], "loss": cell["loss"], "features": cell["features"],
                "augmentation": aug_name, "seed": seed, "regime": args.regime,
            }
        except Exception as exc:
            logger.exception("cell failed: %s", exc)
            rec = {
                "ok": False, "elapsed_s": time.time() - started,
                "model": cell["model"], "loss": cell["loss"], "features": cell["features"],
                "augmentation": aug_name, "seed": seed, "regime": args.regime,
                "error": str(exc),
            }
        results.append(rec)
        out_path.write_text(json.dumps(results, indent=2))
        if rec.get("ok"):
            mab = rec["metrics"].get("report/val/coef_mse_amb_aware", float("nan"))
            logger.info("  -> val/coef_mse_amb_aware=%.6f elapsed=%.1fs", mab, rec["elapsed_s"])
        del P_t, pk_t
        gc.collect()

    n_ok = sum(1 for r in results if r["ok"])
    logger.info("S2 done: %d / %d cells ok", n_ok, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
