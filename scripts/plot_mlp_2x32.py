"""Generate the standard figure suite for the `mlp_2x32` baseline cell.

Replays the S4 experiment (`mlp_2x32` + `coef_mse` + `raw_flat` +
`field_additive_noise`) across the four generation regimes, with one seed per
regime, and emits

- the framework's standard report (``coef_histograms``, ``coef_scatter``,
  ``per_l_breakdown``, ``field_comparison``) on ``val`` **and** ``test`` splits,
- a per-regime training-loss curve, captured via a list-collecting sink,
- two cross-regime comparison bar charts (`mlp_2x32` vs `mlp_2x512` from
  ``S3_results.json``) on ``val/coef_mse_amb_aware`` and ``val/field_nrmse_w``.

Output layout::

    experiments/baseline/figures_mlp_2x32/
        <regime>/val/{coef_histograms,coef_scatter,per_l_breakdown,field_comparison}.pdf
        <regime>/test/...
        <regime>/loss_curves.pdf
        cross_regime/amb_aware_vs_mlp_2x512.pdf
        cross_regime/field_nrmse_vs_mlp_2x512.pdf
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_inprocess import (  # type: ignore[import-not-found]
    _build_features,
    _build_loss,
    _build_model,
)

from mpinv.analysis.plots.loss_curves import build_loss_curves_figure
from mpinv.analysis.reports.run_report import RunArtifacts, build_run_report
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import _ArrayDataset
from mpinv.core.grid import GRID_DEFAULT
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.augment import apply_augmentation, build_augmentation
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


REGIMES: dict[str, dict[str, Any]] = {
    "gaussian": {"mode": "gaussian"},
    "colored_a1": {"mode": "colored", "color_alpha": 1.0},
    "colored_a2": {"mode": "colored", "color_alpha": 2.0},
    "sparse_p10": {"mode": "sparse", "sparse_active_fraction": 0.1},
}


REGIME_PRETTY = {
    "gaussian": "Gaussian",
    "colored_a1": "Colored α=1",
    "colored_a2": "Colored α=2",
    "sparse_p10": "Sparse 10%",
}


class _HistorySink:
    """A trainer sink that just records ``(step, value)`` pairs per metric.

    Implements the minimal sink protocol expected by ``Trainer.fit``:
    lifecycle hooks (``on_fit_start``, ``on_epoch_end``, ``on_run_end``) plus
    ``log_metrics`` / ``log_metric``.
    """

    def __init__(self) -> None:
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        for k, v in metrics.items():
            try:
                self.history[k].append((int(step), float(v)))
            except (TypeError, ValueError):
                continue

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        try:
            self.history[key].append((int(step), float(value)))
        except (TypeError, ValueError):
            pass

    def on_fit_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
        return None

    def on_epoch_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        if ctx.last_eval_metrics:
            for k, v in ctx.last_eval_metrics.items():
                try:
                    self.history[k].append((int(ctx.global_step), float(v)))
                except (TypeError, ValueError):
                    continue

    def on_run_end(self, status: str) -> None:  # type: ignore[no-untyped-def]
        return None


def _predict(model: torch.nn.Module, decoder: DifferentiableMultipoleField,
             z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(z).float()).cpu().numpy()
        P_pred = decoder(torch.from_numpy(preds).float()).cpu().numpy()
    return preds, P_pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="experiments/baseline/figures_mlp_2x32",
                   type=str)
    p.add_argument("--regimes", default="gaussian,colored_a1,colored_a2,sparse_p10",
                   type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--max-epochs", default=30, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--n-train", default=4096, type=int)
    p.add_argument("--n-val", default=1024, type=int)
    p.add_argument("--n-test", default=1024, type=int)
    p.add_argument("--seed-train", default=1234, type=int)
    p.add_argument("--seed-val", default=5678, type=int)
    p.add_argument("--seed-test", default=9012, type=int)
    p.add_argument("--l-max", default=5, type=int)
    p.add_argument("--s3-results", default="experiments/baseline/S3_results.json",
                   type=str)
    p.add_argument("--s4-results",
                   default="experiments/baseline/S4_mlp_2x32_results.json", type=str)
    return p.parse_args()


def run_regime(
    regime_name: str,
    regime_kwargs: dict[str, Any],
    args: argparse.Namespace,
    grid,
    basis,
    out_root: Path,
) -> dict[str, Any]:
    """Train, predict, build standard report for one regime. Returns metrics."""
    model_name = "mlp_2x32"
    feat_name = "raw_flat"
    loss_name = "coef_mse"
    aug_spec = {"name": "field_additive_noise", "relative_sigma": 0.02}

    out_regime = out_root / regime_name
    out_regime.mkdir(parents=True, exist_ok=True)

    gen_cfg = SyntheticGeneratorConfig(grid=grid, l_max=args.l_max, **regime_kwargs)
    gen = SyntheticGenerator(cfg=gen_cfg, basis=basis)
    rng_train = np.random.default_rng(args.seed_train)
    rng_val = np.random.default_rng(args.seed_val)
    rng_test = np.random.default_rng(args.seed_test)
    P_train, packed_train = gen.generate_batch(args.n_train, rng_train)
    P_val, packed_val = gen.generate_batch(args.n_val, rng_val)
    P_test, packed_test = gen.generate_batch(args.n_test, rng_test)

    aug_cfg = build_augmentation(aug_spec)
    if aug_cfg is not None:
        P_train, packed_train = apply_augmentation(
            P_train, packed_train, cfg=aug_cfg,
            rng=np.random.default_rng(4242 + args.seed),
            basis=basis, l_max=args.l_max,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feat = _build_features(feat_name, grid=grid, l_max=args.l_max)
    feat.fit(P_train=P_train)
    z_train = feat.transform(P=P_train)
    z_val = feat.transform(P=P_val)
    z_test = feat.transform(P=P_test)

    K = args.l_max * (args.l_max + 2)
    model = _build_model(model_name, input_dim=feat.feature_dim, output_dim=4 * K)
    loss_fn, loss_kind = _build_loss(loss_name, grid=grid, l_max=args.l_max)
    optimiser = build_optimiser(model, OptimiserConfig(name="adamw", lr=1e-3))

    train_ds = _ArrayDataset(z_train, packed_train, P_train)
    val_ds = _ArrayDataset(z_val, packed_val, P_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    history = _HistorySink()
    trainer = Trainer(TrainerConfig(max_epochs=args.max_epochs, log_every_n_steps=10))
    callbacks = [
        LoggingCallback(log_every_n_steps=10),
        ValidationCallback(every_n_epochs=1),
        GradClipCallback(max_norm=1.0),
        EarlyStoppingCallback(patience=10),
    ]
    started = time.time()
    trainer.fit(
        model=model, train_loader=train_loader, loss_fn=loss_fn,
        optimiser=optimiser, loss_kind=loss_kind, val_loader=val_loader,
        callbacks=callbacks, sinks=[history],
    )
    elapsed = time.time() - started
    logger.info("regime=%s elapsed=%.1fs", regime_name, elapsed)

    decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)

    val_preds, P_val_pred = _predict(model, decoder, z_val)
    test_preds, P_test_pred = _predict(model, decoder, z_test)

    val_metrics = build_run_report(
        RunArtifacts(
            pred_packed=val_preds, target_packed=packed_val,
            P_pred=P_val_pred, P_true=P_val, l_max=args.l_max, grid=grid,
        ),
        output_dir=out_regime / "val",
    )
    test_metrics = build_run_report(
        RunArtifacts(
            pred_packed=test_preds, target_packed=packed_test,
            P_pred=P_test_pred, P_true=P_test, l_max=args.l_max, grid=grid,
        ),
        output_dir=out_regime / "test",
    )

    curve_keys = [k for k in history.history.keys() if "loss" in k.lower()]
    pruned = {k: history.history[k] for k in curve_keys}
    if pruned:
        fig = build_loss_curves_figure(
            pruned, title=f"mlp_2x32 — {REGIME_PRETTY[regime_name]}", log_y=True
        )
        fig.savefig(out_regime / "loss_curves.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        logger.warning("no loss-bearing keys in sink history; loss_curves skipped")

    return {
        "regime": regime_name,
        "elapsed_s": elapsed,
        "val": {f"report/val/{k.split('/', 1)[1] if '/' in k else k}": v
                for k, v in val_metrics.items()},
        "test": {f"report/test/{k.split('/', 1)[1] if '/' in k else k}": v
                 for k, v in test_metrics.items()},
    }


def _mean_metric_per_regime(records: list[dict[str, Any]],
                            metric_key: str) -> dict[str, float]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if not r.get("ok"):
            continue
        v = r.get("metrics", {}).get(metric_key)
        if v is None:
            continue
        out[r["regime"]].append(float(v))
    return {r: float(np.mean(vs)) for r, vs in out.items() if vs}


def cross_regime_plots(s3_results_path: Path, s4_results_path: Path,
                       out_dir: Path) -> None:
    s3 = json.loads(s3_results_path.read_text()) if s3_results_path.exists() else []
    s4 = json.loads(s4_results_path.read_text()) if s4_results_path.exists() else []
    if not s3 and not s4:
        logger.warning("S3 and S4 results both missing; skipping cross-regime plots")
        return

    regimes = list(REGIMES.keys())

    for metric_key, ylabel, fname, log_y in [
        ("report/val/coef_mse_amb_aware",
         "val/coef_mse_amb_aware (lower better)",
         "amb_aware_vs_mlp_2x512.pdf", True),
        ("report/val/field_nrmse_w",
         "val/field_nrmse_w (lower better; 1.0 = predicting zero)",
         "field_nrmse_vs_mlp_2x512.pdf", False),
    ]:
        m_2x32 = _mean_metric_per_regime(s4, metric_key)
        m_2x512 = _mean_metric_per_regime(s3, metric_key)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(regimes))
        width = 0.36
        vals_2x32 = [m_2x32.get(r, np.nan) for r in regimes]
        vals_2x512 = [m_2x512.get(r, np.nan) for r in regimes]
        bars1 = ax.bar(x - width / 2, vals_2x32, width, label="mlp_2x32",
                       color="#3a86ff")
        bars2 = ax.bar(x + width / 2, vals_2x512, width, label="mlp_2x512 (S3)",
                       color="#fb5607")
        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3g}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_PRETTY[r] for r in regimes])
        ax.set_ylabel(ylabel)
        ax.set_title(f"mlp_2x32 vs mlp_2x512 — {metric_key}")
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    regime_names = [r.strip() for r in args.regimes.split(",") if r.strip()]
    summary: list[dict[str, Any]] = []
    for r in regime_names:
        if r not in REGIMES:
            logger.warning("unknown regime %s; skipping", r)
            continue
        logger.info("=== regime: %s ===", r)
        rec = run_regime(r, REGIMES[r], args, grid=grid, basis=basis, out_root=out_root)
        summary.append(rec)
        gc.collect()

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    cross_regime_plots(
        s3_results_path=Path(args.s3_results),
        s4_results_path=Path(args.s4_results),
        out_dir=out_root / "cross_regime",
    )

    logger.info("all figures written under %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
