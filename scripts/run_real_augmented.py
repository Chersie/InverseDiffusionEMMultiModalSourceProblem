"""Real-augmented experiment: limited holdout + on-manifold augmentation.

Pipeline:
1. Load up to ``--n-source`` real-antenna samples from ``--holdout-root``
   (default ``data/raw/real_antenna``) via :class:`RealAntennaLoaderConfig`.
2. Truncate the target coefficients to ``l_max`` and **re-synthesise** the
   power pattern from the truncated coefficients so ``(P, packed)`` is
   consistent on the bandlimit-`l_max` manifold. This drops any `l > l_max`
   content from the original measurement; that is a deliberate fidelity /
   runtime trade-off, recorded as honest gap #3 in
   ``research/baseline-experiments/manifest.md`` R7.
3. Split the sources **by sample id** into train (default ``--n-train-sources
   80``) and val (the remaining sources). No augmented copy of a train source
   ever appears in val.
4. Augment the train sources to ``--n-augmented`` total samples by composing
   ``field_phi_roll → coef_mode_dropout → field_additive_noise``. Each step is
   applied independently per augmented sample, so the resulting `(P, packed)`
   pairs span the on-manifold orbit of the train sources.
5. Train ``mlp_2x32 / coef_mse / raw_flat`` (the S2-winner config) for
   ``--max-epochs`` epochs.
6. Evaluate on three splits: ``train_aug`` (sanity check), ``val_real`` (the
   primary signal — unaugmented, sample-id-disjoint), and optionally
   ``synthetic_test`` (a synthetic colored α=2 split as a distribution
   reference; emitted iff ``--include-synthetic-test`` is set).
7. Write per-cell metrics to ``--output``, the standard 4-PDF figure suite for
   each evaluated split to ``--figures-dir`` (skipped if ``--no-figures``).

Usage::

    uv run python scripts/run_real_augmented.py \\
        --holdout-root data/raw/real_antenna \\
        --output experiments/baseline/S5_real_augmented_results.json \\
        --figures-dir experiments/baseline/figures_real_augmented

Smoke-test mode: pass ``--smoke-test`` to substitute synthetic samples for the
holdout corpus when ``data/raw/real_antenna`` is empty / unavailable. This is
for code-path validation only; results from smoke mode are NOT meaningful.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; figures are saved, never shown
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

from mpinv.analysis.metrics.coefficient_metrics import packed_mse, packed_r2
from mpinv.analysis.metrics.field_metrics import (
    bin_accuracy_P,
    bin_within_k_accuracy_P,
    hard_rank_bin_mse_P,
    per_sample_bin_accuracy_P,
    per_sample_weighted_r2_P,
    spearman_rho_P,
    weighted_mse_P,
    weighted_nrmse_P,
    weighted_r2_P,
)
from mpinv.analysis.metrics.mode_metrics import reflected_conjugate_aware_loss
from mpinv.analysis.plots.field_comparison import build_field_comparison_grid_figure
from mpinv.analysis.plots.r2_distribution import (
    build_bin_accuracy_distribution_figure,
    build_r2_distribution_figure,
)
from mpinv.analysis.reports.run_report import RunArtifacts, build_run_report
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import _ArrayDataset
from mpinv.core.grid import GRID_DEFAULT
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.real_antenna_loader import (
    RealAntennaLoaderConfig,
    list_real_antenna_samples,
    load_real_antenna,
)
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.training.optim import (
    OptimiserConfig,
    SchedulerConfig,
    build_optimiser,
    build_scheduler,
)
from mpinv.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-root", default="data/raw/real_antenna", type=str)
    p.add_argument("--feature-subdir", default="E_in_plane", type=str)
    p.add_argument("--n-source", default=100, type=int,
                   help="Maximum number of real-antenna samples to load.")
    p.add_argument("--n-train-sources", default=80, type=int,
                   help="Sample-id-level train split size; the remainder is val.")
    p.add_argument("--n-augmented", default=10000, type=int,
                   help="Total number of augmented training samples to build.")
    p.add_argument("--shuffle-seed", default=42, type=int,
                   help="Deterministic shuffle of the source files before splitting.")
    p.add_argument("--aug-seed", default=4242, type=int,
                   help="Seed for the augmentation RNG.")
    p.add_argument("--seed", default=0, type=int,
                   help="Seed for model init / data loaders / augmentation order.")
    p.add_argument("--l-max", default=5, type=int)
    p.add_argument("--dropout-prob", default=0.1, type=float,
                   help="Per-mode dropout probability for coef_mode_dropout.")
    p.add_argument("--field-sigma", default=1e-8, type=float,
                   help="relative_sigma for field_additive_noise on the train inputs.")
    p.add_argument("--scale-factor", default=1.0, type=float,
                   help="Multiplicative scale applied to all real-antenna P "
                        "values (and sqrt(scale) to packed coefficients) on "
                        "every split (train/val/holdout) AND the synthetic "
                        "test split, for unit consistency. P scales as |E|^2 "
                        "and packed scales as |E|, so the |E|^2 = scale * |E|^2 "
                        "contract requires packed -> sqrt(scale) * packed. "
                        "Use to lift very small real-antenna magnitudes "
                        "(O(1e-6) on this corpus) into a numerically friendlier "
                        "regime. 1.0 = no scaling.")
    p.add_argument("--aug-chunk-size", default=500, type=int,
                   help="Chunk size for the coefficient-space augmentation "
                        "re-synthesis step. Lower values reduce peak memory at "
                        "small throughput cost. ~500 ~= 500 MB peak per family "
                        "on the 360x179 grid.")
    p.add_argument("--max-epochs", default=30, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--early-stop-patience", default=10, type=int)
    p.add_argument("--model", default="mlp_2x64", type=str)
    p.add_argument("--loss", default="coef_mse", type=str)
    p.add_argument("--features", default="raw_flat", type=str)
    # Optimiser knobs.
    p.add_argument("--optimiser", default="adamw",
                   choices=["adamw", "adam", "sgd"],
                   help="Optimiser. SGD pairs naturally with --momentum and "
                        "--nesterov; AdamW uses its own (beta1, beta2) so "
                        "those flags are ignored.")
    p.add_argument("--momentum", default=0.9, type=float,
                   help="Momentum coefficient for SGD (ignored for adam/adamw).")
    p.add_argument("--nesterov", action="store_true",
                   help="Enable Nesterov-accelerated SGD. No effect on Adam/AdamW.")
    p.add_argument("--weight-decay", default=0.0, type=float,
                   help="L2 weight decay (decoupled in AdamW; standard L2 in SGD).")
    # Adaptive learning-rate schedule.
    p.add_argument("--scheduler", default="none",
                   choices=["none", "plateau", "cosine", "cosine_with_warmup", "step"],
                   help="LR schedule. 'plateau' is the canonical 'adaptive LR' "
                        "for SGD: halves LR (factor=0.5) when val/loss "
                        "plateaus for --scheduler-plateau-patience epochs.")
    p.add_argument("--scheduler-min-lr", default=1e-6, type=float)
    p.add_argument("--scheduler-warmup-steps", default=50, type=int)
    p.add_argument("--scheduler-plateau-patience", default=5, type=int)
    p.add_argument("--scheduler-plateau-factor", default=0.5, type=float)
    p.add_argument("--scheduler-step-size", default=10, type=int)
    p.add_argument("--scheduler-gamma", default=0.1, type=float)
    # Checkpointing knobs.
    p.add_argument("--checkpoint-every-n-epochs", default=10, type=int,
                   help="Save model + optimiser + scheduler state every N "
                        "epochs into <checkpoint-dir>/epoch_NNNN.pt. Also "
                        "saves best.pt (best val/loss seen so far) and "
                        "last.pt (final state). 0 disables checkpointing "
                        "entirely (and skips writing the feature extractor "
                        "pickle / manifest.json).")
    p.add_argument("--checkpoint-dir", default=None, type=str,
                   help="Directory for model checkpoints + feature extractor "
                        "pickle + manifest.json. Defaults to "
                        "<figures-dir>/checkpoints/ when --figures-dir is "
                        "set, else <output>.parent/<output_stem>_checkpoints/.")
    p.add_argument("--keep-last-checkpoints", default=3, type=int,
                   help="Keep at most this many epoch_NNNN.pt files; older "
                        "ones are pruned automatically. best.pt and last.pt "
                        "are always retained.")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--figures-dir", default=None, type=str)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--include-synthetic-test", action="store_true",
                   help="Also evaluate on a colored \u03b1=2 synthetic test split.")
    p.add_argument("--n-synthetic-test", default=512, type=int)
    p.add_argument("--seed-synthetic-test", default=9012, type=int)
    p.add_argument("--eval-batch-size", default=256, type=int,
                   help="Batch size for inference on val / train_aug / synthetic_test.")
    p.add_argument("--n-train-eval-samples", default=1024, type=int,
                   help="Random subsample of the augmented train set used for the "
                        "train_aug sanity-check eval (full train set is too large to "
                        "forward in one go).")
    p.add_argument("--n-figure-samples", default=64, type=int,
                   help="Cap the number of samples used for figure rendering per "
                        "split. Reduces memory cost of P_pred (each sample is "
                        "n_theta\u00d7n_phi floats).")
    p.add_argument("--n-figure-grid-samples", default=8, type=int,
                   help="Number of (P_true, P_pred, residual) rows to draw in "
                        "the per-split field_comparison_grid.pdf figure.")
    p.add_argument("--figures-every-n-epochs", default=0, type=int,
                   help="If > 0, emit the full per-split figure suite "
                        "(coef_*.pdf, field_comparison_grid.pdf, "
                        "r2_distribution.pdf, bin_accuracy_distribution.pdf) "
                        "every N epochs of training, into "
                        "<figures_dir>/epoch_<NNNN>/. End-of-training "
                        "figures are still always emitted in the parent "
                        "<figures_dir>/. 0 = end of training only (default).")
    p.add_argument("--n-holdout-samples", default=100, type=int,
                   help="Real-antenna samples held out from train+val (sample-id "
                        "disjoint, never augmented, never seen in training). Set "
                        "to 0 to disable the holdout split.")
    p.add_argument("--holdout-shuffle-seed", default=314159, type=int,
                   help="Seed for selecting holdout sample ids from the pool of "
                        "real-antenna samples not in train+val.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Substitute synthetic samples for the holdout corpus. "
                        "Code-path check only \u2014 results are not meaningful.")
    p.add_argument("--cache-dir", default="data/cache/real_augmented", type=str,
                   help="Directory for caching the augmented (P_aug, pk_aug) "
                        "+ (P_val, pk_val) artifacts. Cache key is derived from "
                        "the source sample IDs and all augmentation parameters.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable cache read/write entirely (always rebuild).")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Ignore any existing cache file and rebuild + overwrite.")
    return p.parse_args()


def _truncate_and_resynthesise(
    P_orig: np.ndarray,
    packed: np.ndarray,
    *,
    basis,
) -> np.ndarray:
    """Backwards-compatible alias for :func:`mpinv.data.real_augmented_pipeline.truncate_and_resynthesise`."""
    from mpinv.data.real_augmented_pipeline import truncate_and_resynthesise

    return truncate_and_resynthesise(P_orig, packed, basis=basis)


def _load_real(
    args: argparse.Namespace, grid, l_max: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Backwards-compatible alias for :func:`mpinv.data.real_augmented_pipeline.load_real`."""
    from mpinv.data.real_augmented_pipeline import load_real

    return load_real(
        holdout_root=args.holdout_root,
        feature_subdir=args.feature_subdir,
        grid=grid,
        l_max=l_max,
        shuffle_seed=args.shuffle_seed,
        n_source=args.n_source,
    )


def _load_smoke(
    args: argparse.Namespace, grid, l_max: int, basis
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Backwards-compatible alias for :func:`mpinv.data.real_augmented_pipeline.load_smoke`."""
    from mpinv.data.real_augmented_pipeline import load_smoke

    return load_smoke(
        grid=grid,
        l_max=l_max,
        basis=basis,
        n_source=args.n_source,
        shuffle_seed=args.shuffle_seed,
    )


def _build_augmented(
    P_src: np.ndarray,
    packed_src: np.ndarray,
    *,
    n_augmented: int,
    dropout_prob: float,
    field_sigma: float,
    l_max: int,
    basis,
    rng: np.random.Generator,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible alias for :func:`mpinv.data.real_augmented_pipeline.build_augmented`."""
    from mpinv.data.real_augmented_pipeline import build_augmented

    return build_augmented(
        P_src,
        packed_src,
        n_augmented=n_augmented,
        dropout_prob=dropout_prob,
        field_sigma=field_sigma,
        l_max=l_max,
        basis=basis,
        rng=rng,
        chunk_size=chunk_size,
    )


def _maybe_synthetic_test(
    args: argparse.Namespace, grid, l_max: int, basis
) -> tuple[np.ndarray, np.ndarray] | None:
    if not args.include_synthetic_test:
        return None
    cfg = SyntheticGeneratorConfig(grid=grid, l_max=l_max, mode="colored",
                                   color_alpha=2.0)
    gen = SyntheticGenerator(cfg=cfg, basis=basis)
    rng = np.random.default_rng(args.seed_synthetic_test)
    P, packed = gen.generate_batch(args.n_synthetic_test, rng)
    return P, packed


def _predict_chunked(
    model: torch.nn.Module,
    decoder: DifferentiableMultipoleField | None,
    z: np.ndarray,
    *,
    batch_size: int,
    want_field: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Forward ``model`` (and optionally ``decoder``) on ``z`` in batches.

    Returns ``(packed_preds, P_pred)``. ``P_pred`` is ``None`` when
    ``want_field`` is ``False`` or no decoder is supplied.
    """
    n = z.shape[0]
    preds_list: list[np.ndarray] = []
    P_list: list[np.ndarray] = [] if (want_field and decoder is not None) else []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            zb = torch.from_numpy(z[i : i + batch_size]).float()
            pb = model(zb).cpu().numpy()
            preds_list.append(pb)
            if want_field and decoder is not None:
                pp = decoder(torch.from_numpy(pb).float()).cpu().numpy()
                P_list.append(pp)
    preds = np.concatenate(preds_list, axis=0)
    P_pred = np.concatenate(P_list, axis=0) if P_list else None
    return preds, P_pred


def _eval_split_chunked(
    model: torch.nn.Module,
    decoder: DifferentiableMultipoleField,
    z: np.ndarray,
    packed: np.ndarray,
    P: np.ndarray,
    *,
    l_max: int,
    grid,
    tag: str,
    batch_size: int,
) -> dict[str, float]:
    preds, P_pred = _predict_chunked(model, decoder, z, batch_size=batch_size,
                                     want_field=True)
    if P_pred is None:
        raise RuntimeError("decoder forward returned no field")
    # Rank-based eval companions of the soft rank-bin training loss.
    n_bins = 2 * l_max + 1
    return {
        f"report/{tag}/coef_mse": packed_mse(preds, packed),
        f"report/{tag}/coef_r2": packed_r2(preds, packed),
        f"report/{tag}/coef_mse_amb_aware":
            reflected_conjugate_aware_loss(preds, packed, l_max),
        f"report/{tag}/field_mse_w": weighted_mse_P(P_pred, P, grid=grid),
        f"report/{tag}/field_nrmse_w": weighted_nrmse_P(P_pred, P, grid=grid),
        f"report/{tag}/field_r2_w": weighted_r2_P(P_pred, P, grid=grid),
        # Dataset-level rank/order metrics. ``spearman_rho`` is the canonical
        # bin-free summary; ``bin_accuracy`` / ``bin_within_1_accuracy`` /
        # ``hard_rank_bin_mse`` use the same ``n_bins = 2*l_max+1`` quantile
        # binning as the training loss in :mod:`mpinv.losses.rank_bin`.
        f"report/{tag}/p_spearman_rho": spearman_rho_P(P_pred, P),
        f"report/{tag}/p_bin_accuracy": bin_accuracy_P(P_pred, P, n_bins),
        f"report/{tag}/p_bin_within_1_accuracy":
            bin_within_k_accuracy_P(P_pred, P, n_bins, k=1),
        f"report/{tag}/p_hard_rank_bin_mse":
            hard_rank_bin_mse_P(P_pred, P, n_bins),
        f"report/{tag}/p_n_bins": float(n_bins),
    }


def _figures_for(
    out_root: Path | None,
    tag: str,
    *,
    pred_packed: np.ndarray,
    target_packed: np.ndarray,
    P_pred: np.ndarray,
    P_true: np.ndarray,
    l_max: int,
    grid,
) -> None:
    if out_root is None:
        return
    target = out_root / tag
    target.mkdir(parents=True, exist_ok=True)
    build_run_report(
        RunArtifacts(
            pred_packed=pred_packed,
            target_packed=target_packed,
            P_pred=P_pred,
            P_true=P_true,
            l_max=l_max,
            grid=grid,
        ),
        output_dir=target,
    )


_AUG_CACHE_VERSION = "v2"  # bumped: v2 carries holdout_real arrays.


def _peek_split_ids(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str]]:
    """Backwards-compatible alias for :func:`mpinv.data.real_augmented_pipeline.peek_split_ids`."""
    from mpinv.data.real_augmented_pipeline import peek_split_ids

    return peek_split_ids(
        holdout_root=args.holdout_root,
        feature_subdir=args.feature_subdir,
        grid=GRID_DEFAULT,
        l_max=args.l_max,
        shuffle_seed=args.shuffle_seed,
        n_source=args.n_source,
        n_train_sources=args.n_train_sources,
        n_holdout_samples=args.n_holdout_samples,
        holdout_shuffle_seed=args.holdout_shuffle_seed,
    )


def _aug_cache_key(
    args: argparse.Namespace,
    train_sids: list[str],
    val_sids: list[str],
    holdout_sids: list[str],
) -> str:
    """Deterministic short hex key over every input that determines all
    cached arrays (augmented train, val, holdout).
    """
    payload = {
        "version": _AUG_CACHE_VERSION,
        "train_sids": list(train_sids),
        "val_sids": list(val_sids),
        "holdout_sids": list(holdout_sids),
        "n_augmented": int(args.n_augmented),
        "dropout_prob": float(args.dropout_prob),
        "field_sigma": float(args.field_sigma),
        "aug_seed": int(args.aug_seed),
        "l_max": int(args.l_max),
        "feature_subdir": args.feature_subdir,
        "scale_factor": float(args.scale_factor),
        "grid": (
            int(GRID_DEFAULT.n_phi),
            int(GRID_DEFAULT.n_theta),
            float(GRID_DEFAULT.theta_start_deg),
            float(GRID_DEFAULT.theta_end_deg),
        ),
    }
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _load_aug_cache(path: Path) -> dict[str, Any]:
    """Read a previously-saved augmented-dataset cache."""
    with np.load(path, allow_pickle=False) as f:
        out: dict[str, Any] = {
            "P_aug": f["P_aug"],
            "pk_aug": f["pk_aug"],
            "P_val": f["P_val"],
            "pk_val": f["pk_val"],
            "sids_train": f["sids_train"].tolist(),
            "sids_val": f["sids_val"].tolist(),
        }
        if "P_holdout" in f.files:
            out["P_holdout"] = f["P_holdout"]
            out["pk_holdout"] = f["pk_holdout"]
            out["sids_holdout"] = f["sids_holdout"].tolist()
        else:
            out["P_holdout"] = np.empty(
                (0, GRID_DEFAULT.n_theta, GRID_DEFAULT.n_phi), dtype=np.float32
            )
            out["pk_holdout"] = np.empty((0, 0), dtype=np.float32)
            out["sids_holdout"] = []
        return out


def _save_aug_cache(
    path: Path,
    *,
    P_aug: np.ndarray,
    pk_aug: np.ndarray,
    P_val: np.ndarray,
    pk_val: np.ndarray,
    sids_train: list[str],
    sids_val: list[str],
    P_holdout: np.ndarray,
    pk_holdout: np.ndarray,
    sids_holdout: list[str],
) -> None:
    """Write the augmented-dataset cache atomically.

    Uses a sibling ``.partial`` file + atomic ``Path.replace`` so concurrent
    readers never see a torn file. ``np.savez`` auto-appends ``.npz``, so we
    write to ``<key>.partial`` (an unrelated stem) and rename to ``<key>.npz``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez(
        tmp.with_suffix(""),  # np.savez auto-appends .npz
        P_aug=P_aug,
        pk_aug=pk_aug,
        P_val=P_val,
        pk_val=pk_val,
        sids_train=np.asarray(sids_train),
        sids_val=np.asarray(sids_val),
        P_holdout=P_holdout,
        pk_holdout=pk_holdout,
        sids_holdout=np.asarray(sids_holdout),
    )
    tmp.replace(path)


def load_run(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    which: str = "best",
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Reload a previously-saved real-augmented run.

    Reads ``<checkpoint_dir>/manifest.json``, unpickles the fitted feature
    extractor from ``<checkpoint_dir>/feature_extractor.pkl``, rebuilds the
    model architecture from the saved CLI args, and loads the chosen
    checkpoint's weights into it.

    Parameters
    ----------
    checkpoint_dir : str | Path
        The directory written by ``run_real_augmented.py``. Typically the
        value of ``--checkpoint-dir`` for that run, or
        ``<figures-dir>/checkpoints/`` if the default was used.
    device : str
        Torch device for ``torch.load(..., map_location=...)``. ``"cpu"``
        is the safe default; pass ``"cuda"`` if you want the model on GPU.
    which : str
        Which checkpoint file to load: ``"best"``, ``"last"``, or any
        ``"epoch_NNNN"`` (without the ``.pt`` suffix). The function
        appends ``.pt`` automatically.

    Returns
    -------
    (model, feature_extractor, manifest)
        ``model`` is in eval mode and ready for inference.
        ``feature_extractor`` is the fitted pipeline object whose
        ``transform(P=...)`` produces the input ``z`` the model expects.
        ``manifest`` is the parsed ``manifest.json`` dict for downstream
        introspection (args, metrics, file names).
    """
    import pickle

    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest.json under {checkpoint_dir!s}; "
            f"is this a real-augmented checkpoint dir?"
        )
    manifest = json.loads(manifest_path.read_text())

    feat_pkl = checkpoint_dir / manifest.get(
        "feature_extractor_pkl", "feature_extractor.pkl"
    )
    with feat_pkl.open("rb") as f:
        feat_blob = pickle.load(f)
    feat = feat_blob["feature_extractor"]

    saved_args = manifest["args"]
    l_max = int(saved_args["l_max"])
    K = l_max * (l_max + 2)
    model = _build_model(
        saved_args["model"],
        input_dim=int(feat.feature_dim),
        output_dim=4 * K,
    )

    ckpt_path = checkpoint_dir / f"{which}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint {ckpt_path!s} not found; "
            f"available: {sorted(p.name for p in checkpoint_dir.glob('*.pt'))}"
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, feat, manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    rng_aug = np.random.default_rng(args.aug_seed)

    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)

    cache_path: Path | None = None
    cached: dict[str, Any] | None = None
    if not args.smoke_test and not args.no_cache:
        try:
            train_sids_pre, val_sids_pre, holdout_sids_pre = _peek_split_ids(args)
        except Exception as exc:
            logger.warning(
                "could not pre-list holdout (%s); cache disabled this run", exc
            )
            train_sids_pre, val_sids_pre, holdout_sids_pre = [], [], []
        if train_sids_pre:
            key = _aug_cache_key(
                args, train_sids_pre, val_sids_pre, holdout_sids_pre
            )
            cache_path = Path(args.cache_dir) / f"aug_{key}.npz"
            if cache_path.exists() and not args.rebuild_cache:
                logger.info(
                    "cache HIT: loading augmented dataset from %s "
                    "(skipping load + decompose + augment)",
                    cache_path,
                )
                started_load = time.time()
                cached = _load_aug_cache(cache_path)
                logger.info(
                    "cache loaded: P_aug=%s pk_aug=%s P_val=%s pk_val=%s "
                    "P_holdout=%s in %.1fs",
                    cached["P_aug"].shape, cached["pk_aug"].shape,
                    cached["P_val"].shape, cached["pk_val"].shape,
                    cached["P_holdout"].shape,
                    time.time() - started_load,
                )
            else:
                logger.info(
                    "cache MISS%s: will rebuild and write %s",
                    " (--rebuild-cache)" if args.rebuild_cache else "",
                    cache_path,
                )

    if cached is not None:
        P_aug = cached["P_aug"]
        pk_aug = cached["pk_aug"]
        P_val = cached["P_val"]
        pk_val = cached["pk_val"]
        sids_train = cached["sids_train"]
        sids_val = cached["sids_val"]
        P_holdout = cached["P_holdout"]
        pk_holdout = cached["pk_holdout"]
        sids_holdout = cached["sids_holdout"]
        n_train_src = len(sids_train)
        n_val_src = len(sids_val)
        n_holdout = len(sids_holdout)
        n_real = n_train_src + n_val_src
        started_aug = time.time()
    else:
        # Resolve which sample IDs go to which split *first* so we know exactly
        # how many real-antenna samples to load (train+val+holdout).
        if args.smoke_test:
            sids_train_pre: list[str] = []
            sids_val_pre: list[str] = []
            sids_holdout_pre: list[str] = []
        else:
            sids_train_pre, sids_val_pre, sids_holdout_pre = _peek_split_ids(args)

        if args.smoke_test:
            P_real, packed_real, sids = _load_smoke(args, grid, args.l_max, basis)
            sids_train = sids[: args.n_train_sources]
            sids_val = sids[args.n_train_sources : args.n_source]
            sids_holdout = []
        else:
            # Build the union of IDs we want to materialise, then ask the
            # loader for only those by overriding ``max_samples`` to len(union)
            # *after* the deterministic shuffle. Because the loader's shuffle
            # already produced a canonical order, we just load enough to cover
            # the highest needed index, then index into the result.
            wanted = sids_train_pre + sids_val_pre + sids_holdout_pre
            wanted_set = set(wanted)
            cfg_full = RealAntennaLoaderConfig(
                root=args.holdout_root,
                feature_subdir=args.feature_subdir,
                grid=grid,
                l_max=args.l_max,
                shuffle_seed=args.shuffle_seed,
                max_samples=None,
            )
            all_pairs = list_real_antenna_samples(cfg_full)
            sid_to_pos = {sid: i for i, (_, _, sid) in enumerate(all_pairs)}
            highest = max((sid_to_pos[sid] for sid in wanted_set if sid in sid_to_pos),
                          default=-1)
            cfg = RealAntennaLoaderConfig(
                root=args.holdout_root,
                feature_subdir=args.feature_subdir,
                grid=grid,
                l_max=args.l_max,
                shuffle_seed=args.shuffle_seed,
                max_samples=highest + 1,
            )
            samples = load_real_antenna(cfg)
            by_sid = {s.sample_id: s for s in samples}

            def _stack(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
                if not ids:
                    return (
                        np.empty((0, grid.n_theta, grid.n_phi), dtype=np.float32),
                        np.empty((0, 4 * args.l_max * (args.l_max + 2)),
                                 dtype=np.float32),
                    )
                P = np.stack([by_sid[sid].P for sid in ids], axis=0)
                pk = np.stack([by_sid[sid].packed for sid in ids], axis=0)
                return P, pk

            P_train_src, pk_train_src = _stack(sids_train_pre)
            P_val, pk_val = _stack(sids_val_pre)
            P_holdout, pk_holdout = _stack(sids_holdout_pre)
            sids_train = sids_train_pre
            sids_val = sids_val_pre
            sids_holdout = sids_holdout_pre
            P_train_src = _truncate_and_resynthesise(
                P_train_src, pk_train_src, basis=basis
            ) if len(sids_train) else P_train_src
            P_val = _truncate_and_resynthesise(P_val, pk_val, basis=basis) \
                if len(sids_val) else P_val
            P_holdout = _truncate_and_resynthesise(
                P_holdout, pk_holdout, basis=basis
            ) if len(sids_holdout) else P_holdout
            logger.info(
                "loaded %d real-antenna samples; split: %d train + %d val + %d holdout",
                len(by_sid), len(sids_train), len(sids_val), len(sids_holdout),
            )
        if args.smoke_test:
            P_real_local = _truncate_and_resynthesise(P_real, packed_real, basis=basis)
            train_idx = np.arange(args.n_train_sources)
            val_idx = np.arange(args.n_train_sources, args.n_source)
            P_train_src = P_real_local[train_idx]
            pk_train_src = packed_real[train_idx]
            P_val = P_real_local[val_idx]
            pk_val = packed_real[val_idx]
            P_holdout = np.empty(
                (0, grid.n_theta, grid.n_phi), dtype=np.float32
            )
            pk_holdout = np.empty(
                (0, 4 * args.l_max * (args.l_max + 2)), dtype=np.float32
            )

        # Apply the optional global scale factor while (P, packed) is still
        # consistent under the project's basis. P scales as |E|^2 and packed
        # scales as |E|, so to make P -> s * P we set packed -> sqrt(s) *
        # packed. Augmentations downstream (phi_roll, mode_dropout,
        # field_additive_noise) are linear / multiplicative in (P, packed) and
        # preserve this relationship exactly.
        if args.scale_factor != 1.0:
            s = float(args.scale_factor)
            if s <= 0:
                raise ValueError(f"--scale-factor must be > 0; got {s}")
            sp = float(np.sqrt(s))
            P_train_src = (P_train_src * s).astype(np.float32, copy=False)
            pk_train_src = (pk_train_src * sp).astype(np.float32, copy=False)
            P_val = (P_val * s).astype(np.float32, copy=False)
            pk_val = (pk_val * sp).astype(np.float32, copy=False)
            P_holdout = (P_holdout * s).astype(np.float32, copy=False)
            pk_holdout = (pk_holdout * sp).astype(np.float32, copy=False)
            logger.info(
                "applied scale_factor=%g (P x %g, packed x %g) to all "
                "real-data splits before augmentation",
                s, s, sp,
            )

        n_train_src = len(sids_train)
        n_val_src = len(sids_val)
        n_holdout = len(sids_holdout)
        n_real = n_train_src + n_val_src
        if n_train_src == 0:
            raise ValueError("train split is empty")
        logger.info(
            "split (sample-id): %d train, %d val, %d holdout",
            n_train_src, n_val_src, n_holdout,
        )
        if args.n_augmented < n_train_src:
            logger.warning(
                "n_augmented=%d < n_train_sources=%d; not all sources will be sampled",
                args.n_augmented,
                n_train_src,
            )

        started_aug = time.time()
        P_aug, pk_aug = _build_augmented(
            P_train_src,
            pk_train_src,
            n_augmented=args.n_augmented,
            dropout_prob=args.dropout_prob,
            field_sigma=args.field_sigma,
            l_max=args.l_max,
            basis=basis,
            rng=rng_aug,
            chunk_size=args.aug_chunk_size,
        )
        if cache_path is not None:
            started_save = time.time()
            try:
                _save_aug_cache(
                    cache_path,
                    P_aug=P_aug, pk_aug=pk_aug,
                    P_val=P_val, pk_val=pk_val,
                    sids_train=sids_train, sids_val=sids_val,
                    P_holdout=P_holdout, pk_holdout=pk_holdout,
                    sids_holdout=sids_holdout,
                )
                logger.info(
                    "cache write: %s (%.1f MB) in %.1fs",
                    cache_path,
                    cache_path.stat().st_size / (1024 * 1024),
                    time.time() - started_save,
                )
            except Exception as exc:
                logger.warning("cache write failed: %s", exc)
        # NOTE: free the source arrays we no longer need.
        del P_train_src, pk_train_src
        gc.collect()
    logger.info(
        "augmented set ready: %d sources -> %d samples (build/load %.1fs)%s "
        "(recipe: phi_roll -> mode_dropout(p=%.3f) -> field_additive_noise(\u03c3=%g))",
        n_train_src,
        P_aug.shape[0],
        time.time() - started_aug,
        " [from cache]" if cached is not None else "",
        args.dropout_prob,
        args.field_sigma,
    )

    synth_test = _maybe_synthetic_test(args, grid, args.l_max, basis)
    if synth_test is not None and args.scale_factor != 1.0:
        # Bring synthetic_test onto the same units as the real-data splits so
        # field-space metrics and figure colormaps are comparable.
        s = float(args.scale_factor)
        sp = float(np.sqrt(s))
        synth_test = (
            (synth_test[0] * s).astype(np.float32, copy=False),
            (synth_test[1] * sp).astype(np.float32, copy=False),
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feat = _build_features(args.features, grid=grid, l_max=args.l_max)
    feat.fit(P_train=P_aug)
    z_train = feat.transform(P=P_aug)
    z_val = feat.transform(P=P_val)
    z_synth_test = feat.transform(P=synth_test[0]) if synth_test is not None else None
    z_holdout = feat.transform(P=P_holdout) if P_holdout.shape[0] > 0 else None

    K = args.l_max * (args.l_max + 2)
    model = _build_model(args.model, input_dim=feat.feature_dim, output_dim=4 * K)
    loss_fn, loss_kind = _build_loss(args.loss, grid=grid, l_max=args.l_max)
    optimiser = build_optimiser(
        model,
        OptimiserConfig(
            name=args.optimiser,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
        ),
    )
    if args.optimiser != "sgd" and args.nesterov:
        logger.warning(
            "--nesterov has no effect on optimiser=%s; only SGD uses it.",
            args.optimiser,
        )

    train_ds = _ArrayDataset(z_train, pk_aug, P_aug)
    val_ds = _ArrayDataset(z_val, pk_val, P_val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Build the LR scheduler if requested. Cosine variants need an explicit
    # total step count (one optimiser step per train batch); plateau and step
    # don't.
    steps_per_epoch = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs
    scheduler = build_scheduler(
        optimiser,
        SchedulerConfig(
            name=args.scheduler,
            total_steps=total_steps,
            warmup_steps=args.scheduler_warmup_steps,
            min_lr=args.scheduler_min_lr,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
            plateau_patience=args.scheduler_plateau_patience,
            plateau_factor=args.scheduler_plateau_factor,
        ),
    )
    if args.scheduler != "none":
        logger.info(
            "scheduler=%s (total_steps=%d, warmup=%d, min_lr=%g)",
            args.scheduler, total_steps, args.scheduler_warmup_steps,
            args.scheduler_min_lr,
        )

    # Decoder + eval subsamples are constructed *before* training so they can
    # be captured by the per-epoch figures callback (if --figures-every-n-epochs
    # > 0). The model itself is shared by reference; calling the closure mid-
    # training sees the latest in-place updated weights.
    decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)
    n_train_eval = min(args.n_train_eval_samples, P_aug.shape[0])
    eval_rng = np.random.default_rng(args.seed)
    train_eval_idx = eval_rng.choice(P_aug.shape[0], size=n_train_eval, replace=False)
    z_train_eval = z_train[train_eval_idx]
    pk_train_eval = pk_aug[train_eval_idx]
    P_train_eval = P_aug[train_eval_idx]

    # Resolve the checkpoint directory and persist the fitted feature
    # extractor *once*. Feature extractors in this codebase are fitted
    # exactly once before trainer.fit() and frozen for the rest of training,
    # so a single pickle alongside the model checkpoints is the right
    # cadence — see the plan in
    # .cursor/plans/checkpoint-models-and-features_*.plan.md for context.
    checkpoint_dir: Path | None = None
    if args.checkpoint_every_n_epochs > 0:
        if args.checkpoint_dir is not None:
            checkpoint_dir = Path(args.checkpoint_dir)
        elif args.figures_dir is not None:
            checkpoint_dir = Path(args.figures_dir) / "checkpoints"
        else:
            out_path = Path(args.output)
            checkpoint_dir = out_path.parent / f"{out_path.stem}_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        import pickle as _pickle
        feat_pkl_path = checkpoint_dir / "feature_extractor.pkl"
        with feat_pkl_path.open("wb") as _f:
            _pickle.dump(
                {
                    "feature_name": args.features,
                    "feature_extractor": feat,
                    "feature_dim": int(feat.feature_dim),
                    "l_max": int(args.l_max),
                    "grid": GRID_DEFAULT,
                },
                _f,
            )
        logger.info(
            "feature extractor saved: %s (dim=%d)",
            feat_pkl_path, feat.feature_dim,
        )

    trainer = Trainer(TrainerConfig(max_epochs=args.max_epochs, log_every_n_steps=200))
    callbacks: list[Any] = [
        LoggingCallback(log_every_n_steps=200),
        ValidationCallback(every_n_epochs=1),
        GradClipCallback(max_norm=1.0),
        EarlyStoppingCallback(patience=args.early_stop_patience),
    ]
    if checkpoint_dir is not None:
        from mpinv.callbacks.checkpoint_cb import CheckpointCallback
        callbacks.append(CheckpointCallback(
            output_dir=str(checkpoint_dir),
            save_every_n_epochs=int(args.checkpoint_every_n_epochs),
            keep_last=int(args.keep_last_checkpoints),
            save_best_metric="val/loss",
            higher_is_better=False,
        ))
        logger.info(
            "checkpoints: every %d epochs into %s (keep_last=%d)",
            args.checkpoint_every_n_epochs, checkpoint_dir,
            args.keep_last_checkpoints,
        )

    started_train = time.time()
    # The figures-per-N-epochs callback (if requested) needs ``emit_all_figures``
    # which is defined inside the post-training block below; we register the
    # callback *after* defining it but *before* calling trainer.fit().
    figures_root: Path | None = None
    emit_all_figures = None
    if not args.no_figures:
        figures_root = Path(
            args.figures_dir or "experiments/baseline/figures_real_augmented"
        )
        figures_root.mkdir(parents=True, exist_ok=True)

        n_bins_metric = 2 * args.l_max + 1

        # Splits in plotting order. Each entry is
        # (tag, z_features, packed_target, P_target, sample_id_list).
        figure_splits: list[
            tuple[str, np.ndarray, np.ndarray, np.ndarray, list[str]]
        ] = [
            ("train_aug", z_train_eval, pk_train_eval, P_train_eval, []),
            ("val_real", z_val, pk_val, P_val, list(sids_val)),
        ]
        if z_holdout is not None and P_holdout.shape[0] > 0:
            figure_splits.append(
                ("holdout_real", z_holdout, pk_holdout, P_holdout,
                 list(sids_holdout))
            )
        if synth_test is not None and z_synth_test is not None:
            P_st_full = synth_test[0]
            pk_st_full = synth_test[1]
            figure_splits.append(
                ("synthetic_test", z_synth_test, pk_st_full, P_st_full, [])
            )

        def _emit_all_figures(out_dir: Path, *, epoch_label: str = "") -> None:
            """Render the full per-split figure suite and the two cross-split
            distribution figures into ``out_dir``.

            Captures ``model``, ``decoder``, ``grid``, ``args``,
            ``eval_rng``, ``figure_splits``, ``n_bins_metric`` from the
            enclosing scope. Used both at the end of training and from a
            per-N-epochs callback. Leaves ``model`` in eval mode; callers
            running mid-train should call ``model.train()`` afterwards.
            """
            out_dir.mkdir(parents=True, exist_ok=True)
            r2_per_split: dict[str, np.ndarray] = {}
            bin_acc_per_split: dict[str, np.ndarray] = {}

            for tag, z, pk, P_true, sids in figure_splits:
                n_avail = z.shape[0]
                if n_avail == 0:
                    continue
                n_fig = min(max(1, args.n_figure_samples), n_avail)
                sub_idx = eval_rng.choice(n_avail, size=n_fig, replace=False)
                preds, P_pred = _predict_chunked(
                    model, decoder, z[sub_idx],
                    batch_size=args.eval_batch_size,
                )
                assert P_pred is not None
                _figures_for(
                    out_dir, tag,
                    pred_packed=preds, target_packed=pk[sub_idx],
                    P_pred=P_pred, P_true=P_true[sub_idx],
                    l_max=args.l_max, grid=grid,
                )
                sample_r2 = per_sample_weighted_r2_P(
                    P_pred, P_true[sub_idx], grid=grid
                )
                r2_per_split[tag] = np.asarray(sample_r2, dtype=np.float64)
                sample_bin_acc = per_sample_bin_accuracy_P(
                    P_pred, P_true[sub_idx], n_bins=n_bins_metric
                )
                bin_acc_per_split[tag] = np.asarray(
                    sample_bin_acc, dtype=np.float64
                )
                n_grid = min(args.n_figure_grid_samples, P_pred.shape[0])
                if n_grid <= 0:
                    continue
                order = np.argsort(sample_r2)
                n_third = max(1, n_grid // 3)
                picks: list[int] = []
                picks.extend(order[:n_third].tolist())
                picks.extend(order[-n_third:][::-1].tolist())
                mid = len(order) // 2
                mid_band = order[max(0, mid - n_third // 2):
                                 min(len(order), mid - n_third // 2 + n_third)].tolist()
                picks.extend([int(i) for i in mid_band if i not in picks])
                remaining = [int(i) for i in range(P_pred.shape[0]) if i not in picks]
                if remaining and len(picks) < n_grid:
                    pad = eval_rng.choice(
                        len(remaining),
                        size=min(n_grid - len(picks), len(remaining)),
                        replace=False,
                    )
                    picks.extend(int(remaining[int(p)]) for p in pad)
                picks = picks[:n_grid]
                picks.sort(key=lambda i: float(sample_r2[i]))
                picked_sids = (
                    [sids[int(sub_idx[i])] for i in picks]
                    if len(sids) >= n_avail else None
                )
                grid_fig = build_field_comparison_grid_figure(
                    P_pred, P_true[sub_idx],
                    sample_indices=picks,
                    sample_ids=picked_sids,
                    per_sample_metric=sample_r2.tolist(),
                    metric_label="R²",
                    metric_fmt="{:+.3f}",
                    title=(
                        f"{tag}{(' @ ' + epoch_label) if epoch_label else ''}: "
                        f"{n_grid} samples ranked by sin-θ-weighted R² "
                        f"(rows top→bottom: worst → best)"
                    ),
                )
                grid_path = out_dir / tag / "field_comparison_grid.pdf"
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                grid_fig.savefig(grid_path, bbox_inches="tight")
                plt.close(grid_fig)

            title_suffix = f" — {epoch_label}" if epoch_label else ""
            common_title = (
                f"L={args.l_max}, scale={args.scale_factor:g}, "
                f"model={args.model}, loss={args.loss}{title_suffix}"
            )
            if r2_per_split:
                r2_dist_fig = build_r2_distribution_figure(
                    r2_per_split,
                    title=f"R² distribution across splits — {common_title}",
                )
                r2_dist_fig.savefig(out_dir / "r2_distribution.pdf",
                                    bbox_inches="tight")
                plt.close(r2_dist_fig)
            if bin_acc_per_split:
                bin_dist_fig = build_bin_accuracy_distribution_figure(
                    bin_acc_per_split,
                    n_bins_metric=n_bins_metric,
                    title=(
                        f"Hard rank-bin accuracy distribution across splits "
                        f"(n_bins={n_bins_metric}) — {common_title}"
                    ),
                )
                bin_dist_fig.savefig(out_dir / "bin_accuracy_distribution.pdf",
                                     bbox_inches="tight")
                plt.close(bin_dist_fig)

        emit_all_figures = _emit_all_figures

        # Register the per-epoch figures callback if requested.
        if args.figures_every_n_epochs > 0:
            from mpinv.callbacks.base import Callback as _CallbackBase

            @dataclass(slots=True)
            class _PerEpochFiguresCallback(_CallbackBase):
                every_n_epochs: int
                emit_fn: Any
                figures_root: Path

                def on_epoch_end(self, ctx) -> None:
                    if self.every_n_epochs <= 0:
                        return
                    if ctx.epoch <= 0:
                        return
                    if (ctx.epoch % self.every_n_epochs) != 0:
                        return
                    epoch_dir = self.figures_root / f"epoch_{ctx.epoch:04d}"
                    started = time.time()
                    self.emit_fn(epoch_dir, epoch_label=f"epoch {ctx.epoch}")
                    logger.info(
                        "per-epoch figures: epoch=%d wrote %s in %.1fs",
                        ctx.epoch, epoch_dir, time.time() - started,
                    )
                    # Restore train mode — _predict_chunked left it in eval.
                    ctx.model.train()

            callbacks.append(_PerEpochFiguresCallback(
                every_n_epochs=int(args.figures_every_n_epochs),
                emit_fn=emit_all_figures,
                figures_root=figures_root,
            ))
            logger.info(
                "per-epoch figures: enabled every %d epochs into %s",
                args.figures_every_n_epochs, figures_root,
            )

    trainer.fit(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimiser=optimiser,
        loss_kind=loss_kind,
        scheduler=scheduler,
        val_loader=val_loader,
        callbacks=callbacks,
    )
    train_elapsed = time.time() - started_train
    logger.info("training finished in %.1fs", train_elapsed)

    metrics: dict[str, float] = {}
    metrics.update(_eval_split_chunked(
        model, decoder, z_train_eval, pk_train_eval, P_train_eval,
        l_max=args.l_max, grid=grid, tag="train_aug",
        batch_size=args.eval_batch_size,
    ))
    metrics.update(_eval_split_chunked(
        model, decoder, z_val, pk_val, P_val,
        l_max=args.l_max, grid=grid, tag="val_real",
        batch_size=args.eval_batch_size,
    ))
    if z_holdout is not None and P_holdout.shape[0] > 0:
        metrics.update(_eval_split_chunked(
            model, decoder, z_holdout, pk_holdout, P_holdout,
            l_max=args.l_max, grid=grid, tag="holdout_real",
            batch_size=args.eval_batch_size,
        ))
    if synth_test is not None and z_synth_test is not None:
        P_st, pk_st = synth_test
        metrics.update(_eval_split_chunked(
            model, decoder, z_synth_test, pk_st, P_st,
            l_max=args.l_max, grid=grid, tag="synthetic_test",
            batch_size=args.eval_batch_size,
        ))

    # End-of-training: emit the standard figure suite into figures_root.
    # The same closure is also called by the per-epoch callback (if
    # --figures-every-n-epochs > 0); see the pre-training block above.
    if emit_all_figures is not None and figures_root is not None:
        emit_all_figures(figures_root, epoch_label="")
        logger.info("figures written under %s", figures_root)

    record = {
        "ok": True,
        "elapsed_s_aug": time.time() - started_aug,
        "elapsed_s_train": train_elapsed,
        "feat_dim": int(feat.feature_dim),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "model": args.model,
        "loss": args.loss,
        "features": args.features,
        "augmentation": (
            f"phi_roll \u2192 mode_dropout(p={args.dropout_prob}) "
            f"\u2192 field_additive_noise(\u03c3={args.field_sigma})"
        ),
        "regime": "real_augmented" + (" (smoke)" if args.smoke_test else ""),
        "n_real_loaded": int(n_real),
        "n_train_sources": int(n_train_src),
        "n_val_sources": int(n_val_src),
        "n_holdout_sources": int(n_holdout),
        "n_augmented": int(P_aug.shape[0]),
        "n_train_eval_samples": int(n_train_eval),
        "l_max": int(args.l_max),
        "dropout_prob": float(args.dropout_prob),
        "field_sigma": float(args.field_sigma),
        "scale_factor": float(args.scale_factor),
        "max_epochs": int(args.max_epochs),
        "lr": float(args.lr),
        "optimiser": args.optimiser,
        "momentum": float(args.momentum),
        "nesterov": bool(args.nesterov),
        "weight_decay": float(args.weight_decay),
        "scheduler": args.scheduler,
        "scheduler_min_lr": float(args.scheduler_min_lr),
        "scheduler_warmup_steps": int(args.scheduler_warmup_steps),
        "scheduler_plateau_patience": int(args.scheduler_plateau_patience),
        "scheduler_plateau_factor": float(args.scheduler_plateau_factor),
        "smoke_test": bool(args.smoke_test),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("results written to %s", out_path)

    # Write a self-contained manifest.json next to the checkpoints so a
    # later ``load_run()`` call can reload the entire artefact tree from a
    # single path. We avoid pickling ``args`` directly (it's an
    # ``argparse.Namespace``); ``vars(args)`` returns a plain dict, which
    # is JSON-friendly because every flag is a primitive type.
    if checkpoint_dir is not None:
        manifest = {
            "args": vars(args),
            "feature_dim": int(feat.feature_dim),
            "feature_extractor_pkl": "feature_extractor.pkl",
            "model_checkpoints": {
                "best": "best.pt",
                "last": "last.pt",
                "epoch_pattern": "epoch_NNNN.pt",
                "every_n_epochs": int(args.checkpoint_every_n_epochs),
                "keep_last": int(args.keep_last_checkpoints),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "n_train_eval_samples": int(n_train_eval),
            "elapsed_s_aug": float(record["elapsed_s_aug"]),
            "elapsed_s_train": float(record["elapsed_s_train"]),
        }
        (checkpoint_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        logger.info("manifest written to %s", checkpoint_dir / "manifest.json")

    summary_keys = [
        ("report/train_aug/coef_mse_amb_aware", "train_aug coef_mse_amb_aware"),
        ("report/train_aug/field_r2_w", "train_aug field_r2_w"),
        ("report/train_aug/p_spearman_rho", "train_aug p_spearman_rho"),
        ("report/train_aug/p_bin_accuracy", "train_aug p_bin_accuracy"),
        ("report/val_real/coef_mse_amb_aware", "val_real coef_mse_amb_aware"),
        ("report/val_real/field_nrmse_w", "val_real field_nrmse_w"),
        ("report/val_real/field_r2_w", "val_real field_r2_w"),
        ("report/val_real/p_spearman_rho", "val_real p_spearman_rho"),
        ("report/val_real/p_bin_accuracy", "val_real p_bin_accuracy"),
        ("report/val_real/p_bin_within_1_accuracy",
         "val_real p_bin_within_1_accuracy"),
        ("report/holdout_real/coef_mse_amb_aware",
         "holdout_real coef_mse_amb_aware"),
        ("report/holdout_real/field_nrmse_w", "holdout_real field_nrmse_w"),
        ("report/holdout_real/field_r2_w", "holdout_real field_r2_w"),
        ("report/holdout_real/p_spearman_rho", "holdout_real p_spearman_rho"),
        ("report/holdout_real/p_bin_accuracy", "holdout_real p_bin_accuracy"),
        ("report/holdout_real/p_bin_within_1_accuracy",
         "holdout_real p_bin_within_1_accuracy"),
        ("report/synthetic_test/coef_mse_amb_aware",
         "synthetic_test coef_mse_amb_aware"),
        ("report/synthetic_test/field_r2_w", "synthetic_test field_r2_w"),
        ("report/synthetic_test/p_spearman_rho",
         "synthetic_test p_spearman_rho"),
    ]
    for key, label in summary_keys:
        val = metrics.get(key)
        if val is None:
            continue
        logger.info("  %s = %.6g", label, val)

    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
