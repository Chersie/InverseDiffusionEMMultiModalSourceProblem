"""Stage-wise real-augmented experiment for the multi-head per-l-band MLP.

Mirrors :mod:`scripts.run_real_augmented` (same real-antenna load + on-manifold
augmentation pipeline, same feature extractor, same physics+rank loss) but
replaces the single ``Trainer.fit(...)`` call with an N-stage loop driven by
:class:`mpinv.training.staged.StagedTrainer`. Each stage trains exactly one
per-l-band head (or a configurable group); earlier heads stay frozen at their
trained weights, later heads are zero-frozen so they emit identically zero
through the physics decoder.

Two key flows the script supports:

1. **Stage 1..N from scratch**: train an L=5 multi-head model, group by group.

   .. code-block:: bash

        uv run python scripts/run_staged_real_augmented.py \\
            --holdout-root data/raw/real_antenna --n-source 200 \\
            --n-train-sources 180 --n-augmented 10000 \\
            --scale-factor 1000000 --l-max 5 \\
            --groups 'auto' \\
            --backbone-policy freeze_after_stage1 \\
            --features raw_plus_sh --loss physics_power_rank \\
            --optimiser sgd --nesterov --scheduler cosine_with_warmup \\
            --lr 1e-4 --stage-max-epochs 30 --batch-size 128 \\
            --output experiments/staged/staged_l5.json \\
            --figures-dir experiments/staged/figures_l5

2. **"Modes 6-15 with fixed 1-5"**: load a previously trained smaller-L model
   from a checkpoint dir, transplant its heads (and backbone) into a larger
   L=15 multi-head model, then run stages 6..15 with heads 1..5 frozen.

   .. code-block:: bash

        uv run python scripts/run_staged_real_augmented.py \\
            --l-max 15 --groups 'auto' \\
            --transplant-from experiments/staged/figures_l5/checkpoints/stage_5 \\
            --starting-stage 6 \\
            --backbone-policy freeze_after_stage1 \\
            --output experiments/staged/staged_l15_after_l5.json \\
            --figures-dir experiments/staged/figures_l15_after_l5

The script reuses :mod:`scripts.run_real_augmented`'s data + cache + augmentation
primitives by direct import. It writes per-stage checkpoints under
``<figures-dir>/checkpoints/stage_<k>/{best,last,epoch_*}.pt`` plus a single
``model_config.json`` describing the :class:`MultiHeadMLPConfig` used (so a
later run can transplant from this dir without further metadata).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib  # noqa: E402  -- must precede pyplot import below
matplotlib.use("Agg")  # headless backend; figures are saved, never shown
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline_inprocess import (  # type: ignore[import-not-found]
    _build_features,
    _build_loss,
)
from run_real_augmented import (  # type: ignore[import-not-found]
    _aug_cache_key,
    _build_augmented,
    _eval_split_chunked,
    _figures_for,
    _load_aug_cache,
    _load_real,
    _load_smoke,
    _maybe_synthetic_test,
    _peek_split_ids,
    _predict_chunked,
    _save_aug_cache,
    _truncate_and_resynthesise,
)

from mpinv.analysis.metrics.field_metrics import (
    per_sample_bin_accuracy_P,
    per_sample_weighted_r2_P,
)
from mpinv.analysis.plots.field_comparison import build_field_comparison_grid_figure
from mpinv.analysis.plots.r2_distribution import (
    build_bin_accuracy_distribution_figure,
    build_r2_distribution_figure,
)
from mpinv.callbacks.base import Callback as _CallbackBase
from mpinv.callbacks.checkpoint_cb import CheckpointCallback
from mpinv.cli._builders import build_physics_power_loss
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import _ArrayDataset
from mpinv.core.grid import GRID_DEFAULT
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.models.multi_head_mlp import (
    MultiHeadMLP,
    MultiHeadMLPConfig,
    expected_output_dim,
    transplant_heads,
)
from mpinv.training.optim import OptimiserConfig, SchedulerConfig
from mpinv.training.staged import (
    BackbonePolicy,
    StagedTrainer,
    StagedTrainerConfig,
)
from mpinv.training.trainer import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data + augmentation knobs (subset of run_real_augmented; kept identical
    # in semantics so the cache is interchangeable).
    p.add_argument("--holdout-root", default="data/raw/real_antenna", type=str)
    p.add_argument("--feature-subdir", default="E_in_plane", type=str)
    p.add_argument("--n-source", default=100, type=int)
    p.add_argument("--n-train-sources", default=80, type=int)
    p.add_argument("--n-augmented", default=10000, type=int)
    p.add_argument("--shuffle-seed", default=42, type=int)
    p.add_argument("--aug-seed", default=4242, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--l-max", default=5, type=int)
    p.add_argument("--dropout-prob", default=0.1, type=float)
    p.add_argument("--field-sigma", default=1e-8, type=float)
    p.add_argument("--scale-factor", default=1.0, type=float)
    p.add_argument("--aug-chunk-size", default=500, type=int)
    p.add_argument("--n-holdout-samples", default=100, type=int)
    p.add_argument("--holdout-shuffle-seed", default=314159, type=int)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--cache-dir", default="data/cache/real_augmented", type=str)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")

    # Model architecture (MultiHeadMLP body knobs). Defaults match the L=5 best
    # config (mlp_5x200 body + ELU + dropout 0.001).
    p.add_argument("--hidden-size", default=200, type=int)
    p.add_argument("--n-hidden-layers", default=5, type=int)
    p.add_argument("--architecture", default="flat",
                   choices=["flat", "pyramid", "bottleneck", "residual"])
    p.add_argument("--activation", default="elu",
                   choices=["silu", "relu", "gelu", "elu"])
    p.add_argument("--use-layer-norm", action="store_true")
    p.add_argument("--no-bias", action="store_true",
                   help="Disable bias on every Linear in body and heads.")
    p.add_argument("--model-dropout", default=0.001, type=float)
    p.add_argument("--groups", default="auto", type=str,
                   help="JSON l-band partition (e.g. '[[1],[2],[3],[4],[5]]') "
                        "or 'auto' (= one head per l in 1..l_max).")

    # Loss + features (default to the best-known combo).
    p.add_argument("--loss", default="physics_power_rank", type=str)
    p.add_argument("--features", default="raw_plus_sh", type=str)
    # Per-component weights inside PhysicsPowerLoss. None ⇒ use the legacy
    # baseline default for the chosen --loss (rank_bin_weight=0.1 for
    # physics_power_rank; coef_aux_weight=0.1 for physics_power_mixed; both 0
    # for plain physics_power). Pass an explicit float to override; e.g.
    # --rank-bin-weight 0.0 turns the rank regulariser off, --rank-bin-weight
    # 1.0 makes the rank term the dominant signal.
    p.add_argument("--rank-bin-weight", default=None, type=float,
                   help="Override PhysicsPowerLossConfig.rank_bin_weight. "
                        "Defaults: 0.1 for --loss physics_power_rank, "
                        "0.0 for everything else.")
    p.add_argument("--coef-aux-weight", default=None, type=float,
                   help="Override PhysicsPowerLossConfig.coef_aux_weight. "
                        "Defaults: 0.1 for --loss physics_power_mixed, "
                        "0.0 for everything else.")
    p.add_argument("--rank-bin-n-bins", default=None, type=int,
                   help="Number of soft rank bins. None (default) ⇒ "
                        "2*l_max+1 (matches angular resolution).")
    p.add_argument("--rank-bin-beta", default=10.0, type=float,
                   help="Sigmoid temperature for the rank-bin soft binning.")
    p.add_argument("--physics-log-ratio", action="store_true",
                   help="Compute physics MSE in log(P_pred+eps) space.")

    # Optimiser knobs.
    p.add_argument("--optimiser", default="adamw",
                   choices=["adamw", "adam", "sgd"])
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", default=0.0, type=float)
    p.add_argument("--scheduler", default="none",
                   choices=["none", "plateau", "cosine", "cosine_with_warmup", "step"])
    p.add_argument("--scheduler-min-lr", default=1e-6, type=float)
    p.add_argument("--scheduler-warmup-steps", default=50, type=int)
    p.add_argument("--scheduler-plateau-patience", default=5, type=int)
    p.add_argument("--scheduler-plateau-factor", default=0.5, type=float)
    p.add_argument("--scheduler-step-size", default=10, type=int)
    p.add_argument("--scheduler-gamma", default=0.1, type=float)

    # Training cadence.
    p.add_argument("--stage-max-epochs", default=30, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--early-stop-patience", default=10, type=int)

    # Stage policy.
    p.add_argument("--backbone-policy", default="freeze_after_stage1",
                   choices=[
                       "trainable_always",
                       "freeze_after_stage1",
                       "lower_lr_after_stage1",
                   ])
    p.add_argument("--backbone-lr-factor", default=0.1, type=float)
    p.add_argument("--no-reinit-active-head", action="store_true",
                   help="Skip head.reset_parameters() at stage boundaries; the "
                        "head keeps whatever state (zero or default-init) it "
                        "had. Default behaviour reinits to non-zero weights.")
    p.add_argument("--no-zero-init-future-heads", action="store_true",
                   help="Leave future heads at their default-distributed init "
                        "(still requires_grad=False). Use for the 'frozen "
                        "random heads' ablation. Default behaviour zeros.")
    p.add_argument("--starting-stage", default=1, type=int,
                   help="1-indexed first stage to run. Use with --transplant-from "
                        "to skip the stages whose heads are already trained.")
    p.add_argument("--first-stage-only", action="store_true",
                   help="Sanity check: run only stage 1 then exit.")

    # Transplant.
    p.add_argument("--transplant-from", default=None, type=str,
                   help="Path to a directory containing model_config.json + "
                        "either best.pt or last.pt (written by a previous "
                        "staged run). Heads matching the destination groups "
                        "are copied and frozen; backbone is copied iff the "
                        "two configs share body knobs.")
    p.add_argument("--transplant-which", default="best",
                   choices=["best", "last"])
    p.add_argument("--no-copy-backbone-on-transplant", action="store_true",
                   help="Copy only the per-l head weights from the transplant "
                        "source; leave the destination backbone at its random "
                        "init. Required when source and destination have "
                        "different feature dims (e.g. raw_plus_sh at L=5 vs "
                        "L=15) because the first backbone Linear's input "
                        "shape changes with l_max.")

    # Eval / figures / I/O.
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--n-train-eval-samples", default=1024, type=int)
    p.add_argument("--n-figure-samples", default=64, type=int)
    p.add_argument("--n-figure-grid-samples", default=8, type=int)
    p.add_argument("--n-synthetic-test", default=512, type=int)
    p.add_argument("--seed-synthetic-test", default=9012, type=int)
    p.add_argument("--include-synthetic-test", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--figures-dir", default=None, type=str)
    p.add_argument("--figures-every-n-epochs", default=0, type=int,
                   help="If > 0, emit the full figure suite into "
                        "<figures_dir>/stage_<k>/epoch_<NNNN>/ every N "
                        "epochs *within each stage*. Mirrors baseline's "
                        "--figures-every-n-epochs but scopes the per-epoch "
                        "directory to the current stage. End-of-stage and "
                        "end-of-training figures are always emitted unless "
                        "--no-figures-after-each-stage / --no-figures.")
    p.add_argument("--no-figures-after-each-stage", action="store_true",
                   help="Skip the per-stage figure suite emitted into "
                        "<figures_dir>/stage_<k>/ at the end of every stage. "
                        "End-of-training figures still go to <figures_dir>/final/.")
    p.add_argument("--checkpoint-every-n-epochs", default=10, type=int)
    p.add_argument("--keep-last-checkpoints", default=3, type=int)
    p.add_argument("--output", required=True, type=str)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model construction + transplant
# ---------------------------------------------------------------------------


def _build_loss_with_overrides(
    args: argparse.Namespace, *, grid, l_max: int
) -> tuple[Any, str, dict[str, float | int | bool | None]]:
    """Wrap :func:`scripts.run_baseline_inprocess._build_loss` with per-knob
    overrides for the ``physics_power*`` family.

    Behaviour for the ``physics_power*`` losses:

    - Start from the legacy baseline defaults (the same ones hard-coded in
      :func:`run_baseline_inprocess._build_loss`):
      ``physics_power_rank``  → ``rank_bin_weight=0.1, coef_aux_weight=0.0``
      ``physics_power_mixed`` → ``rank_bin_weight=0.0, coef_aux_weight=0.1``
      ``physics_power``       → ``rank_bin_weight=0.0, coef_aux_weight=0.0``
    - Apply CLI overrides (``--rank-bin-weight``, ``--coef-aux-weight``,
      ``--rank-bin-n-bins``, ``--rank-bin-beta``, ``--physics-log-ratio``)
      iff the user passed them. Defaults in argparse are ``None`` for the
      override knobs precisely so we can distinguish "not passed" from "0.0".

    For all other losses (``coef_mse``, ``rank_bin_p`` …) the call falls
    through to the legacy ``_build_loss`` and the override flags are ignored
    (logged, not silently dropped).

    Returns ``(loss_module, loss_kind, weights_dict)`` where ``weights_dict``
    is suitable for inclusion in the run's JSON record (so the resolved
    weights are reproducible from the result file alone).
    """
    name = args.loss
    if name in {"physics_power", "physics_power_rank", "physics_power_mixed"}:
        defaults: dict[str, float] = {
            "physics_power":        {"rank_bin_weight": 0.0, "coef_aux_weight": 0.0},
            "physics_power_rank":   {"rank_bin_weight": 0.1, "coef_aux_weight": 0.0},
            "physics_power_mixed":  {"rank_bin_weight": 0.0, "coef_aux_weight": 0.1},
        }[name]
        rank_w = args.rank_bin_weight if args.rank_bin_weight is not None else defaults["rank_bin_weight"]
        coef_w = args.coef_aux_weight if args.coef_aux_weight is not None else defaults["coef_aux_weight"]
        rank_n = args.rank_bin_n_bins  # None ⇒ 2*l_max+1, handled inside the loss
        rank_beta = float(args.rank_bin_beta)
        log_ratio = bool(args.physics_log_ratio)
        loss = build_physics_power_loss(
            grid=grid, l_max=l_max,
            rank_bin_weight=float(rank_w),
            coef_aux_weight=float(coef_w),
            rank_bin_n_bins=rank_n,
            rank_bin_beta=rank_beta,
            log_ratio=log_ratio,
        )
        weights = {
            "loss": name,
            "rank_bin_weight": float(rank_w),
            "coef_aux_weight": float(coef_w),
            "rank_bin_n_bins": (int(rank_n) if rank_n is not None else (2 * l_max + 1)),
            "rank_bin_beta": float(rank_beta),
            "log_ratio": bool(log_ratio),
        }
        logger.info(
            "loss=%s  rank_bin_weight=%g  coef_aux_weight=%g  rank_bin_n_bins=%d  "
            "rank_bin_beta=%g  log_ratio=%s",
            name, weights["rank_bin_weight"], weights["coef_aux_weight"],
            weights["rank_bin_n_bins"], weights["rank_bin_beta"], weights["log_ratio"],
        )
        return loss, "physics", weights

    # All other losses: fall back to the legacy registry-builder, and warn
    # if the user passed any physics-loss override flags (they are silently
    # ignored by the legacy builder — surface that to avoid silent bugs).
    if (
        args.rank_bin_weight is not None
        or args.coef_aux_weight is not None
        or args.rank_bin_n_bins is not None
        or args.physics_log_ratio
    ):
        logger.warning(
            "ignoring --rank-bin-weight/--coef-aux-weight/--rank-bin-n-bins/"
            "--physics-log-ratio because --loss=%s does not use them",
            name,
        )
    loss_fn, loss_kind = _build_loss(name, grid=grid, l_max=l_max)
    return loss_fn, loss_kind, {"loss": name}


def _resolve_groups(arg: str, l_max: int) -> list[list[int]]:
    if arg == "auto":
        return [[l] for l in range(1, l_max + 1)]
    parsed = json.loads(arg)
    if not isinstance(parsed, list) or not all(isinstance(g, list) for g in parsed):
        raise ValueError(f"--groups must be JSON list[list[int]] or 'auto'; got {arg!r}")
    return [[int(l) for l in g] for g in parsed]


def _build_multi_head_model(
    args: argparse.Namespace, *, input_dim: int, l_max: int, groups: list[list[int]]
) -> MultiHeadMLP:
    cfg = MultiHeadMLPConfig(
        input_dim=int(input_dim),
        output_dim=expected_output_dim(l_max),
        l_max=int(l_max),
        groups=groups,
        hidden_size=int(args.hidden_size),
        n_hidden_layers=int(args.n_hidden_layers),
        architecture=args.architecture,
        dropout=float(args.model_dropout),
        use_layer_norm=bool(args.use_layer_norm),
        use_bias=not bool(args.no_bias),
        activation=args.activation,
    )
    return MultiHeadMLP(cfg)


def _save_model_config(checkpoint_root: Path, model: MultiHeadMLP) -> None:
    """Persist the :class:`MultiHeadMLPConfig` next to the checkpoints so a
    later run can rebuild an identical model for transplanting.
    """
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(model.cfg)
    (checkpoint_root / "model_config.json").write_text(json.dumps(cfg_dict, indent=2))


def _load_transplant_source(
    transplant_dir: Path, which: str = "best"
) -> MultiHeadMLP:
    """Reconstruct the source :class:`MultiHeadMLP` from a previous staged run."""
    cfg_path = transplant_dir / "model_config.json"
    if not cfg_path.exists():
        # Fall back: model_config.json may live one level up if --transplant-from
        # pointed at a per-stage subdir.
        parent_cfg = transplant_dir.parent / "model_config.json"
        if parent_cfg.exists():
            cfg_path = parent_cfg
        else:
            raise FileNotFoundError(
                f"no model_config.json found under {transplant_dir} or its parent; "
                f"transplant requires the staged run's model_config.json."
            )
    cfg_dict = json.loads(cfg_path.read_text())
    src_cfg = MultiHeadMLPConfig(**cfg_dict)
    src = MultiHeadMLP(src_cfg)

    # Resolve checkpoint path. Accept best.pt / last.pt directly, or a stage_<k>
    # subdir whose best.pt is the checkpoint.
    candidates = [
        transplant_dir / f"{which}.pt",
        transplant_dir / "best.pt",
        transplant_dir / "last.pt",
    ]
    ckpt_path = next((c for c in candidates if c.exists()), None)
    if ckpt_path is None:
        # Try the deepest stage_*/best.pt as a convenience.
        stage_dirs = sorted(transplant_dir.glob("stage_*"))
        for d in reversed(stage_dirs):
            for c in (d / f"{which}.pt", d / "best.pt", d / "last.pt"):
                if c.exists():
                    ckpt_path = c
                    break
            if ckpt_path is not None:
                break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"no .pt checkpoint found in {transplant_dir} (looked for "
            f"{which}.pt / best.pt / last.pt and any stage_*/*.pt)"
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src.load_state_dict(state["model"])
    logger.info("transplant source reconstructed from %s (cfg %s)", ckpt_path, cfg_path)
    return src


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    # ------------------- data + augmentation (cache-aware) -------------------
    cache_path: Path | None = None
    cached: dict[str, Any] | None = None
    if not args.smoke_test and not args.no_cache:
        try:
            train_sids_pre, val_sids_pre, holdout_sids_pre = _peek_split_ids(args)
        except Exception as exc:
            logger.warning("could not pre-list holdout (%s); cache disabled", exc)
            train_sids_pre, val_sids_pre, holdout_sids_pre = [], [], []
        if train_sids_pre:
            key = _aug_cache_key(args, train_sids_pre, val_sids_pre, holdout_sids_pre)
            cache_path = Path(args.cache_dir) / f"aug_{key}.npz"
            if cache_path.exists() and not args.rebuild_cache:
                logger.info("cache HIT: %s", cache_path)
                cached = _load_aug_cache(cache_path)
            else:
                logger.info("cache MISS: will rebuild and write %s", cache_path)

    if cached is not None:
        P_aug = cached["P_aug"]; pk_aug = cached["pk_aug"]
        P_val = cached["P_val"]; pk_val = cached["pk_val"]
        sids_train = cached["sids_train"]; sids_val = cached["sids_val"]
        P_holdout = cached["P_holdout"]; pk_holdout = cached["pk_holdout"]
        sids_holdout = cached["sids_holdout"]
        n_train_src = len(sids_train); n_val_src = len(sids_val)
        n_holdout = len(sids_holdout); n_real = n_train_src + n_val_src
        started_aug = time.time()
    else:
        # Full real-augmented build (mirrors run_real_augmented.main, abridged).
        if args.smoke_test:
            P_real, packed_real, sids = _load_smoke(args, grid, args.l_max, basis)
            sids_train = sids[: args.n_train_sources]
            sids_val = sids[args.n_train_sources : args.n_source]
            sids_holdout = []
            P_real_local = _truncate_and_resynthesise(P_real, packed_real, basis=basis)
            train_idx = np.arange(args.n_train_sources)
            val_idx = np.arange(args.n_train_sources, args.n_source)
            P_train_src = P_real_local[train_idx]
            pk_train_src = packed_real[train_idx]
            P_val = P_real_local[val_idx]
            pk_val = packed_real[val_idx]
            P_holdout = np.empty((0, grid.n_theta, grid.n_phi), dtype=np.float32)
            pk_holdout = np.empty(
                (0, 4 * args.l_max * (args.l_max + 2)), dtype=np.float32
            )
        else:
            P_real, pk_real, sids = _load_real(args, grid, args.l_max)
            P_real = _truncate_and_resynthesise(P_real, pk_real, basis=basis)
            sids_train = sids[: args.n_train_sources]
            sids_val = sids[args.n_train_sources : args.n_source]
            sids_holdout = sids[args.n_source : args.n_source + args.n_holdout_samples]
            train_idx = list(range(args.n_train_sources))
            val_idx = list(range(args.n_train_sources, args.n_source))
            holdout_idx = list(
                range(args.n_source, args.n_source + len(sids_holdout))
            )
            P_train_src = P_real[train_idx]; pk_train_src = pk_real[train_idx]
            P_val = P_real[val_idx]; pk_val = pk_real[val_idx]
            P_holdout = P_real[holdout_idx] if holdout_idx else np.empty(
                (0, grid.n_theta, grid.n_phi), dtype=np.float32
            )
            pk_holdout = pk_real[holdout_idx] if holdout_idx else np.empty(
                (0, 4 * args.l_max * (args.l_max + 2)), dtype=np.float32
            )

        if args.scale_factor != 1.0:
            s = float(args.scale_factor); sp = float(np.sqrt(s))
            P_train_src = (P_train_src * s).astype(np.float32, copy=False)
            pk_train_src = (pk_train_src * sp).astype(np.float32, copy=False)
            P_val = (P_val * s).astype(np.float32, copy=False)
            pk_val = (pk_val * sp).astype(np.float32, copy=False)
            P_holdout = (P_holdout * s).astype(np.float32, copy=False)
            pk_holdout = (pk_holdout * sp).astype(np.float32, copy=False)

        n_train_src = len(sids_train); n_val_src = len(sids_val)
        n_holdout = len(sids_holdout); n_real = n_train_src + n_val_src
        if n_train_src == 0:
            raise ValueError("train split is empty")

        started_aug = time.time()
        P_aug, pk_aug = _build_augmented(
            P_train_src, pk_train_src,
            n_augmented=args.n_augmented,
            dropout_prob=args.dropout_prob,
            field_sigma=args.field_sigma,
            l_max=args.l_max,
            basis=basis, rng=rng_aug,
            chunk_size=args.aug_chunk_size,
        )
        if cache_path is not None:
            try:
                _save_aug_cache(
                    cache_path,
                    P_aug=P_aug, pk_aug=pk_aug,
                    P_val=P_val, pk_val=pk_val,
                    sids_train=sids_train, sids_val=sids_val,
                    P_holdout=P_holdout, pk_holdout=pk_holdout,
                    sids_holdout=sids_holdout,
                )
            except Exception as exc:
                logger.warning("cache write failed: %s", exc)
        del P_train_src, pk_train_src
        gc.collect()

    logger.info(
        "augmented set: %d sources -> %d samples (%.1fs)",
        n_train_src, P_aug.shape[0], time.time() - started_aug,
    )

    synth_test = _maybe_synthetic_test(args, grid, args.l_max, basis)
    if synth_test is not None and args.scale_factor != 1.0:
        s = float(args.scale_factor); sp = float(np.sqrt(s))
        synth_test = (
            (synth_test[0] * s).astype(np.float32, copy=False),
            (synth_test[1] * sp).astype(np.float32, copy=False),
        )

    # ------------------- features + dataloaders ------------------------------
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    feat = _build_features(args.features, grid=grid, l_max=args.l_max)
    feat.fit(P_train=P_aug)
    z_train = feat.transform(P=P_aug)
    z_val = feat.transform(P=P_val)
    z_synth_test = feat.transform(P=synth_test[0]) if synth_test is not None else None
    z_holdout = feat.transform(P=P_holdout) if P_holdout.shape[0] > 0 else None

    train_ds = _ArrayDataset(z_train, pk_aug, P_aug)
    val_ds = _ArrayDataset(z_val, pk_val, P_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    steps_per_epoch = max(
        1, (len(train_ds) + args.batch_size - 1) // args.batch_size
    )

    # ------------------- model + optional transplant -------------------------
    groups = _resolve_groups(args.groups, args.l_max)
    model = _build_multi_head_model(
        args, input_dim=feat.feature_dim, l_max=args.l_max, groups=groups,
    )
    transplant_info: dict[str, list[int] | bool] | None = None
    if args.transplant_from is not None:
        src = _load_transplant_source(
            Path(args.transplant_from), which=args.transplant_which
        )
        transplant_info = transplant_heads(
            src, model,
            freeze_src_heads=True,
            copy_backbone=not args.no_copy_backbone_on_transplant,
        )
        logger.info("transplant: %s", transplant_info)

    loss_fn, loss_kind, loss_weights = _build_loss_with_overrides(
        args, grid=grid, l_max=args.l_max,
    )

    # ------------------- per-stage checkpoint root ---------------------------
    figures_root = (
        Path(args.figures_dir)
        if args.figures_dir is not None
        else Path("experiments/staged/figures_real_augmented")
    )
    if not args.no_figures:
        figures_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = figures_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _save_model_config(checkpoint_root, model)

    # Persist the fitted feature extractor once (frozen across all stages).
    import pickle as _pickle
    feat_pkl_path = checkpoint_root / "feature_extractor.pkl"
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

    # ------------------- staged training loop --------------------------------
    optim_cfg = OptimiserConfig(
        name=args.optimiser, lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum, nesterov=args.nesterov,
    )
    sched_cfg = SchedulerConfig(
        name=args.scheduler,
        total_steps=0,  # recomputed per stage from steps_per_epoch
        warmup_steps=args.scheduler_warmup_steps,
        min_lr=args.scheduler_min_lr,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
        plateau_patience=args.scheduler_plateau_patience,
        plateau_factor=args.scheduler_plateau_factor,
    )

    backbone_policy: BackbonePolicy = args.backbone_policy  # type: ignore[assignment]
    staged_cfg = StagedTrainerConfig(
        stage_max_epochs=args.stage_max_epochs,
        backbone_policy=backbone_policy,
        backbone_lr_factor=args.backbone_lr_factor,
        reinit_active_head=not args.no_reinit_active_head,
        zero_init_future_heads=not args.no_zero_init_future_heads,
        starting_stage=args.starting_stage,
        checkpoint_root=str(checkpoint_root),
    )
    inner_trainer = Trainer(TrainerConfig(
        max_epochs=args.stage_max_epochs, log_every_n_steps=200,
    ))
    staged_trainer = StagedTrainer(staged_cfg, trainer=inner_trainer)

    # ------------------- eval splits + figure emitter (built once) -----------
    # We build the eval/figure plumbing *before* the stage loop so each
    # per-epoch and per-stage emission can reuse the same fixed sample
    # indices: the figures are then directly comparable across stages and
    # epochs (same train/val/holdout subsets every time).
    decoder = DifferentiableMultipoleField(grid=grid, l_max=args.l_max, basis=basis)
    n_train_eval = min(args.n_train_eval_samples, P_aug.shape[0])
    eval_rng = np.random.default_rng(args.seed)
    train_eval_idx = eval_rng.choice(
        P_aug.shape[0], size=n_train_eval, replace=False
    )
    z_train_eval = z_train[train_eval_idx]
    pk_train_eval = pk_aug[train_eval_idx]
    P_train_eval = P_aug[train_eval_idx]

    n_bins_metric = 2 * args.l_max + 1
    figure_splits: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[str]]] = [
        ("train_aug", z_train_eval, pk_train_eval, P_train_eval, []),
        ("val_real", z_val, pk_val, P_val, list(sids_val)),
    ]
    if z_holdout is not None and P_holdout.shape[0] > 0:
        figure_splits.append(
            ("holdout_real", z_holdout, pk_holdout, P_holdout,
             list(sids_holdout))
        )
    if synth_test is not None and z_synth_test is not None:
        figure_splits.append(
            ("synthetic_test", z_synth_test, synth_test[1], synth_test[0], [])
        )

    def _emit_all_figures(out_dir: Path, *, epoch_label: str = "") -> None:
        """Render the full per-split figure suite + cross-split distribution
        figures (R² histogram, hard-bin accuracy histogram) into ``out_dir``.

        Mirrors :func:`scripts.run_real_augmented._emit_all_figures` so the
        per-stage / per-epoch artefacts here are visually identical to the
        baseline run's. Captures ``model``, ``decoder``, ``args``,
        ``eval_rng``, ``figure_splits``, ``n_bins_metric``, ``grid`` from the
        enclosing scope. Leaves ``model`` in eval mode; mid-train callers
        must call ``model.train()`` afterwards.
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
                model, decoder, z[sub_idx], batch_size=args.eval_batch_size,
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
            bin_acc_per_split[tag] = np.asarray(sample_bin_acc, dtype=np.float64)
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
            f"model=multi_head_mlp, loss={args.loss}{title_suffix}"
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

    @dataclass(slots=True)
    class _PerEpochFiguresCallback(_CallbackBase):
        """Emit ``_emit_all_figures`` every ``every_n_epochs`` *within* one
        stage. ``stage_figures_dir`` is the per-stage root (so per-epoch
        figures land under ``stage_<k>/epoch_<NNNN>/``).
        """

        every_n_epochs: int
        stage_figures_dir: Path

        def on_epoch_end(self, ctx) -> None:
            if self.every_n_epochs <= 0:
                return
            if ctx.epoch <= 0:
                return
            if (ctx.epoch % self.every_n_epochs) != 0:
                return
            epoch_dir = self.stage_figures_dir / f"epoch_{ctx.epoch:04d}"
            started = time.time()
            _emit_all_figures(epoch_dir, epoch_label=f"epoch {ctx.epoch}")
            logger.info(
                "per-epoch figures: epoch=%d wrote %s in %.1fs",
                ctx.epoch, epoch_dir, time.time() - started,
            )
            ctx.model.train()

    def callbacks_factory(stage_idx: int) -> list[Any]:
        stage_ckpt_dir = checkpoint_root / f"stage_{stage_idx:02d}"
        stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
        cbs: list[Any] = [
            LoggingCallback(log_every_n_steps=200),
            ValidationCallback(every_n_epochs=1),
            GradClipCallback(max_norm=1.0),
            EarlyStoppingCallback(patience=args.early_stop_patience),
            CheckpointCallback(
                output_dir=str(stage_ckpt_dir),
                save_every_n_epochs=int(args.checkpoint_every_n_epochs),
                keep_last=int(args.keep_last_checkpoints),
                save_best_metric="val/loss",
                higher_is_better=False,
            ),
        ]
        if not args.no_figures and args.figures_every_n_epochs > 0:
            cbs.append(_PerEpochFiguresCallback(
                every_n_epochs=int(args.figures_every_n_epochs),
                stage_figures_dir=figures_root / f"stage_{stage_idx:02d}",
            ))
        return cbs

    # ------------------- staged loop with per-stage figure emission ----------
    # We unroll StagedTrainer.fit here so we can call _emit_all_figures into
    # ``stage_<k>/`` after every stage. This keeps the staged-trainer module
    # itself agnostic of figure plumbing.
    if not args.no_figures and args.figures_every_n_epochs > 0:
        logger.info(
            "per-epoch figures: enabled every %d epochs into %s/stage_<k>/epoch_<NNNN>/",
            args.figures_every_n_epochs, figures_root,
        )

    started_train = time.time()
    reports = []
    end_stage = (
        args.starting_stage if args.first_stage_only
        else model.n_heads
    )
    for stage_idx in range(args.starting_stage, end_stage + 1):
        report = staged_trainer.fit_one_stage(
            model,
            stage_idx=stage_idx,
            train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, loss_kind=loss_kind,
            optim_cfg=optim_cfg, sched_cfg=sched_cfg,
            callbacks=callbacks_factory(stage_idx),
            steps_per_epoch=steps_per_epoch,
        )
        reports.append(report)
        if not args.no_figures and not args.no_figures_after_each_stage:
            stage_fig_dir = figures_root / f"stage_{stage_idx:02d}"
            started_fig = time.time()
            _emit_all_figures(
                stage_fig_dir,
                epoch_label=f"stage {stage_idx} end (l-band {report.group})",
            )
            logger.info(
                "per-stage figures: stage=%d wrote %s in %.1fs",
                stage_idx, stage_fig_dir, time.time() - started_fig,
            )
            model.train()
    train_elapsed = time.time() - started_train
    logger.info("staged training finished in %.1fs (stages run: %d)",
                train_elapsed, len(reports))

    # ------------------- final evaluation ------------------------------------
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

    # ------------------- end-of-training figure suite ------------------------
    if not args.no_figures:
        final_dir = figures_root / "final"
        _emit_all_figures(final_dir, epoch_label="final")
        logger.info("end-of-training figures written under %s", final_dir)

    # ------------------- write JSON record + manifest ------------------------
    record = {
        "ok": True,
        "elapsed_s_aug": time.time() - started_aug,
        "elapsed_s_train": train_elapsed,
        "feat_dim": int(feat.feature_dim),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "model": "multi_head_mlp",
        "loss": args.loss,
        "loss_weights": loss_weights,
        "features": args.features,
        "regime": "real_augmented_staged" + (" (smoke)" if args.smoke_test else ""),
        "n_real_loaded": int(n_real),
        "n_train_sources": int(n_train_src),
        "n_val_sources": int(n_val_src),
        "n_holdout_sources": int(n_holdout),
        "n_augmented": int(P_aug.shape[0]),
        "l_max": int(args.l_max),
        "groups": groups,
        "backbone_policy": args.backbone_policy,
        "starting_stage": int(args.starting_stage),
        "stage_max_epochs": int(args.stage_max_epochs),
        "transplant_from": args.transplant_from,
        "transplant_info": transplant_info,
        "stage_reports": [
            {
                "stage_idx": r.stage_idx,
                "active_head_idx": r.active_head_idx,
                "group": r.group,
                "epochs_run": r.epochs_run,
                "n_trainable_params": r.n_trainable_params,
                "stop_requested": r.stop_requested,
                "trainable_summary": r.trainable_summary,
                "last_eval_metrics": {
                    k: float(v) for k, v in r.last_eval_metrics.items()
                },
            }
            for r in reports
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    logger.info("results written to %s", out_path)

    manifest = {
        "args": vars(args),
        "feature_dim": int(feat.feature_dim),
        "feature_extractor_pkl": "feature_extractor.pkl",
        "model_config_json": "model_config.json",
        "stage_dirs": [f"stage_{r.stage_idx:02d}" for r in reports],
        "metrics": {k: float(v) for k, v in metrics.items()},
        "elapsed_s_train": float(train_elapsed),
    }
    (checkpoint_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
