"""Regenerate demonstration plots for a trained model, mirroring ``mpinv-train``.

This script reproduces the *exact* report-generation path used by the training
entrypoint (:mod:`mpinv.cli.train`). Rather than loading a non-existent
``feature_extractor.pkl``, it reconstructs the pipeline from the resolved Hydra
config that every run writes to ``<run_dir>/.hydra/config.yaml``:

1. Load the resolved config from the checkpoint's run directory.
2. Instantiate the data pipeline (``hydra.utils.instantiate(cfg.data)``).
3. Rebuild the feature extractor (``_build_features``) and fit it on
   ``data["P_train"]`` — identical to training.
4. Build the model with ``input_dim = feat.feature_dim`` and load the checkpoint
   weights (``state["model"]``).
5. Build the :class:`DifferentiableMultipoleField` decoder.
6. Run chunked prediction over each split and emit, per split, the same
   ``build_split_report`` PDFs, plus the five cross-split distribution figures.

Usage
-----
    uv run python paper/demonstration/generate_plots.py \\
        --model-checkpoint outputs/final_step0_base/<ts>/checkpoints/best.pt \\
        --output-dir paper/demonstration/figures/step0

Arguments
---------
--model-checkpoint : Path to the checkpoint ``.pt`` (e.g. ``checkpoints/best.pt``).
--config           : Optional explicit path to the resolved ``config.yaml``.
                     Defaults to ``<checkpoint>/../../.hydra/config.yaml``.
--output-dir       : Directory for the regenerated ``report/`` tree.
                     Defaults to ``<run_dir>/report_regen``.
--splits           : Optional comma-separated subset of splits to evaluate
                     (e.g. ``val,holdout``). Defaults to every available split.

Output
------
Under ``<output-dir>/report/`` the script writes, per split ``<tag>``:
``<tag>/coef_scatter.pdf``, ``<tag>/per_l_breakdown.pdf``,
``<tag>/field_comparison_grid.pdf``, ``<tag>/coef_histograms.pdf`` and, at the
report root, ``r2_distribution.pdf``, ``bin_accuracy_distribution.pdf``,
``spearman_distribution.pdf``, ``nrmse_distribution.pdf`` and
``coef_mse_distribution.pdf``. Aggregated metrics are written to
``<output-dir>/metrics_regen.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from mpinv.analysis.plots.r2_distribution import (  # noqa: E402
    build_bin_accuracy_distribution_figure,
    build_coef_mse_distribution_figure,
    build_nrmse_distribution_figure,
    build_r2_distribution_figure,
    build_spearman_distribution_figure,
)
from mpinv.analysis.reports.run_report import (  # noqa: E402
    RunArtifacts,
    build_split_report,
)
from mpinv.cli._configstore import register_configs  # noqa: E402
from mpinv.cli.train import _build_features  # noqa: E402
from mpinv.core.seeds import set_global_seed  # noqa: E402
from mpinv.losses.differentiable_field import DifferentiableMultipoleField  # noqa: E402

logger = logging.getLogger("mpinv.demo.generate_plots")


# ---------------------------------------------------------------------------
# Config / checkpoint discovery
# ---------------------------------------------------------------------------
def _resolve_config_path(checkpoint: Path, explicit: Path | None) -> Path:
    """Locate the resolved Hydra ``config.yaml`` for a checkpoint.

    Layout written by every run is ``<run_dir>/checkpoints/<name>.pt`` next to
    ``<run_dir>/.hydra/config.yaml``.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"config not found: {explicit}")
        return explicit
    candidate = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"could not auto-locate config at {candidate}; pass --config explicitly"
        )
    return candidate


def _load_config(config_path: Path, output_dir: Path, dry_run: bool = False) -> DictConfig:
    """Load the resolved config and neutralise Hydra-only interpolations.

    The on-disk config carries ``output_dir: ${hydra:runtime.output_dir}`` and a
    ``run_name: ${now:...}`` that only resolve inside a live Hydra run. We
    overwrite them with concrete values so downstream access never triggers an
    unknown-resolver error.

    If ``dry_run`` is True, override dataset sizes to small values for fast testing.
    """
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = str(output_dir)
    if "run_name" in cfg:
        cfg.run_name = "regen"
    if "experiment_name" not in cfg:
        cfg.experiment_name = config_path.parent.parent.name

    if dry_run:
        # Override dataset sizes for fast dry-run
        if "data" in cfg:
            cfg.data.n_source = 20
            cfg.data.n_train_sources = 15
            cfg.data.n_augmented = 100
            cfg.data.n_holdout_samples = 10
            cfg.data.include_synthetic_test = False
            cfg.data.include_dummy_probe = False
        logger.info("dry-run mode: dataset sizes reduced (n_source=20, n_augmented=100)")

    return cfg


# ---------------------------------------------------------------------------
# Model construction (mirrors mpinv.cli.train.main, lines 175-207)
# ---------------------------------------------------------------------------
def _build_model(cfg: DictConfig, feat: Any, data: dict[str, Any]):
    K = data["l_max"] * (data["l_max"] + 2)
    output_dim = 4 * K
    model_target = cfg.model._target_
    model_cfg = OmegaConf.to_container(cfg.model.cfg, resolve=False)
    model_cfg.pop("_target_", None)
    model_cfg["input_dim"] = feat.feature_dim
    model_cfg["output_dim"] = output_dim

    if model_target == "mpinv.models.mlp.MLP":
        from mpinv.models.mlp import MLP, MLPConfig

        return MLP(MLPConfig(**model_cfg))
    if model_target == "mpinv.models.linear_baselines.LinearBaseline":
        from mpinv.models.linear_baselines import LinearBaseline, LinearBaselineConfig

        return LinearBaseline(LinearBaselineConfig(**model_cfg))
    if model_target == "mpinv.models.multi_head_mlp.MultiHeadMLP":
        from mpinv.models.multi_head_mlp import MultiHeadMLP, MultiHeadMLPConfig

        groups = model_cfg.get("groups", None)
        if groups is not None:
            model_cfg["groups"] = [[int(l) for l in g] for g in groups]
        cfg_l_max = int(model_cfg.get("l_max", data["l_max"]))
        if cfg_l_max != data["l_max"]:
            raise ValueError(
                f"multi_head_mlp config l_max={cfg_l_max} disagrees with data "
                f"l_max={data['l_max']}"
            )
        model_cfg["l_max"] = cfg_l_max
        return MultiHeadMLP(MultiHeadMLPConfig(**model_cfg))
    raise ValueError(f"unknown model target: {model_target!r}")


def _load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:  # bare state_dict fallback
        model.load_state_dict(state)
    model.eval()


# ---------------------------------------------------------------------------
# Report generation (mirrors mpinv.cli.train.main, lines 389-594)
# ---------------------------------------------------------------------------
def _generate_report(
    cfg: DictConfig,
    data: dict[str, Any],
    feat: Any,
    model: torch.nn.Module,
    z_train: np.ndarray,
    z_val: np.ndarray,
    report_dir: Path,
    only_splits: set[str] | None,
) -> dict[str, float]:
    report_cfg = cfg.get("report", {}) or {}
    n_train_eval_samples = int(report_cfg.get("n_train_eval_samples", 1024))
    n_grid_samples = int(report_cfg.get("n_grid_samples", 8))
    eval_batch_size = int(report_cfg.get("eval_batch_size", 256))
    report_dir.mkdir(parents=True, exist_ok=True)

    decoder = DifferentiableMultipoleField(
        grid=data["grid"],
        l_max=data["l_max"],
        basis=data["basis"],
    )

    def _predict_chunked(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds_list: list[np.ndarray] = []
        field_list: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for i in range(0, z.shape[0], eval_batch_size):
                zb = torch.from_numpy(z[i : i + eval_batch_size]).float()
                pb = model(zb).cpu().numpy()
                preds_list.append(pb)
                field_list.append(decoder(torch.from_numpy(pb).float()).cpu().numpy())
        return np.concatenate(preds_list, axis=0), np.concatenate(field_list, axis=0)

    # (tag, P_true, packed_true, z_features, dummy_active)
    splits_to_eval: list[
        tuple[str, np.ndarray, np.ndarray, np.ndarray, list[int] | None]
    ] = []
    if data["P_train"].shape[0] > 0:
        n_eval_train = min(n_train_eval_samples, data["P_train"].shape[0])
        splits_to_eval.append((
            "train_aug",
            data["P_train"][:n_eval_train],
            data["packed_train"][:n_eval_train],
            z_train[:n_eval_train],
            None,
        ))
    splits_to_eval.append(("val", data["P_val"], data["packed_val"], z_val, None))

    if "P_test" in data and "packed_test" in data:
        try:
            z_test = feat.transform(P=data["P_test"])
            splits_to_eval.append(
                ("test", data["P_test"], data["packed_test"], z_test, None)
            )
        except Exception as exc:
            logger.warning("synthetic test feature transform failed: %s", exc)

    if (
        "P_holdout" in data
        and "packed_holdout" in data
        and data["P_holdout"].shape[0] > 0
    ):
        try:
            z_ho = feat.transform(P=data["P_holdout"])
            splits_to_eval.append(
                ("holdout", data["P_holdout"], data["packed_holdout"], z_ho, None)
            )
        except Exception as exc:
            logger.warning("holdout feature transform failed: %s", exc)

    if "P_dummy" in data and "packed_dummy" in data:
        try:
            z_dummy = feat.transform(P=data["P_dummy"])
            splits_to_eval.append((
                "dummy",
                data["P_dummy"],
                data["packed_dummy"],
                z_dummy,
                list(
                    data.get(
                        "dummy_active_indices", range(data["P_dummy"].shape[0])
                    )
                ),
            ))
        except Exception as exc:
            logger.warning("dummy feature transform failed: %s", exc)

    if only_splits is not None:
        splits_to_eval = [s for s in splits_to_eval if s[0] in only_splits]
        if not splits_to_eval:
            raise ValueError(
                f"no requested splits {sorted(only_splits)} are available"
            )

    aggregated_metrics: dict[str, float] = {}
    per_sample_pool: dict[str, dict[str, np.ndarray]] = {}
    for tag, P_true, packed_t, z_in, dummy_idx in splits_to_eval:
        try:
            preds_split, P_pred_split = _predict_chunked(z_in)
            art = RunArtifacts(
                pred_packed=preds_split,
                target_packed=packed_t,
                P_pred=P_pred_split,
                P_true=P_true,
                l_max=data["l_max"],
                grid=data["grid"],
                pca_explained_variance_ratio=(
                    getattr(feat, "explained_variance_ratio_", None)
                    if tag == "val"
                    else None
                ),
            )
            metrics_split, per_sample_split = build_split_report(
                art,
                output_dir=report_dir,
                split=tag,
                sink=None,
                n_grid_samples=n_grid_samples,
                dummy_active_indices=dummy_idx,
            )
            for k, v in metrics_split.items():
                logger.info("%s = %.6f", k, v)
            aggregated_metrics.update({k: float(v) for k, v in metrics_split.items()})
            per_sample_pool[tag] = per_sample_split
            logger.info("split %s: %d samples reported", tag, P_true.shape[0])
        except Exception as exc:
            logger.warning("split %s eval failed: %s", tag, exc)

    # Cross-split distribution figures (histogram + violin).
    if per_sample_pool:
        n_bins_metric = 2 * data["l_max"] + 1
        exp_name = cfg.get("experiment_name", "regen")
        fig_specs = [
            (
                "r2_distribution.pdf",
                {tag: ps["r2"] for tag, ps in per_sample_pool.items()},
                lambda d, t: build_r2_distribution_figure(d, title=t),
                f"R² distribution across splits — {exp_name}",
            ),
            (
                "bin_accuracy_distribution.pdf",
                {tag: ps["bin_accuracy"] for tag, ps in per_sample_pool.items()},
                lambda d, t: build_bin_accuracy_distribution_figure(
                    d, n_bins_metric=n_bins_metric, title=t
                ),
                f"Hard rank-bin accuracy (n_bins={n_bins_metric}) across splits "
                f"— {exp_name}",
            ),
            (
                "spearman_distribution.pdf",
                {tag: ps["spearman_rho"] for tag, ps in per_sample_pool.items()},
                lambda d, t: build_spearman_distribution_figure(d, title=t),
                f"Spearman rho across splits — {exp_name}",
            ),
            (
                "nrmse_distribution.pdf",
                {tag: ps["nrmse"] for tag, ps in per_sample_pool.items()},
                lambda d, t: build_nrmse_distribution_figure(d, title=t),
                f"NRMSE_w across splits — {exp_name}",
            ),
            (
                "coef_mse_distribution.pdf",
                {tag: ps["coef_mse"] for tag, ps in per_sample_pool.items()},
                lambda d, t: build_coef_mse_distribution_figure(d, title=t),
                f"Coef MSE per sample across splits — {exp_name}",
            ),
        ]
        for filename, payload, builder, title in fig_specs:
            try:
                fig = builder(payload, title)
                fig.savefig(report_dir / filename, bbox_inches="tight")
                plt.close(fig)
                logger.info("wrote %s", report_dir / filename)
            except Exception as exc:
                logger.warning("run-level %s build failed: %s", filename, exc)

    return aggregated_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate per-split + cross-split report figures for a trained "
            "model, reproducing the mpinv-train report path."
        )
    )
    parser.add_argument(
        "--model-checkpoint",
        required=True,
        type=Path,
        help="Path to the checkpoint .pt (e.g. <run>/checkpoints/best.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit resolved config.yaml; defaults to <run>/.hydra/config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root; defaults to <run_dir>/report_regen.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated subset of splits to evaluate (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reduce dataset sizes for fast testing (n_source=20, n_augmented=100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)

    checkpoint = args.model_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    run_dir = checkpoint.parent.parent
    output_dir = (args.output_dir or (run_dir / "report_regen")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_config_path(checkpoint, args.config)
    logger.info("checkpoint : %s", checkpoint)
    logger.info("config     : %s", config_path)
    logger.info("output dir : %s", output_dir)

    register_configs()
    cfg = _load_config(config_path, output_dir, dry_run=args.dry_run)
    set_global_seed(int(cfg.get("seed", 0)))

    # 1. data pipeline (identical to train.py)
    logger.info("instantiating data pipeline …")
    data = hydra.utils.instantiate(cfg.data)

    # 2. feature pipeline: rebuild from config and fit on train (NOT a pickle)
    logger.info("rebuilding + fitting feature extractor …")
    feat = _build_features(cfg, data)
    feat.fit(P_train=data["P_train"])
    z_train = feat.transform(P=data["P_train"])
    z_val = feat.transform(P=data["P_val"])

    # 3. model + checkpoint weights
    logger.info("building model and loading checkpoint weights …")
    model = _build_model(cfg, feat, data)
    _load_weights(model, checkpoint)

    # 4. report
    only_splits = (
        {s.strip() for s in args.splits.split(",") if s.strip()}
        if args.splits
        else None
    )
    report_dir = output_dir / "report"
    metrics = _generate_report(
        cfg, data, feat, model, z_train, z_val, report_dir, only_splits
    )

    metrics_path = output_dir / "metrics_regen.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "experiment_name": str(cfg.get("experiment_name", "")),
                "checkpoint": str(checkpoint),
                "config": str(config_path),
            },
            indent=2,
        )
    )
    logger.info("metrics summary written to %s", metrics_path)
    logger.info("done — figures under %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
