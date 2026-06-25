"""``mpinv-train``: end-to-end training entrypoint.

Hydra composes the run config from ``configs/train.yaml`` plus any CLI overrides:

    uv run mpinv-train trainer=fast_dev_run model=mlp_pyramid loss=physics_power

The flow:

1. Build the data pipeline (synthetic generator → in-memory arrays).
2. Fit the feature extractor on training samples; transform train/val.
3. Build the model (input_dim resolved from features.feature_dim).
4. Build the loss; if ``loss.kind = "physics"`` it is constructed against the same
   grid + ``l_max`` as the data.
5. Build the optimiser, scheduler, callbacks, and MLflow sink.
6. ``Trainer.fit(...)``.
7. Log loss curves, the resolved Hydra config, and a ``mlflow.pyfunc`` bundle that
   wraps the trained model + the fitted feature pipeline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mpinv.callbacks.base import Callback
from mpinv.callbacks.checkpoint_cb import CheckpointCallback
from mpinv.callbacks.early_stopping_cb import EarlyStoppingCallback
from mpinv.callbacks.grad_clip_cb import GradClipCallback
from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.callbacks.timing_cb import TimingCallback
from mpinv.callbacks.validation_cb import ValidationCallback
from mpinv.cli._builders import make_loaders
from mpinv.cli._configstore import register_configs
from mpinv.core.seeds import set_global_seed
from mpinv.tracking.dataset_logger import DatasetSpec, log_numpy_dataset
from mpinv.tracking.mlflow_sink import MLflowSink, MLflowSinkConfig
from mpinv.tracking.params import flatten_for_mlflow
from mpinv.training.optim import build_optimiser, build_scheduler
from mpinv.training.trainer import Trainer

_CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _build_features(cfg: DictConfig, data: dict[str, Any]):
    """Construct the feature pipeline. Handles the ``composite`` kind specially
    because the list of extractors is not naturally expressed as a single Hydra
    instantiate call.
    """
    feat_cfg = cfg.features
    kind = feat_cfg.get("kind", None)
    if kind != "composite":
        return hydra.utils.instantiate(feat_cfg)

    from mpinv.features.composite import CompositeFeaturesConfig, CompositePipeline
    from mpinv.features.fft_radial import FFTRadial, FFTRadialConfig
    from mpinv.features.power_pipeline import PowerPCAPipelineConfig
    from mpinv.features.sh_power import SHPower, SHPowerConfig

    pca_cfg = OmegaConf.to_container(feat_cfg.pca, resolve=True) if feat_cfg.get("pca") else {}
    pca_cfg.pop("_target_", None)
    cfg_obj = CompositeFeaturesConfig(
        pca=PowerPCAPipelineConfig(**pca_cfg),
        skip_pca=bool(feat_cfg.get("skip_pca", False)),
        normalise_concat=bool(feat_cfg.get("normalise_concat", True)),
    )
    extractors = []
    for spec in feat_cfg.get("extractors", []):
        spec = OmegaConf.to_container(spec, resolve=True)
        name = spec.pop("name")
        if name == "fft_radial":
            extractors.append(FFTRadial(FFTRadialConfig(**spec)))
        elif name == "sh_power":
            extractors.append(SHPower(SHPowerConfig(**spec), grid=data["grid"]))
        elif name == "hog":
            from mpinv.features.hog import HOGConfig, HOGExtractor

            extractors.append(HOGExtractor(HOGConfig(**spec)))
        elif name == "raw_flat":
            from mpinv.features.raw_flat import RawFlattenPipeline, RawFlattenPipelineConfig

            extractors.append(RawFlattenPipeline(RawFlattenPipelineConfig(**spec)))
        elif name == "subsample_grid":
            from mpinv.features.subsample import (
                SubsampleGridPipeline,
                SubsampleGridPipelineConfig,
            )

            extractors.append(SubsampleGridPipeline(SubsampleGridPipelineConfig(**spec)))
        else:
            raise ValueError(f"unknown composite extractor name {name!r}")
    return CompositePipeline(cfg=cfg_obj, extractors=extractors)


logger = logging.getLogger("mpinv.train")


def _build_optimiser_cfg(d: Mapping[str, Any]):
    from mpinv.training.optim import OptimiserConfig

    keys = {"name", "lr", "weight_decay", "betas", "eps", "momentum", "fused"}
    return OptimiserConfig(**{k: d[k] for k in keys if k in d})


def _build_scheduler_cfg(d: Mapping[str, Any]):
    from mpinv.training.optim import SchedulerConfig

    keys = {
        "name",
        "total_steps",
        "warmup_steps",
        "min_lr",
        "step_size",
        "gamma",
        "plateau_patience",
        "plateau_factor",
    }
    return SchedulerConfig(**{k: d[k] for k in keys if k in d})


def _build_trainer_cfg(d: Mapping[str, Any]):
    from mpinv.training.amp import AMPConfig
    from mpinv.training.trainer import TrainerConfig

    amp_d = dict(d.get("amp", {}))
    amp_d.pop("_target_", None)
    amp = AMPConfig(**amp_d)
    keys = {"max_epochs", "accum_steps", "log_every_n_steps", "sanity_check", "device"}
    return TrainerConfig(amp=amp, **{k: d[k] for k in keys if k in d})


def _build_mlflow_cfg(d: Mapping[str, Any]):
    keys = {
        "tracking_uri",
        "experiment_name",
        "run_name",
        "nested",
        "parent_run_id",
        "tags",
        "log_system_metrics",
    }
    raw = {k: d[k] for k in keys if k in d}
    if "tags" in raw and raw["tags"] is not None:
        raw["tags"] = {str(k): str(v) for k, v in dict(raw["tags"]).items()}
    return MLflowSinkConfig(**raw)


@hydra.main(version_base="1.3", config_path=_CONFIGS_DIR, config_name="train")
def main(cfg: DictConfig) -> float:
    register_configs()
    set_global_seed(int(cfg.seed))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logger.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # 1. data pipeline (returns a plain dict; we control composition explicitly)
    data = hydra.utils.instantiate(cfg.data)

    # 2. feature pipeline (fit on train, transform train+val)
    feat = _build_features(cfg, data)
    feat.fit(P_train=data["P_train"])
    z_train = feat.transform(P=data["P_train"])
    z_val = feat.transform(P=data["P_val"])

    # 3. model — input_dim from features, output_dim from data l_max
    K = data["l_max"] * (data["l_max"] + 2)
    output_dim = 4 * K
    model_target = cfg.model._target_
    model_cfg = OmegaConf.to_container(cfg.model.cfg, resolve=False)
    model_cfg.pop("_target_", None)
    model_cfg["input_dim"] = feat.feature_dim
    model_cfg["output_dim"] = output_dim

    if model_target == "mpinv.models.mlp.MLP":
        from mpinv.models.mlp import MLP, MLPConfig

        model = MLP(MLPConfig(**model_cfg))
    elif model_target == "mpinv.models.linear_baselines.LinearBaseline":
        from mpinv.models.linear_baselines import LinearBaseline, LinearBaselineConfig

        model = LinearBaseline(LinearBaselineConfig(**model_cfg))
    elif model_target == "mpinv.models.multi_head_mlp.MultiHeadMLP":
        from mpinv.models.multi_head_mlp import MultiHeadMLP, MultiHeadMLPConfig

        groups = model_cfg.get("groups", None)
        if groups is not None:
            model_cfg["groups"] = [[int(l) for l in g] for g in groups]
        cfg_l_max = int(model_cfg.get("l_max", data["l_max"]))
        if cfg_l_max != data["l_max"]:
            raise ValueError(
                f"multi_head_mlp config l_max={cfg_l_max} disagrees with data "
                f"l_max={data['l_max']}; either align them or override one"
            )
        model_cfg["l_max"] = cfg_l_max
        model = MultiHeadMLP(MultiHeadMLPConfig(**model_cfg))
    else:
        raise ValueError(f"unknown model target: {model_target!r}")

    # 4. loss — pop the `kind` discriminator out before Hydra instantiates the leaf
    loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
    loss_kind = loss_cfg.pop("kind", "coef")
    if loss_kind == "coef":
        loss_target = loss_cfg.pop("_target_")
        loss_inner_cfg_dict = loss_cfg.pop("cfg", {})
        loss_inner_cfg_dict.pop("_target_", None)
        from mpinv.losses.coef_mse import CoefMSE, CoefMSEConfig

        loss_fn = CoefMSE(CoefMSEConfig(**loss_inner_cfg_dict))
        del loss_target
    elif loss_kind == "physics":
        from mpinv.cli._builders import build_physics_power_loss

        loss_cfg.pop("_target_", None)
        rank_bin_n_bins_raw = loss_cfg.get("rank_bin_n_bins", None)
        rank_bin_n_bins = (
            int(rank_bin_n_bins_raw) if rank_bin_n_bins_raw is not None else None
        )
        truncate_raw = loss_cfg.get("truncate_target_to_band", None)
        truncate_target_to_band = int(truncate_raw) if truncate_raw is not None else None
        loss_fn = build_physics_power_loss(
            grid=data["grid"],
            l_max=data["l_max"],
            log_ratio=bool(loss_cfg.get("log_ratio", False)),
            log_eps=float(loss_cfg.get("log_eps", 1e-12)),
            coef_aux_weight=float(loss_cfg.get("coef_aux_weight", 0.0)),
            rank_bin_weight=float(loss_cfg.get("rank_bin_weight", 0.0)),
            rank_bin_n_bins=rank_bin_n_bins,
            rank_bin_beta=float(loss_cfg.get("rank_bin_beta", 10.0)),
            truncate_target_to_band=truncate_target_to_band,
        )
    else:
        raise ValueError(f"unknown loss kind {loss_kind!r}")

    # 5. optimiser, scheduler, trainer (configs first; staged path defers building
    #    the optimiser/scheduler to per-stage so the live param set matches each
    #    stage's freezing policy).
    opt_cfg = _build_optimiser_cfg(OmegaConf.to_container(cfg.optimiser, resolve=True))
    sched_cfg = _build_scheduler_cfg(OmegaConf.to_container(cfg.scheduler, resolve=True))
    tr_cfg = _build_trainer_cfg(OmegaConf.to_container(cfg.trainer, resolve=True))

    # 6. dataloaders
    train_loader, val_loader = make_loaders(
        P_train=data["P_train"],
        packed_train=data["packed_train"],
        z_train=z_train,
        P_val=data["P_val"],
        packed_val=data["packed_val"],
        z_val=z_val,
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
    )

    # 7. callbacks (factory used by both single-stage and staged branches; the
    #    staged trainer needs *fresh* callbacks per stage because EarlyStopping
    #    and Checkpoint are stateful — see staged.py docstring).
    cb_cfg = cfg.callbacks

    def _make_callbacks(checkpoint_subdir: str) -> list[Callback]:
        out_dir = Path(cfg.output_dir) / checkpoint_subdir
        return [
            LoggingCallback(log_every_n_steps=int(cb_cfg.log_every_n_steps)),
            TimingCallback(),
            ValidationCallback(every_n_epochs=int(cb_cfg.validation_every_n_epochs)),
            GradClipCallback(max_norm=float(cb_cfg.grad_clip_max_norm)),
            CheckpointCallback(
                output_dir=str(out_dir),
                save_every_n_epochs=int(cb_cfg.checkpoint_every_n_epochs),
                keep_last=int(cb_cfg.keep_last_checkpoints),
            ),
            EarlyStoppingCallback(patience=int(cb_cfg.early_stop_patience)),
        ]

    # 8. tracking sink
    tracking_disabled = bool(cfg.tracking.get("disabled", False))
    sinks: list[Any] = []
    sink: MLflowSink | None = None
    if not tracking_disabled:
        mlf_cfg = _build_mlflow_cfg(OmegaConf.to_container(cfg.tracking, resolve=True))
        sink = MLflowSink(mlf_cfg)
        try:
            params_flat = flatten_for_mlflow(OmegaConf.to_container(cfg, resolve=True))
            sink.on_run_start(params_flat)
            sink.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
            log_numpy_dataset(
                z_train,
                data["packed_train"],
                DatasetSpec(name="synthetic_train", context="training"),
            )
            log_numpy_dataset(
                z_val,
                data["packed_val"],
                DatasetSpec(name="synthetic_val", context="validation"),
            )
            sinks.append(sink)
        except Exception as exc:
            logger.warning("MLflow disabled (%s)", exc)
            sink = None

    # Decide single-stage vs staged based on cfg.training (optional).
    training_target = None
    training_node = cfg.get("training", None)
    if training_node is not None:
        training_target = training_node.get("_target_", None)
    is_staged = training_target == "mpinv.training.staged.StagedTrainerConfig"

    if is_staged:
        from mpinv.models.multi_head_mlp import MultiHeadMLP
        from mpinv.training.staged import StagedTrainer, StagedTrainerConfig

        if not isinstance(model, MultiHeadMLP):
            raise ValueError(
                "cfg.training points at StagedTrainerConfig but the model is not a "
                f"MultiHeadMLP (got {type(model).__name__}); pick "
                "model=multi_head_mlp_5x200 or remove cfg.training"
            )
        staged_dict = OmegaConf.to_container(cfg.training, resolve=True)
        staged_dict.pop("_target_", None)
        staged_cfg = StagedTrainerConfig(**staged_dict)
        inner_trainer = Trainer(tr_cfg)
        staged_trainer = StagedTrainer(staged_cfg, trainer=inner_trainer)

        def _callbacks_factory(stage_idx: int) -> list[Callback]:
            return _make_callbacks(f"checkpoints/stage_{stage_idx}")

        def _sinks_factory(_stage_idx: int) -> list[Any]:
            return list(sinks)

        try:
            steps_per_epoch = len(train_loader)  # type: ignore[arg-type]
        except TypeError:
            steps_per_epoch = None

        reports = staged_trainer.fit(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optim_cfg=opt_cfg,
            loss_kind=loss_kind,
            sched_cfg=sched_cfg,
            val_loader=val_loader,
            callbacks_factory=_callbacks_factory,
            sinks_factory=_sinks_factory,
            steps_per_epoch=steps_per_epoch,
        )
        reports_path = Path(cfg.output_dir) / "stage_reports.json"
        reports_path.parent.mkdir(parents=True, exist_ok=True)
        with reports_path.open("w") as f:
            json.dump([asdict(r) for r in reports], f, indent=2)
        logger.info("staged training: %d stage reports written to %s",
                    len(reports), reports_path)

        last_metrics = reports[-1].last_eval_metrics if reports else {}
        final_metric = float(last_metrics.get("val/loss", float("nan")))
    else:
        optimiser = build_optimiser(model, opt_cfg)
        scheduler = build_scheduler(optimiser, sched_cfg)
        trainer = Trainer(tr_cfg)
        callbacks = _make_callbacks("checkpoints")

        ctx = trainer.fit(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimiser=optimiser,
            scheduler=scheduler,
            loss_kind=loss_kind,
            val_loader=val_loader,
            callbacks=callbacks,
            sinks=sinks,
        )

        final_metric = (ctx.last_eval_metrics or {}).get("val/loss", float(ctx.last_loss))

    # 9. End-of-run analysis report - per-split (4 PDFs + worst-to-best
    #    field_comparison_grid) plus run-level cross-split distribution
    #    figures (R-squared + bin-accuracy + Spearman rho + NRMSE + coef-MSE
    #    violins).
    aggregated_metrics: dict[str, float] = {}
    try:
        from mpinv.analysis.plots.r2_distribution import (
            build_bin_accuracy_distribution_figure,
            build_coef_mse_distribution_figure,
            build_nrmse_distribution_figure,
            build_r2_distribution_figure,
            build_spearman_distribution_figure,
        )
        from mpinv.analysis.reports.run_report import RunArtifacts, build_split_report
        from mpinv.losses.differentiable_field import DifferentiableMultipoleField

        report_cfg = cfg.get("report", {}) or {}
        n_train_eval_samples = int(report_cfg.get("n_train_eval_samples", 1024))
        n_grid_samples = int(report_cfg.get("n_grid_samples", 8))
        eval_batch_size = int(report_cfg.get("eval_batch_size", 256))
        report_dir = Path(cfg.output_dir) / "report"
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

        # Build the list of (tag, P_true, packed_true, z_features, dummy_active).
        # Each entry is materialised lazily; train_aug subsample uses the first
        # `n_train_eval_samples` rows so the cell stays under memory budget.
        splits_to_eval: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[int] | None]] = []
        if data["P_train"].shape[0] > 0:
            n_eval_train = min(n_train_eval_samples, data["P_train"].shape[0])
            splits_to_eval.append((
                "train_aug",
                data["P_train"][:n_eval_train],
                data["packed_train"][:n_eval_train],
                z_train[:n_eval_train],
                None,
            ))
        splits_to_eval.append((
            "val", data["P_val"], data["packed_val"], z_val, None,
        ))
        if "P_test" in data and "packed_test" in data:
            try:
                z_test = feat.transform(P=data["P_test"])
                splits_to_eval.append((
                    "test", data["P_test"], data["packed_test"], z_test, None,
                ))
            except Exception as exc:
                logger.warning("synthetic test feature transform failed: %s", exc)

        # Real-antenna holdout: prefer arrays embedded in the data dict; else
        # fall back to the cfg.holdout file-load path for synthetic pipelines.
        holdout_cfg = cfg.get("holdout", None)
        holdout_arrays: tuple[np.ndarray, np.ndarray] | None = None
        if "P_holdout" in data and "packed_holdout" in data and data["P_holdout"].shape[0] > 0:
            holdout_arrays = (data["P_holdout"], data["packed_holdout"])
        elif holdout_cfg is not None and not bool(holdout_cfg.get("disabled", False)):
            try:
                from mpinv.data.real_antenna_loader import (
                    RealAntennaLoaderConfig,
                    load_real_antenna,
                )

                ho_cfg = RealAntennaLoaderConfig(
                    root=str(holdout_cfg.get("root", "data/raw/real_antenna")),
                    feature_subdir=str(holdout_cfg.get("feature_subdir", "E_in_plane")),
                    target_glob=str(holdout_cfg.get("target_glob", "Results_*.txt")),
                    grid=data["grid"],
                    l_max=data["l_max"],
                    shuffle_seed=int(holdout_cfg.get("shuffle_seed", 42)),
                    max_samples=holdout_cfg.get("max_samples"),
                )
                samples = load_real_antenna(ho_cfg)
                if samples:
                    holdout_arrays = (
                        np.stack([s.P for s in samples], axis=0),
                        np.stack([s.packed for s in samples], axis=0),
                    )
                else:
                    logger.warning(
                        "real-antenna holdout requested but no samples found at %s",
                        ho_cfg.root,
                    )
            except Exception as exc:
                logger.warning("real-antenna holdout load failed: %s", exc)

        if holdout_arrays is not None:
            try:
                P_ho, packed_ho = holdout_arrays
                z_ho = feat.transform(P=P_ho)
                splits_to_eval.append(("holdout", P_ho, packed_ho, z_ho, None))
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
                    list(data.get("dummy_active_indices", range(data["P_dummy"].shape[0]))),
                ))
            except Exception as exc:
                logger.warning("dummy feature transform failed: %s", exc)

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
                        getattr(feat, "explained_variance_ratio_", None) if tag == "val" else None
                    ),
                )
                metrics_split, per_sample_split = build_split_report(
                    art,
                    output_dir=report_dir,
                    split=tag,
                    sink=sink,
                    n_grid_samples=n_grid_samples,
                    dummy_active_indices=dummy_idx,
                )
                for k, v in metrics_split.items():
                    logger.info("%s = %.6f", k, v)
                    if sink is not None:
                        try:
                            sink.log_metric(k, float(v))
                        except Exception:
                            pass
                aggregated_metrics.update({k: float(v) for k, v in metrics_split.items()})
                per_sample_pool[tag] = per_sample_split
            except Exception as exc:
                logger.warning("split %s eval failed: %s", tag, exc)

        # Run-level cross-split distribution figures (histogram + violin).
        if per_sample_pool:
            n_bins_metric = 2 * data["l_max"] + 1
            fig_specs = [
                (
                    "r2_distribution.pdf",
                    {tag: ps["r2"] for tag, ps in per_sample_pool.items()},
                    lambda d, t: build_r2_distribution_figure(d, title=t),
                    f"R² distribution across splits — {cfg.experiment_name}",
                ),
                (
                    "bin_accuracy_distribution.pdf",
                    {tag: ps["bin_accuracy"] for tag, ps in per_sample_pool.items()},
                    lambda d, t: build_bin_accuracy_distribution_figure(
                        d, n_bins_metric=n_bins_metric, title=t
                    ),
                    f"Hard rank-bin accuracy (n_bins={n_bins_metric}) across splits "
                    f"— {cfg.experiment_name}",
                ),
                (
                    "spearman_distribution.pdf",
                    {tag: ps["spearman_rho"] for tag, ps in per_sample_pool.items()},
                    lambda d, t: build_spearman_distribution_figure(d, title=t),
                    f"Spearman rho across splits - {cfg.experiment_name}",
                ),
                (
                    "nrmse_distribution.pdf",
                    {tag: ps["nrmse"] for tag, ps in per_sample_pool.items()},
                    lambda d, t: build_nrmse_distribution_figure(d, title=t),
                    f"NRMSE_w across splits — {cfg.experiment_name}",
                ),
                (
                    "coef_mse_distribution.pdf",
                    {tag: ps["coef_mse"] for tag, ps in per_sample_pool.items()},
                    lambda d, t: build_coef_mse_distribution_figure(d, title=t),
                    f"Coef MSE per sample across splits — {cfg.experiment_name}",
                ),
            ]
            for filename, payload, builder, title in fig_specs:
                try:
                    fig = builder(payload, title)
                    fig_path = report_dir / filename
                    fig.savefig(fig_path, bbox_inches="tight")
                    if sink is not None:
                        try:
                            sink.log_figure(fig, f"plots/{filename}")
                        except Exception:
                            pass
                    import matplotlib.pyplot as _plt

                    _plt.close(fig)
                except Exception as exc:
                    logger.warning("run-level %s build failed: %s", filename, exc)
    except Exception as exc:
        logger.warning("report builder failed: %s", exc)

    # Persist the union of all per-split metrics to disk so downstream selection
    # tools (scripts/select_best_step.py) can compare cells without an MLflow
    # round-trip. The file is a flat dict ``{metric_key: float}``; the keys use
    # the canonical ``report/<split>/<metric>`` namespace consistent with MLflow.
    if aggregated_metrics:
        try:
            metrics_path = Path(cfg.output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "metrics": dict(aggregated_metrics),
                "experiment_name": str(cfg.get("experiment_name", "")),
                "run_name": str(cfg.get("run_name", "")),
            }
            metrics_path.write_text(json.dumps(payload, indent=2))
            logger.info("metrics summary written to %s", metrics_path)
        except Exception as exc:
            logger.warning("failed to persist metrics.json: %s", exc)

    return float(final_metric)


if __name__ == "__main__":
    main()
