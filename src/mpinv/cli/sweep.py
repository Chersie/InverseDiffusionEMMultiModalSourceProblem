"""``mpinv-sweep``: Hydra-driven Optuna HPO with nested MLflow runs.

Why custom: ``hydra-optuna-sweeper`` 1.2.0 pins ``optuna<3``, which is incompatible
with Optuna 4.x and is effectively unmaintained (R2b in
``research/framework-rebuild/manifest.md``). Instead we run Optuna 4.x directly,
sample search-space dimensions per ``cfg.sweep``, mutate the in-memory Hydra cfg,
and call the same ``train.main`` body that a single ``mpinv-train`` invocation
uses. Each trial opens its own nested MLflow run inside the sweep's parent run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from mpinv.tracking.mlflow_sink import MLflowSink, MLflowSinkConfig
from mpinv.tracking.params import flatten_for_mlflow

logger = logging.getLogger("mpinv.sweep")

_CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _sample(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    t = spec["type"]
    if t == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if t == "int":
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
        )
    if t == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=spec.get("step"),
            log=bool(spec.get("log", False)),
        )
    raise ValueError(f"unknown search-space type {t!r} for {name!r}")


def _set_dotted(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def _make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"unknown sampler {name!r}")


def _make_pruner(name: str) -> optuna.pruners.BasePruner:
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name == "nop":
        return optuna.pruners.NopPruner()
    raise ValueError(f"unknown pruner {name!r}")


@hydra.main(version_base="1.3", config_path=_CONFIGS_DIR, config_name="sweep/optuna_mlp")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    sweep_cfg = cfg.sweep
    parent_run_id: str | None = None

    # 1. Open the parent MLflow run so trials can be nested under it.
    tracking_disabled = bool(cfg.tracking.get("disabled", False))
    parent_sink: MLflowSink | None = None
    if not tracking_disabled:
        mlf = OmegaConf.to_container(cfg.tracking, resolve=True)
        keys = {
            "tracking_uri",
            "experiment_name",
            "run_name",
            "tags",
            "log_system_metrics",
        }
        parent_cfg = MLflowSinkConfig(
            **{k: mlf[k] for k in keys if k in mlf},
            run_name=str(sweep_cfg.study_name),
        )
        parent_sink = MLflowSink(parent_cfg)
        parent_sink.on_run_start(flatten_for_mlflow(OmegaConf.to_container(cfg, resolve=True)))
        parent_run_id = parent_sink._run.info.run_id  # type: ignore[union-attr]
        parent_sink.log_metric("sweep/n_trials_target", int(sweep_cfg.n_trials))

    # 2. Build the study.
    sampler = _make_sampler(str(sweep_cfg.sampler), int(sweep_cfg.seed))
    pruner = _make_pruner(str(sweep_cfg.pruner))
    storage = sweep_cfg.get("storage", None)
    study = optuna.create_study(
        study_name=str(sweep_cfg.study_name),
        direction=str(sweep_cfg.direction),
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(sweep_cfg.get("load_if_exists", True)),
    )

    # 3. Define the objective. Each trial deep-copies cfg, applies the sampled
    #    overrides, and runs the same training body as `mpinv-train`.
    from mpinv.cli.train import main as train_main

    def objective(trial: optuna.Trial) -> float:
        trial_cfg: DictConfig = deepcopy(cfg)
        for name, spec in sweep_cfg.search_space.items():
            v = _sample(trial, name, OmegaConf.to_container(spec, resolve=True))
            _set_dotted(trial_cfg, name, v)
        # Force every trial run to be nested under the sweep parent.
        if parent_run_id is not None:
            with OmegaConf.open_dict(trial_cfg):
                trial_cfg.tracking = OmegaConf.merge(
                    trial_cfg.tracking,
                    {
                        "nested": True,
                        "parent_run_id": parent_run_id,
                        "run_name": f"trial_{trial.number:03d}",
                        "tags": dict(trial_cfg.tracking.get("tags", {}))
                        | {
                            "mpinv.run_kind": "hpo_trial",
                            "mpinv.trial_number": str(trial.number),
                        },
                    },
                )
        try:
            return float(train_main(trial_cfg))
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.exception("trial %d failed: %s", trial.number, exc)
            return float("inf") if str(sweep_cfg.direction) == "minimize" else -float("inf")

    # 4. Optimise.
    study.optimize(objective, n_trials=int(sweep_cfg.n_trials), n_jobs=int(sweep_cfg.n_jobs))

    # 5. Log study summary.
    if parent_sink is not None:
        try:
            best = study.best_trial
            parent_sink.log_metric("sweep/best_value", float(best.value))
            parent_sink.log_dict(
                {
                    "best_value": float(best.value),
                    "best_params": dict(best.params),
                    "n_trials": len(study.trials),
                },
                "sweep_summary.json",
            )
            parent_sink.log_metric("sweep/n_trials", len(study.trials))
        finally:
            parent_sink.on_run_end("FINISHED")


if __name__ == "__main__":
    main()
