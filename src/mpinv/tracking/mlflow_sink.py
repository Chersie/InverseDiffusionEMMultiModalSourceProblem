"""MLflow sink implementing the ``Sink`` protocol.

Backed by R3 in research/framework-rebuild/manifest.md. Uses the verified MLflow
3.x API surface: ``set_tracking_uri``, ``set_experiment``, ``start_run``,
``log_param(s)``, ``log_metric(s)``, ``log_figure``, ``log_dict``,
``log_artifact``, ``mlflow.pyfunc.log_model(name=...)`` (the legacy
``artifact_path=`` is deprecated as of MLflow 3.x). Stages are deprecated; we use
``MlflowClient.set_registered_model_alias`` for promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLflowSinkConfig:
    """Knobs for :class:`MLflowSink`."""

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "mpinv"
    run_name: str | None = None
    nested: bool = False
    parent_run_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    log_system_metrics: bool = False


class MLflowSink:
    """Sink that opens an MLflow run on ``on_run_start`` and closes it on
    ``on_run_end``. Cadence-friendly metric logging via ``log_metric(s)``.
    """

    def __init__(self, cfg: MLflowSinkConfig | None = None):
        self.cfg = cfg or MLflowSinkConfig()
        self._run = None

    def on_run_start(self, params: dict[str, str] | None = None) -> None:
        import mlflow

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)
        self._run = mlflow.start_run(
            run_name=self.cfg.run_name,
            nested=self.cfg.nested,
            parent_run_id=self.cfg.parent_run_id,
            tags=self.cfg.tags or None,
            log_system_metrics=self.cfg.log_system_metrics,
        )
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        logger.info("MLflow run started: %s (uri=%s)", self._run.info.run_id, self.cfg.tracking_uri)

    def on_fit_start(self, ctx: Any) -> None:
        if self._run is None:
            self.on_run_start({})

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        import mlflow

        mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_figure(self, fig: Any, artifact_file: str) -> None:
        import mlflow

        mlflow.log_figure(fig, artifact_file)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        import mlflow

        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, d: dict[str, Any], artifact_file: str) -> None:
        import mlflow

        mlflow.log_dict(d, artifact_file)

    def on_epoch_end(self, ctx: Any) -> None: ...

    def log_pyfunc_model(
        self,
        python_model: Any,
        artifacts: dict[str, str] | None = None,
        input_example: Any = None,
        signature: Any = None,
        pip_requirements: list[str] | None = None,
        registered_model_name: str | None = None,
        name: str = "model",
    ) -> None:
        import mlflow

        mlflow.pyfunc.log_model(  # type: ignore[attr-defined]
            name=name,
            python_model=python_model,
            artifacts=artifacts,
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_requirements,
            registered_model_name=registered_model_name,
        )

    def set_alias(self, registered_name: str, alias: str, version: int) -> None:
        from mlflow import MlflowClient

        MlflowClient().set_registered_model_alias(
            name=registered_name, alias=alias, version=str(version)
        )

    # MLflow only accepts the five canonical run statuses; anything else
    # raises in `RunStatus.from_string`. We translate the trainer's richer
    # vocabulary into a valid MLflow status and stash the original on the
    # run as a tag so the early-stop signal is not lost downstream.
    _MLFLOW_VALID_STATUSES = frozenset(
        {"RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"}
    )
    _STATUS_ALIASES = {
        # The trainer emits "EARLY_STOPPED" via Trainer.fit's finally block when
        # EarlyStoppingCallback trips. The run did finish cleanly — the model
        # converged and stopped per design — so it maps to FINISHED.
        "EARLY_STOPPED": "FINISHED",
    }

    def on_run_end(self, status: str = "FINISHED") -> None:
        import mlflow

        if self._run is None:
            return
        original = str(status).upper()
        mapped = self._STATUS_ALIASES.get(original, original)
        if mapped not in self._MLFLOW_VALID_STATUSES:
            logger.warning(
                "MLflowSink: unknown run status %r; recording as FAILED",
                original,
            )
            mapped = "FAILED"
        if original != mapped:
            try:
                mlflow.set_tag("mpinv.run_status", original)
            except Exception:
                pass
        mlflow.end_run(status=mapped)
        self._run = None
