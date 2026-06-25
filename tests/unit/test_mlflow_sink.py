"""Smoke tests for the MLflow sink against a local file:// store.

Uses a temp directory so the test is hermetic and does not require a running
mlflow server.
"""

from __future__ import annotations

import pytest

from mpinv.tracking.mlflow_sink import MLflowSink, MLflowSinkConfig


@pytest.fixture
def file_store_uri(tmp_path):
    # MLflow 3.12 emits a FutureWarning for plain `file:...` stores; use sqlite which
    # is the production backend per R3 in research/framework-rebuild/manifest.md.
    db = tmp_path / "mlflow.db"
    return f"sqlite:///{db}"


def test_mlflow_sink_run_lifecycle(file_store_uri, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = MLflowSinkConfig(
        tracking_uri=file_store_uri,
        experiment_name="mpinv-test",
        run_name="smoke",
    )
    sink = MLflowSink(cfg)
    sink.on_run_start({"learning_rate": "0.001", "model": "mlp"})
    sink.log_metric("train/loss", 1.0, step=0)
    sink.log_metric("train/loss", 0.5, step=1)
    sink.log_metrics({"val/loss": 0.4, "val/coef_mse": 0.3}, step=1)
    sink.log_dict({"k": "v"}, "config.yaml")
    sink.on_run_end("FINISHED")


def test_mlflow_sink_handles_metrics_without_run(file_store_uri):
    cfg = MLflowSinkConfig(tracking_uri=file_store_uri, experiment_name="mpinv-test")
    sink = MLflowSink(cfg)
    sink.on_run_start({})
    sink.log_metric("a", 1.0)
    sink.on_run_end()


def test_mlflow_sink_translates_early_stopped_to_finished(file_store_uri, tmp_path, monkeypatch):
    """Regression: the trainer's finally-block emits ``EARLY_STOPPED`` when
    EarlyStoppingCallback trips, but MLflow's ``RunStatus.from_string`` only
    accepts {RUNNING, SCHEDULED, FINISHED, FAILED, KILLED}. The sink must
    translate the trainer's richer vocabulary into a valid MLflow status and
    record the original as a run tag.
    """
    import mlflow

    monkeypatch.chdir(tmp_path)
    cfg = MLflowSinkConfig(
        tracking_uri=file_store_uri,
        experiment_name="mpinv-test-early-stop",
        run_name="early-stop-smoke",
    )
    sink = MLflowSink(cfg)
    sink.on_run_start({})
    run_id = sink._run.info.run_id  # pyright: ignore[reportOptionalMemberAccess]
    sink.on_run_end("EARLY_STOPPED")
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags.get("mpinv.run_status") == "EARLY_STOPPED"


def test_mlflow_sink_unknown_status_falls_back_to_failed(file_store_uri, tmp_path, monkeypatch):
    """Defensive: any string that is neither valid MLflow nor in the alias
    map is recorded as FAILED, with the original preserved as a tag."""
    import mlflow

    monkeypatch.chdir(tmp_path)
    cfg = MLflowSinkConfig(
        tracking_uri=file_store_uri,
        experiment_name="mpinv-test-unknown-status",
    )
    sink = MLflowSink(cfg)
    sink.on_run_start({})
    run_id = sink._run.info.run_id  # pyright: ignore[reportOptionalMemberAccess]
    sink.on_run_end("DEFINITELY_NOT_A_REAL_STATUS")
    run = mlflow.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags.get("mpinv.run_status") == "DEFINITELY_NOT_A_REAL_STATUS"
