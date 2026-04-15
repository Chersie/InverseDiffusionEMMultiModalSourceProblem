from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any


try:
    import mlflow  # type: ignore

    # The installed MLflow defaults to sqlite:///mlflow.db which generates
    # mlflow-artifacts:// URIs that require a live HTTP server.
    # Use a local file store instead — works offline, no server needed.
    # Override by setting MLFLOW_TRACKING_URI=http://... before importing.
    _env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not _env_uri or _env_uri.startswith("sqlite:"):
        _project_root = Path(__file__).resolve().parent.parent.parent
        _mlruns_dir = _project_root / "mlruns"
        mlflow.set_tracking_uri(str(_mlruns_dir))

except Exception:  # pragma: no cover - fallback if mlflow is unavailable
    mlflow = None


@contextlib.contextmanager
def start_run(run_name: str, params: dict[str, Any] | None = None):
    """
    Start an MLflow run if MLflow is available.
    Falls back to a no-op context for non-breaking execution.
    """
    if mlflow is None:
        start_ts = time.time()
        print(f"[mlflow disabled] run={run_name}")
        try:
            yield
        finally:
            print(f"[mlflow disabled] elapsed_s={time.time() - start_ts:.3f}")
        return

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield


def log_pipeline_artifacts(paths: list[Path]) -> None:
    if mlflow is None:
        return
    for path in paths:
        if path.exists():
            if path.is_file():
                mlflow.log_artifact(str(path))
            elif path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)


def log_basic_metrics(metrics: dict) -> None:
    """Log basic metrics to MLflow; filters non-numeric values as tags."""
    if mlflow is None:
        return
    
    # Filter out non-numeric values and log them as tags instead
    numeric_metrics = {}
    non_numeric_tags = {}
    
    for k, v in metrics.items():
        try:
            numeric_metrics[k] = float(v)
        except (ValueError, TypeError):
            # Log non-numeric values as tags instead of metrics
            non_numeric_tags[k] = str(v)
    
    # Log numeric metrics
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)
    
    # Log non-numeric values as tags
    for tag_key, tag_value in non_numeric_tags.items():
        mlflow.set_tag(tag_key, tag_value)


def log_images(image_paths: list[Path], artifact_subdir: str = "") -> None:
    """Log a list of image files as artifacts under artifact_subdir."""
    if mlflow is None:
        return
    for path in image_paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir or None)


def set_tag(key: str, value: str) -> None:
    """Set a single MLflow tag; no-op if MLflow is unavailable."""
    if mlflow is None:
        return
    mlflow.set_tag(key, value)
