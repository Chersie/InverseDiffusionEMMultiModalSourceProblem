"""MLflow dataset entity logging via ``mlflow.data.from_numpy``.

See R3 in research/framework-rebuild/manifest.md for the verified API surface.
The dataset entity persists name + digest + schema + profile (not the bytes), so
this is cheap regardless of dataset size.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class DatasetSpec:
    """Identification metadata for a dataset to log to MLflow."""

    name: str
    context: str  # "training" | "validation" | "holdout" | "dummy"
    source: str | None = None  # path or URI describing where it came from


def _digest(features: np.ndarray, targets: np.ndarray | None) -> str:
    h = hashlib.sha256()
    h.update(b"features|")
    h.update(features.shape.__repr__().encode())
    h.update(features.dtype.str.encode())
    h.update(features.tobytes()[: 1 << 16])
    if targets is not None:
        h.update(b"|targets|")
        h.update(targets.shape.__repr__().encode())
        h.update(targets.tobytes()[: 1 << 16])
    return h.hexdigest()[:16]


def log_numpy_dataset(
    features: np.ndarray,
    targets: np.ndarray | None,
    spec: DatasetSpec,
    tags: dict[str, str] | None = None,
) -> None:
    """Log a numpy ``(features, targets)`` pair as an MLflow dataset.

    Lazy import of ``mlflow`` so unit tests can run without the server.
    """
    import mlflow

    dataset = mlflow.data.from_numpy(  # type: ignore[attr-defined]
        features,
        targets=targets,
        source=spec.source,
        name=spec.name,
        digest=_digest(features, targets),
    )
    mlflow.log_input(dataset, context=spec.context, tags=tags or {})
