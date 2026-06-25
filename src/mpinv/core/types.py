"""Protocols for the framework's pluggable boundaries.

These are intentionally loose: any class implementing the protocol satisfies it, no
inheritance required. The registries in ``models/registry.py``, ``losses/registry.py``,
``features/registry.py``, ``analysis/metrics/registry.py``, and the callbacks in
``callbacks/`` are all typed against these protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class FeatureExtractor(Protocol):
    """Per-batch feature extractor.

    The contract is: ``transform(P)`` accepts a batch of power patterns of shape
    ``(B, n_theta, n_phi)`` (numpy or torch) and returns a 2-D feature matrix
    ``(B, feature_dim)``. ``fit`` is called once on the training subset; subsequent
    calls reuse the fitted state. Stateless extractors implement ``fit`` as a no-op
    that returns ``self``.
    """

    feature_dim: int

    def fit(self, P_train: np.ndarray) -> FeatureExtractor: ...

    def transform(self, P: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Model(Protocol):
    """Trainable model that maps feature vectors to packed coefficients."""

    input_dim: int
    output_dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def parameters(self) -> Any: ...


@runtime_checkable
class LossFn(Protocol):
    """Differentiable loss producing a scalar tensor for back-propagation.

    ``forward(pred, target, **kwargs)`` returns a scalar tensor. Multi-component losses
    additionally expose a ``last_components`` mapping for per-term logging.
    """

    last_components: Mapping[str, float]

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...


@runtime_checkable
class Sink(Protocol):
    """Output sink for a training run (logging, MLflow, TensorBoard, ...)."""

    def on_run_start(self, params: Mapping[str, Any]) -> None: ...

    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...

    def log_metrics(self, values: Mapping[str, float], step: int | None = None) -> None: ...

    def log_figure(self, fig: Any, artifact_file: str) -> None: ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...

    def on_run_end(self, status: str = "FINISHED") -> None: ...


@runtime_checkable
class CallbackProto(Protocol):
    """Trainer callback. Hooks correspond to the practice.pdf "universal loop" surface."""

    def on_fit_start(self, ctx: Any) -> None: ...

    def on_epoch_start(self, ctx: Any) -> None: ...

    def on_batch_start(self, ctx: Any) -> None: ...

    def on_forward_end(self, ctx: Any) -> None: ...

    def on_loss_end(self, ctx: Any) -> None: ...

    def on_backward_end(self, ctx: Any) -> None: ...

    def on_step_end(self, ctx: Any) -> None: ...

    def on_batch_end(self, ctx: Any) -> None: ...

    def on_epoch_end(self, ctx: Any) -> None: ...

    def on_validation_end(self, ctx: Any) -> None: ...

    def on_fit_end(self, ctx: Any) -> None: ...


@runtime_checkable
class DatasetProto(Protocol):
    """Minimal dataset returning ``(features, targets)`` tensor pairs."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Any, Any]: ...


@runtime_checkable
class TorchModule(Protocol):
    """A torch ``nn.Module``-compatible interface."""

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def parameters(self, recurse: bool = True) -> Any: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool = True) -> Any: ...


# Helper: distinguishing nn.Module instances at runtime.
def is_torch_module(x: Any) -> bool:
    return isinstance(x, nn.Module)
