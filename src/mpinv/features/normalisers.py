"""Stateful normalisers for input and target tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Normaliser(Protocol):
    """A fit/transform/inverse_transform contract for tensor scaling."""

    def fit(self, X: np.ndarray) -> Normaliser: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class StandardScaler:
    """Per-feature mean/std scaling (broadcast over the leading batch axis)."""

    eps: float = 1e-8
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted")
        return ((X - self.mean_) / (self.std_ + self.eps)).astype(np.float32, copy=False)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler not fitted")
        return (X * (self.std_ + self.eps) + self.mean_).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass(slots=True)
class PassthroughScaler:
    """No-op scaler with the same interface, useful as a default."""

    def fit(self, X: np.ndarray) -> PassthroughScaler:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32, copy=False)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


def build_normaliser(name: str) -> Normaliser:
    if name == "standard":
        return StandardScaler()
    if name == "passthrough":
        return PassthroughScaler()
    raise ValueError(f"unknown normaliser: {name!r}")
