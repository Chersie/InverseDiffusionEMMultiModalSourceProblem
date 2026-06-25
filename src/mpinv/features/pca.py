"""Randomized / Incremental PCA wrapper.

Two modes:

- :class:`RandomizedPCA` for the in-memory case: thin wrapper around
  ``sklearn.decomposition.PCA(svd_solver='randomized')`` that exposes the same
  ``fit / transform / inverse_transform`` contract as the rest of the framework.
- :class:`IncrementalPCAStream` for the streaming case: chunked fitting via
  ``sklearn.decomposition.IncrementalPCA`` for datasets too large to fit in RAM.
  Used by the streaming-data path in Phase F.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


@dataclass(slots=True)
class RandomizedPCA:
    """In-memory randomized PCA.

    Parameters
    ----------
    n_components : int
        Target latent dimension.
    whiten : bool
        Scale projections to unit variance per component if True.
    random_state : int
        Seed for the randomized SVD.
    """

    n_components: int = 128
    whiten: bool = False
    random_state: int = 0
    _pca: PCA | None = field(default=None, init=False)

    def fit(self, X: np.ndarray) -> RandomizedPCA:
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("RandomizedPCA not fitted")
        return self._pca.transform(X).astype(np.float32, copy=False)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("RandomizedPCA not fitted")
        return self._pca.inverse_transform(X).astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("RandomizedPCA not fitted")
        return self._pca.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("RandomizedPCA not fitted")
        return self._pca.components_


@dataclass(slots=True)
class IncrementalPCAStream:
    """Chunked PCA fit using sklearn IncrementalPCA. Use when the training matrix
    does not fit in RAM. Calls :meth:`fit_chunks` once over an iterable of chunks.
    """

    n_components: int = 128
    batch_size: int | None = None
    _ipca: IncrementalPCA | None = field(default=None, init=False)

    def fit_chunks(self, chunks: Iterable[np.ndarray]) -> IncrementalPCAStream:
        self._ipca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
        for X in chunks:
            self._ipca.partial_fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._ipca is None:
            raise RuntimeError("IncrementalPCAStream not fitted")
        return self._ipca.transform(X).astype(np.float32, copy=False)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._ipca is None:
            raise RuntimeError("IncrementalPCAStream not fitted")
        return self._ipca.inverse_transform(X).astype(np.float32, copy=False)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self._ipca is None:
            raise RuntimeError("IncrementalPCAStream not fitted")
        return self._ipca.explained_variance_ratio_
