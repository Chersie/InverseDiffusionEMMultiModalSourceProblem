"""Raw-flatten feature pipeline: select channels, optional log, flatten, normalise.

This is the "no-PCA" sibling of :class:`mpinv.features.power_pipeline.PowerPCAPipeline`.
Setting ``pca_components = n_theta * n_phi`` on the PCA pipeline does **not**
bypass PCA — sklearn's :class:`sklearn.decomposition.PCA` is still fitted and
applied (`src/mpinv/features/pca.py:fit`). This extractor implements the
honest "feed every pixel to the model" baseline that the PCA pipeline cannot
emulate without paying for the SVD.

Output shape: ``(B, C * n_theta * n_phi)`` with ``C ∈ {1, 2, 4}`` per
:class:`mpinv.features.modes.InputMode`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpinv.core.types import FeatureExtractor
from mpinv.features.modes import InputMode, select_channels
from mpinv.features.normalisers import Normaliser, PassthroughScaler, StandardScaler
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class RawFlattenPipelineConfig:
    """Knobs for :class:`RawFlattenPipeline`."""

    input_mode: InputMode = InputMode.POWER
    log_input: bool = False
    log_eps: float = 1e-12
    normalise_features: bool = True


@register_feature("raw_flat")
@dataclass(slots=True)
class RawFlattenPipeline(FeatureExtractor):
    """Channel-select + flatten + normalise. No PCA."""

    cfg: RawFlattenPipelineConfig = field(default_factory=RawFlattenPipelineConfig)
    _scaler: Normaliser = field(default_factory=PassthroughScaler, init=False)
    _flat_input_dim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.cfg.normalise_features:
            self._scaler = StandardScaler()

    @property
    def feature_dim(self) -> int:
        return self._flat_input_dim

    def _channels(self, E: np.ndarray | None, P: np.ndarray | None) -> np.ndarray:
        return select_channels(E, P, self.cfg.input_mode)

    def _to_flat(self, ch: np.ndarray) -> np.ndarray:
        if self.cfg.log_input:
            ch = np.log(np.maximum(ch, self.cfg.log_eps))
        return ch.reshape(ch.shape[0], -1).astype(np.float32, copy=False)

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> RawFlattenPipeline:
        ch = self._channels(E_train, P_train)
        flat = self._to_flat(ch)
        self._flat_input_dim = flat.shape[1]
        self._scaler.fit(flat)
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._flat_input_dim == 0:
            raise RuntimeError("RawFlattenPipeline not fitted")
        ch = self._channels(E, P)
        flat = self._to_flat(ch)
        return self._scaler.transform(flat)
