"""Default feature pipeline: select channels, flatten, optional log, PCA, normalise.

This is the pipeline used in the vertical slice (Phase B) and is the strong baseline
for every other comparison. It implements the contract of
:class:`mpinv.core.types.FeatureExtractor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpinv.core.types import FeatureExtractor
from mpinv.features.modes import InputMode, select_channels
from mpinv.features.normalisers import Normaliser, PassthroughScaler, StandardScaler
from mpinv.features.pca import RandomizedPCA
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class PowerPCAPipelineConfig:
    """Config for the default ``power -> flatten -> PCA -> normalise`` pipeline."""

    input_mode: InputMode = InputMode.POWER
    log_input: bool = False
    log_eps: float = 1e-12
    pca_components: int = 128
    pca_whiten: bool = False
    pca_random_state: int = 0
    normalise_features: bool = True
    normalise_targets: bool = True


@register_feature("power_pca")
@dataclass(slots=True)
class PowerPCAPipeline(FeatureExtractor):
    """Power-then-PCA feature pipeline.

    The ``transform`` step accepts either:

    - ``E`` (complex field) of shape ``(B, 2, n_theta, n_phi)``, or
    - ``P`` (power pattern) of shape ``(B, n_theta, n_phi)``,

    chooses channels per ``input_mode``, optionally takes ``log(. + eps)``, flattens to
    ``(B, F)``, and projects onto the fitted PCA basis. ``fit`` is called once on the
    training data; subsequent ``transform`` calls are stateless.
    """

    cfg: PowerPCAPipelineConfig = field(default_factory=PowerPCAPipelineConfig)
    _pca: RandomizedPCA | None = field(default=None, init=False)
    _scaler: Normaliser = field(default_factory=PassthroughScaler, init=False)
    _flat_input_dim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pca = RandomizedPCA(
            n_components=self.cfg.pca_components,
            whiten=self.cfg.pca_whiten,
            random_state=self.cfg.pca_random_state,
        )
        if self.cfg.normalise_features:
            self._scaler = StandardScaler()

    @property
    def feature_dim(self) -> int:
        return self.cfg.pca_components

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
    ) -> PowerPCAPipeline:
        ch = self._channels(E_train, P_train)
        flat = self._to_flat(ch)
        self._flat_input_dim = flat.shape[1]
        self._pca.fit(flat)
        z = self._pca.transform(flat)
        self._scaler.fit(z)
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PowerPCAPipeline not fitted")
        ch = self._channels(E, P)
        flat = self._to_flat(ch)
        z = self._pca.transform(flat)
        return self._scaler.transform(z)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PowerPCAPipeline not fitted")
        return self._pca.explained_variance_ratio_
