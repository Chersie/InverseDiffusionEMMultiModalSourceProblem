"""Composite feature pipeline: PCA-on-channels + appended CV features.

Lets the user combine the strong PCA baseline with one or more CV extractors
(FFT radial, HOG, SH power). Each extractor is fit on the training data and
concatenated to the PCA features along the feature axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mpinv.features.normalisers import Normaliser, PassthroughScaler, StandardScaler
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class CompositeFeaturesConfig:
    """Knobs for :class:`CompositePipeline`."""

    pca: PowerPCAPipelineConfig = field(default_factory=PowerPCAPipelineConfig)
    skip_pca: bool = False
    normalise_concat: bool = True


@register_feature("composite")
class CompositePipeline:
    """PCA + a list of CV extractors concatenated along the feature axis.

    Construct as ``CompositePipeline(cfg, extractors=[...])`` where each extractor
    obeys the ``FeatureExtractor`` protocol (``fit``, ``transform``,
    ``feature_dim``). The composite ``fit(P_train, E_train)`` calls each extractor's
    own ``fit``; ``transform`` concatenates their outputs.
    """

    def __init__(
        self,
        cfg: CompositeFeaturesConfig | None = None,
        extractors: Sequence[object] = (),
    ):
        self.cfg = cfg or CompositeFeaturesConfig()
        self.pca = PowerPCAPipeline(cfg=self.cfg.pca) if not self.cfg.skip_pca else None
        self.extractors = list(extractors)
        self._scaler: Normaliser = (
            StandardScaler() if self.cfg.normalise_concat else PassthroughScaler()
        )
        self._feature_dim: int = 0

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> CompositePipeline:
        parts: list[np.ndarray] = []
        if self.pca is not None:
            self.pca.fit(E_train=E_train, P_train=P_train)
            parts.append(self.pca.transform(E=E_train, P=P_train))
        for ext in self.extractors:
            ext.fit(E_train=E_train, P_train=P_train)
            parts.append(ext.transform(E=E_train, P=P_train))
        Z = np.concatenate(parts, axis=-1)
        self._scaler.fit(Z)
        self._feature_dim = Z.shape[-1]
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.pca is not None:
            parts.append(self.pca.transform(E=E, P=P))
        for ext in self.extractors:
            parts.append(ext.transform(E=E, P=P))
        Z = np.concatenate(parts, axis=-1)
        return self._scaler.transform(Z)
