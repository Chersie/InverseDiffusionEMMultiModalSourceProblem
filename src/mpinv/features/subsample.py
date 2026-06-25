"""Sub-sampled-grid feature pipeline.

Two flavours:

- **Deterministic stride** (default): take every ``theta_stride``-th polar sample
  and every ``phi_stride``-th azimuthal sample.
- **Random k%-mask** (when ``random_fraction`` is set): pick a fixed fraction of
  the ``n_theta * n_phi`` pixels uniformly at random with seed
  ``mask_seed`` and reuse the same mask at every transform call.

In both flavours we select channels per :class:`mpinv.features.modes.InputMode`,
apply optional ``log1p``, gather the kept pixels, flatten, and normalise.

This is the closest analogue to "feed the model a coarsely-sampled version of
``P``" — useful both as a feature ablation and as a stand-in for measurement
geometries with fewer angular samples than the canonical 360x179.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpinv.core.types import FeatureExtractor
from mpinv.features.modes import InputMode, select_channels
from mpinv.features.normalisers import Normaliser, PassthroughScaler, StandardScaler
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class SubsampleGridPipelineConfig:
    """Knobs for :class:`SubsampleGridPipeline`."""

    input_mode: InputMode = InputMode.POWER
    log_input: bool = False
    log_eps: float = 1e-12
    theta_stride: int = 4
    phi_stride: int = 4
    random_fraction: float | None = None
    mask_seed: int = 0
    normalise_features: bool = True


@register_feature("subsample_grid")
@dataclass(slots=True)
class SubsampleGridPipeline(FeatureExtractor):
    """Stride-based or random-mask sub-grid + flatten + normalise."""

    cfg: SubsampleGridPipelineConfig = field(default_factory=SubsampleGridPipelineConfig)
    _scaler: Normaliser = field(default_factory=PassthroughScaler, init=False)
    _flat_input_dim: int = field(default=0, init=False)
    _mask_flat: np.ndarray | None = field(default=None, init=False)
    _theta_idx: np.ndarray | None = field(default=None, init=False)
    _phi_idx: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.cfg.normalise_features:
            self._scaler = StandardScaler()
        if self.cfg.theta_stride < 1 or self.cfg.phi_stride < 1:
            raise ValueError("strides must be >= 1")
        if self.cfg.random_fraction is not None:
            if not (0 < self.cfg.random_fraction <= 1):
                raise ValueError("random_fraction must be in (0, 1]")

    @property
    def feature_dim(self) -> int:
        return self._flat_input_dim

    def _channels(self, E: np.ndarray | None, P: np.ndarray | None) -> np.ndarray:
        return select_channels(E, P, self.cfg.input_mode)

    def _build_indices(self, n_theta: int, n_phi: int) -> None:
        if self.cfg.random_fraction is not None:
            rng = np.random.default_rng(self.cfg.mask_seed)
            n_pix = n_theta * n_phi
            n_keep = max(1, int(round(self.cfg.random_fraction * n_pix)))
            mask_flat = np.zeros(n_pix, dtype=bool)
            mask_flat[rng.choice(n_pix, size=n_keep, replace=False)] = True
            self._mask_flat = mask_flat
        else:
            self._theta_idx = np.arange(0, n_theta, self.cfg.theta_stride, dtype=np.int64)
            self._phi_idx = np.arange(0, n_phi, self.cfg.phi_stride, dtype=np.int64)

    def _gather(self, ch: np.ndarray) -> np.ndarray:
        # ch: (B, C, n_theta, n_phi)
        if self._mask_flat is not None:
            B, C, n_theta, n_phi = ch.shape
            flat = ch.reshape(B, C, n_theta * n_phi)
            kept = flat[..., self._mask_flat]
            return kept.reshape(B, -1)
        if self._theta_idx is None or self._phi_idx is None:
            raise RuntimeError("indices not built")
        sub = ch[:, :, self._theta_idx[:, None], self._phi_idx[None, :]]
        return sub.reshape(sub.shape[0], -1)

    def _to_features(self, ch: np.ndarray) -> np.ndarray:
        if self.cfg.log_input:
            ch = np.log(np.maximum(ch, self.cfg.log_eps))
        return self._gather(ch).astype(np.float32, copy=False)

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> SubsampleGridPipeline:
        ch = self._channels(E_train, P_train)
        self._build_indices(n_theta=ch.shape[-2], n_phi=ch.shape[-1])
        flat = self._to_features(ch)
        self._flat_input_dim = flat.shape[1]
        self._scaler.fit(flat)
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._flat_input_dim == 0:
            raise RuntimeError("SubsampleGridPipeline not fitted")
        ch = self._channels(E, P)
        flat = self._to_features(ch)
        return self._scaler.transform(flat)
