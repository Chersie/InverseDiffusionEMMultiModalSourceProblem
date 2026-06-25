"""Histogram of oriented gradients feature.

Wraps ``skimage.feature.hog`` per-sample, per-channel. Gracefully falls back to a
no-op extractor if scikit-image is not installed (it lives in the ``cv`` extra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpinv.features.modes import InputMode, select_channels
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class HOGConfig:
    """Knobs for :class:`HOGExtractor`."""

    input_mode: InputMode = InputMode.POWER
    pixels_per_cell: tuple[int, int] = (16, 16)
    cells_per_block: tuple[int, int] = (2, 2)
    orientations: int = 9
    block_norm: str = "L2-Hys"


@register_feature("hog")
class HOGExtractor:
    """Per-sample HOG descriptor.

    Shape contract: input ``(B, C, n_theta, n_phi)`` -> output ``(B, F)`` where ``F``
    depends on the grid and config; resolved at ``fit`` time on a probe sample.
    """

    def __init__(self, cfg: HOGConfig | None = None):
        self.cfg = cfg or HOGConfig()

    @property
    def feature_dim(self) -> int:
        return getattr(self, "_feature_dim", 0)

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> HOGExtractor:
        try:
            from skimage.feature import hog  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised in environments without cv extra
            raise RuntimeError("HOGExtractor requires the 'cv' extra (scikit-image)") from exc
        ch = select_channels(E_train, P_train, self.cfg.input_mode)
        sample = self._compute_one(ch[0])
        self._feature_dim = sample.size
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "_feature_dim"):
            raise RuntimeError("HOGExtractor not fitted")
        ch = select_channels(E, P, self.cfg.input_mode)
        out = np.empty((ch.shape[0], self._feature_dim), dtype=np.float32)
        for i in range(ch.shape[0]):
            out[i] = self._compute_one(ch[i])
        return out

    def _compute_one(self, sample: np.ndarray) -> np.ndarray:
        from skimage.feature import hog

        descriptors = []
        for c in range(sample.shape[0]):
            descriptors.append(
                hog(
                    sample[c],
                    pixels_per_cell=self.cfg.pixels_per_cell,
                    cells_per_block=self.cfg.cells_per_block,
                    orientations=self.cfg.orientations,
                    block_norm=self.cfg.block_norm,
                    feature_vector=True,
                )
            )
        return np.concatenate(descriptors).astype(np.float32, copy=False)
