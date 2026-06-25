"""Radial 2D-FFT power-spectrum feature.

For each input ``(C, n_theta, n_phi)`` channel we compute the 2D FFT magnitude,
shift the zero frequency to the centre, then radially bin into ``n_bins``
equal-radius rings. Rings are ordered low-frequency-first.

Optionally we apply a ``sin theta`` window before the FFT to reduce pole-area
distortion (the same area weight used in the physics loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpinv.features.modes import InputMode, select_channels
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class FFTRadialConfig:
    """Knobs for :class:`FFTRadial`."""

    input_mode: InputMode = InputMode.POWER
    n_bins: int = 32
    sin_theta_window: bool = True
    log1p: bool = True


@register_feature("fft_radial")
class FFTRadial:
    """Radial 2D-FFT spectrum per input channel."""

    def __init__(self, cfg: FFTRadialConfig | None = None):
        self.cfg = cfg or FFTRadialConfig()

    @property
    def feature_dim(self) -> int:
        # filled at fit time to also account for the channel multiplier
        return getattr(self, "_feature_dim", 0)

    def _radial_grid(self, n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
        ky = np.fft.fftshift(np.fft.fftfreq(n_theta))
        kx = np.fft.fftshift(np.fft.fftfreq(n_phi))
        KY, KX = np.meshgrid(ky, kx, indexing="ij")
        R = np.sqrt(KX**2 + KY**2)
        edges = np.linspace(0.0, R.max(), self.cfg.n_bins + 1)
        binned = np.digitize(R, edges) - 1
        binned = np.clip(binned, 0, self.cfg.n_bins - 1)
        return R, binned

    def _channels(self, E: np.ndarray | None, P: np.ndarray | None) -> np.ndarray:
        return select_channels(E, P, self.cfg.input_mode)

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> FFTRadial:
        ch = self._channels(E_train, P_train)
        n_theta, n_phi = ch.shape[-2], ch.shape[-1]
        self._feature_dim = self.cfg.n_bins * ch.shape[1]
        self._n_theta = n_theta
        self._n_phi = n_phi
        _, self._binned = self._radial_grid(n_theta, n_phi)
        if self.cfg.sin_theta_window:
            theta_axis = np.linspace(np.pi / (2 * n_theta), np.pi - np.pi / (2 * n_theta), n_theta)
            self._win = np.sin(theta_axis)[None, None, :, None].astype(np.float32)
        else:
            self._win = None
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "_binned"):
            raise RuntimeError("FFTRadial not fitted")
        ch = self._channels(E, P)
        if self._win is not None:
            ch = ch * self._win
        F = np.fft.fftshift(np.fft.fft2(ch, axes=(-2, -1)), axes=(-2, -1))
        mag = np.abs(F).astype(np.float32)
        B = mag.shape[0]
        C = mag.shape[1]
        n_bins = self.cfg.n_bins
        out = np.zeros((B, C, n_bins), dtype=np.float32)
        for b in range(n_bins):
            mask = self._binned == b
            if mask.sum() == 0:
                continue
            out[..., b] = mag[..., mask].mean(axis=-1)
        if self.cfg.log1p:
            out = np.log1p(out)
        return out.reshape(B, -1)
