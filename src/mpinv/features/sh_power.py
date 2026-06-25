"""Spherical-harmonic spectral power feature.

For each (l, m) on the canonical mode list we compute ``|<f, Y_l^m>|^2`` over the
input channel, then bin by ``l`` to produce a per-degree power spectrum vector of
length ``L``. This is a coarse but cheap descriptor that captures the angular-
energy distribution.

Implementation: project each per-sample channel onto the cached scalar SH basis
via ``einsum`` against the area-weighted spherical inner product. The basis is
precomputed once per (grid, l_max).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpinv.core.area_weights import area_weights
from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX, iter_modes
from mpinv.data._basis_cache import _scalar_sph_harm
from mpinv.features.modes import InputMode, select_channels
from mpinv.features.registry import register_feature


@dataclass(slots=True)
class SHPowerConfig:
    """Knobs for :class:`SHPower`."""

    input_mode: InputMode = InputMode.POWER
    l_max: int = L_MAX
    log1p: bool = True


@register_feature("sh_power")
class SHPower:
    """Per-degree spherical-harmonic power spectrum feature."""

    def __init__(self, cfg: SHPowerConfig | None = None, grid: GridSpec | None = None):
        self.cfg = cfg or SHPowerConfig()
        self.grid = grid or GRID_DEFAULT
        self._basis: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return getattr(self, "_feature_dim", 0)

    def _build_basis(self) -> np.ndarray:
        theta_axis = self.grid.theta_axis()
        phi_axis = self.grid.phi_axis()
        T, P = np.meshgrid(theta_axis, phi_axis, indexing="ij")
        K = self.cfg.l_max * (self.cfg.l_max + 2)
        basis = np.zeros((K, self.grid.n_theta, self.grid.n_phi), dtype=np.complex64)
        for k, (l, m) in enumerate(iter_modes(self.cfg.l_max)):
            basis[k] = _scalar_sph_harm(l, m, T, P)
        return basis

    def fit(
        self,
        E_train: np.ndarray | None = None,
        P_train: np.ndarray | None = None,
    ) -> SHPower:
        if self._basis is None:
            self._basis = self._build_basis()
            self._aw = area_weights(self.grid)[:, None].astype(np.float32)
        ch = select_channels(E_train, P_train, self.cfg.input_mode)
        self._feature_dim = self.cfg.l_max * ch.shape[1]
        self._n_channels = ch.shape[1]
        return self

    def transform(
        self,
        E: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._basis is None:
            raise RuntimeError("SHPower not fitted")
        ch = select_channels(E, P, self.cfg.input_mode)
        # weighted inner product of each channel against each basis mode
        weighted = ch * self._aw  # broadcast over phi
        # coeffs shape: (B, C, K)
        coeffs = np.einsum("bctp,ktp->bck", weighted, np.conj(self._basis))
        # power per (l, m), summed over m at each l
        power_lm = (coeffs.real**2 + coeffs.imag**2).astype(np.float32)
        out = np.zeros((ch.shape[0], ch.shape[1], self.cfg.l_max), dtype=np.float32)
        for k, (l, _m) in enumerate(iter_modes(self.cfg.l_max)):
            out[:, :, l - 1] += power_lm[:, :, k]
        if self.cfg.log1p:
            out = np.log1p(out)
        return out.reshape(out.shape[0], -1)
