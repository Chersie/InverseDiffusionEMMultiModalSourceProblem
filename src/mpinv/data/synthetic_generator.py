"""Synthetic dataset generator for the phaseless multipole inverse problem.

We sample complex coefficients ``(a^E, a^M)``, synthesise the complex tangential field
on the project grid via the cached VSH basis (``data/_basis_cache.py``), and compute
the dual-polarisation power pattern ``P = |E_theta|^2 + |E_phi|^2``. Each sample yields
``(P, packed_coefficients)`` aligned to the framework's canonical layouts:

- ``P`` has shape ``(n_theta, n_phi)`` and is real, non-negative.
- ``packed_coefficients`` has shape ``(4 K,)`` per :func:`mpinv.core.packing.pack_coefficients`.

The forward operator used here is **identical** to the one used inside
``losses.differentiable_field`` (same basis tensor), so synthetic targets are
information-theoretically reachable by a perfect inverse model.

Sampling strategies (composable, controlled by :class:`SyntheticGeneratorConfig`):

- ``mode``: ``gaussian`` (i.i.d. complex normal), ``uniform`` (i.i.d. uniform in a box),
  ``colored`` (per-l power scaled as ``(l + 1)^(-alpha)`` to mimic realistic radiator
  spectra), or ``sparse`` (random subset of K modes active).
- ``family_balance``: scalar in [0, 1] giving the relative magnitude of the magnetic
  family vs the electric. 0.5 = equal.
- ``coef_scale_log_uniform_range``: optional ``(lo, hi)`` enabling a per-sample global
  log-uniform amplitude scale (Latin-square-style, R1 in legacy notation).
- ``mode_dropout_prob``: optional probability of zeroing each mode independently.

The generator is reproducible: seeding is per-batch via the supplied ``rng`` argument,
not via global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX, iter_modes, pack_coefficients
from mpinv.data._basis_cache import VSHBasis, load_basis


@dataclass(slots=True)
class SyntheticGeneratorConfig:
    """Configuration knobs for the synthetic data generator."""

    mode: Literal["gaussian", "uniform", "colored", "sparse"] = "gaussian"
    family_balance: float = 0.5
    coef_scale: float = 1.0
    coef_scale_log_uniform_range: tuple[float, float] | None = None
    color_alpha: float = 1.0
    sparse_active_fraction: float = 0.1
    mode_dropout_prob: float = 0.0
    family_balance_jitter: float = 0.0
    seed_offset: int = 0
    grid: GridSpec = field(default_factory=lambda: GRID_DEFAULT)
    l_max: int = L_MAX


class SyntheticGenerator:
    """Stateful generator that owns the cached VSH basis and produces ``(P, packed)`` pairs."""

    def __init__(self, cfg: SyntheticGeneratorConfig | None = None, basis: VSHBasis | None = None):
        self.cfg = cfg or SyntheticGeneratorConfig()
        self.basis = basis if basis is not None else load_basis(self.cfg.grid, self.cfg.l_max)
        if self.basis.l_max != self.cfg.l_max or self.basis.grid != self.cfg.grid:
            raise ValueError(
                f"basis grid/l_max mismatch: basis (l_max={self.basis.l_max}, "
                f"grid={self.basis.grid}) vs cfg (l_max={self.cfg.l_max}, grid={self.cfg.grid})"
            )

    @property
    def n_modes(self) -> int:
        return self.cfg.l_max * (self.cfg.l_max + 2)

    @property
    def packed_dim(self) -> int:
        return 4 * self.n_modes

    def sample_coefficients(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` pairs ``(a^E, a^M)`` of shape ``(n, K)`` complex64.

        The output is built up by composing per-mode sampling, optional spectrum
        colouring, optional sparsity, optional mode dropout, and optional log-uniform
        global scale.
        """
        K = self.n_modes
        cfg = self.cfg

        if cfg.mode == "gaussian":
            a_e = (
                rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
            ) / np.sqrt(2)
            a_m = (
                rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
            ) / np.sqrt(2)
        elif cfg.mode == "uniform":
            a_e = rng.uniform(-1, 1, size=(n, K)) + 1j * rng.uniform(-1, 1, size=(n, K))
            a_m = rng.uniform(-1, 1, size=(n, K)) + 1j * rng.uniform(-1, 1, size=(n, K))
        elif cfg.mode == "sparse":
            mask = rng.uniform(size=(n, K)) < cfg.sparse_active_fraction
            a_e = mask * (rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K)))
            a_m = mask * (rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K)))
        elif cfg.mode == "colored":
            a_e = (
                rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
            ) / np.sqrt(2)
            a_m = (
                rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
            ) / np.sqrt(2)
            l_for_each_mode = np.array([l for l, _ in iter_modes(cfg.l_max)], dtype=np.float64)
            scale = (l_for_each_mode + 1.0) ** (-cfg.color_alpha)
            a_e *= scale
            a_m *= scale
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

        if cfg.mode_dropout_prob > 0:
            keep = rng.uniform(size=(n, K)) >= cfg.mode_dropout_prob
            a_e *= keep
            a_m *= keep

        # Family balance: scale magnetic family relative to electric.
        balance = cfg.family_balance
        if cfg.family_balance_jitter > 0:
            balance = balance + rng.uniform(
                -cfg.family_balance_jitter, cfg.family_balance_jitter, size=(n, 1)
            )
            balance = np.clip(balance, 0.0, 1.0)
        a_e = (1.0 - np.asarray(balance)) * a_e * 2.0
        a_m = np.asarray(balance) * a_m * 2.0

        # Per-sample log-uniform global scale.
        if cfg.coef_scale_log_uniform_range is not None:
            lo, hi = cfg.coef_scale_log_uniform_range
            log_scale = rng.uniform(lo, hi, size=(n, 1))
            sample_scale = np.exp(log_scale)
            a_e *= sample_scale
            a_m *= sample_scale

        a_e *= cfg.coef_scale
        a_m *= cfg.coef_scale
        return a_e.astype(np.complex64), a_m.astype(np.complex64)

    def synthesize(self, a_e: np.ndarray, a_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the complex field and power pattern from coefficients.

        Parameters
        ----------
        a_e, a_m : np.ndarray
            Complex arrays of shape ``(n, K)``.

        Returns
        -------
        E : np.ndarray
            Complex array of shape ``(n, 2, n_theta, n_phi)`` (channels = (theta, phi)).
        P : np.ndarray
            Real array of shape ``(n, n_theta, n_phi)``.
        """
        # basis: (K, 2 families, 2 components, n_theta, n_phi)
        # a_e:   (n, K)
        # E_e:   (n, 2 components, n_theta, n_phi)
        E_e = np.einsum("nk,kctp->nctp", a_e, self.basis.basis[:, 0])
        E_m = np.einsum("nk,kctp->nctp", a_m, self.basis.basis[:, 1])
        E = E_e + E_m
        P = (E.real**2 + E.imag**2).sum(axis=1)
        return E, P.astype(np.float32)

    def generate_batch(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` examples and return ``(P, packed_coefficients)``."""
        a_e, a_m = self.sample_coefficients(n, rng)
        _, P = self.synthesize(a_e, a_m)
        packed = pack_coefficients(a_e, a_m)
        return P, packed

    def generate_batch_with_field(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same as :meth:`generate_batch` but also returns the complex field."""
        a_e, a_m = self.sample_coefficients(n, rng)
        E, P = self.synthesize(a_e, a_m)
        packed = pack_coefficients(a_e, a_m)
        return E, P, packed
