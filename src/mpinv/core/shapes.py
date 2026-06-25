"""Tensor-shape contracts used as inline assertions across the framework.

The new framework eliminates the legacy practice of silently reshaping tensors inside
losses and feature pipelines. When a contract is violated, we raise a ``ValueError``
with a precise message rather than calling ``F.interpolate`` or transposing in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mpinv.core.grid import GRID_DEFAULT, GridSpec


def _shape(t: Any) -> tuple[int, ...]:
    return tuple(t.shape)


def assert_packed_coeffs(t: Any, name: str = "tensor", expected_dim: int | None = None) -> None:
    """Assert ``t`` has shape ``(..., 4 K)`` for some ``K >= 1`` (real-valued).

    If ``expected_dim`` is given it must equal the trailing dimension exactly. When
    omitted, the trailing dim only has to be a positive multiple of 4 — useful for
    layers that work at any truncation order ``L``.
    """
    s = _shape(t)
    last = s[-1]
    if expected_dim is None:
        if last < 4 or last % 4 != 0:
            raise ValueError(f"{name} expected packed-coefficient layout (..., 4 K); got shape {s}")
    elif last != expected_dim:
        raise ValueError(
            f"{name} expected packed-coefficient layout (..., {expected_dim}); got shape {s}"
        )
    dtype = getattr(t, "dtype", None)
    if dtype is not None and getattr(dtype, "is_complex", False):
        raise ValueError(f"{name} must be real-valued for packed coeffs; got dtype {dtype}")


def assert_power_pattern(t: Any, grid: GridSpec = GRID_DEFAULT, name: str = "P") -> None:
    """Assert ``t`` has shape ``(..., n_theta, n_phi)`` matching the canonical layout."""
    s = _shape(t)
    if s[-2:] != (grid.n_theta, grid.n_phi):
        raise ValueError(
            f"{name} expected canonical power-pattern layout (..., {grid.n_theta}, {grid.n_phi}); "
            f"got shape {s}"
        )


def assert_field_complex(t: Any, grid: GridSpec = GRID_DEFAULT, name: str = "E") -> None:
    """Assert ``t`` has shape ``(..., 2, n_theta, n_phi)`` (channels = (E_theta, E_phi))."""
    s = _shape(t)
    if s[-3:] != (2, grid.n_theta, grid.n_phi):
        raise ValueError(
            f"{name} expected complex-field layout (..., 2, {grid.n_theta}, {grid.n_phi}); "
            f"got shape {s}"
        )


def assert_finite(t: Any, name: str = "tensor") -> None:
    """Assert ``t`` contains no NaN or Inf entries."""
    arr = np.asarray(t)
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(f"{name} contains non-finite entries: {n_nan} NaN, {n_inf} Inf")
