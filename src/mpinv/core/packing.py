"""Packed-coefficient layout and conversions to/from the torch-harmonics SHT grid.

The packed real coefficient vector defined in presentation/ch1_full.md §1.1 is

    a = [Re(a^E), Im(a^E), Re(a^M), Im(a^M)]   in R^{4K}

where K = sum_{l=1..L} (2 l + 1) = L (L + 2) is the number of (l, m) pairs per family
and the inner ordering iterates l ascending and m from -l to +l.

For the project default L = 15 we have K = 255 and PACKED_DIM = 4 K = 1020.

This module also provides the bijection between the packed real vector and the dense
complex coefficient grid expected by ``torch_harmonics.InverseRealVectorSHT``,
``coeffs[..., channel, l, m]`` of shape ``(..., 2, lmax, mmax)`` with lmax = L + 1,
mmax = L + 1 (both non-inclusive). Channel 0 stores the spheroidal/electric family,
channel 1 stores the toroidal/magnetic family. Only ``m >= 0`` is stored explicitly;
the negative-m mirror is handled internally by the library through Hermitian symmetry.
See R1 in ``research/framework-rebuild/manifest.md`` for the verified API contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

L_MAX: int = 15
"""Truncation order; matches the thesis convention (presentation/ch1_full.md §1.1)."""

K_MODES: int = L_MAX * (L_MAX + 2)
"""Number of (l, m) modes per family at L = L_MAX. K = L (L + 2) = 255."""

PACKED_DIM: int = 4 * K_MODES
"""Length of the packed real coefficient vector. 4 K = 1020 for L = 15."""


Family = Literal["E", "M"]


def iter_modes(l_max: int = L_MAX) -> Iterator[tuple[int, int]]:
    """Yield ``(l, m)`` pairs in canonical order: l ascending, m from -l to +l."""
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            yield l, m


def flat_index(l: int, m: int, l_max: int = L_MAX) -> int:
    """Index of ``(l, m)`` in the flat per-family list of length ``l_max (l_max + 2)``.

    Closed form: the number of modes at orders ``< l`` is ``(l - 1)(l + 1)``, plus the
    offset ``m + l`` inside order ``l``.
    """
    if not (1 <= l <= l_max):
        raise ValueError(f"l={l} out of range [1, {l_max}]")
    if not (-l <= m <= l):
        raise ValueError(f"m={m} out of range [-{l}, {l}]")
    return (l - 1) * (l + 1) + (m + l)


# Pre-compute the canonical (l, m) order so callers don't recompute it.
_CANONICAL_LM: list[tuple[int, int]] = list(iter_modes(L_MAX))
assert len(_CANONICAL_LM) == K_MODES, (
    f"K_MODES inconsistent with iter_modes: {K_MODES} vs {len(_CANONICAL_LM)}"
)


def pack_coefficients(a_e: np.ndarray, a_m: np.ndarray) -> np.ndarray:
    """Pack two complex coefficient vectors into the real packed layout.

    Parameters
    ----------
    a_e, a_m : np.ndarray
        Complex arrays of shape ``(..., K)``. Coefficient ordering: l ascending,
        m from -l to +l.

    Returns
    -------
    np.ndarray
        Real array of shape ``(..., 4 K)`` in the order
        ``[Re(a^E), Im(a^E), Re(a^M), Im(a^M)]``.
    """
    if a_e.shape != a_m.shape:
        raise ValueError(f"a_e shape {a_e.shape} != a_m shape {a_m.shape}")
    if a_e.shape[-1] < 1:
        raise ValueError(f"trailing dim must be K >= 1, got {a_e.shape[-1]}")
    parts = (a_e.real, a_e.imag, a_m.real, a_m.imag)
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def unpack_coefficients(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_coefficients`.

    Parameters
    ----------
    packed : np.ndarray
        Real array of shape ``(..., 4 K)`` for any ``K >= 1``.

    Returns
    -------
    a_e, a_m : np.ndarray
        Complex arrays of shape ``(..., K)``.
    """
    if packed.shape[-1] % 4 != 0:
        raise ValueError(f"trailing dim must be divisible by 4, got {packed.shape[-1]}")
    re_e, im_e, re_m, im_m = np.split(packed, 4, axis=-1)
    a_e = (re_e + 1j * im_e).astype(np.complex64, copy=False)
    a_m = (re_m + 1j * im_m).astype(np.complex64, copy=False)
    return a_e, a_m


def pack_to_sht_grid(packed: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """Map packed real coefficients to the dense SHT grid expected by torch-harmonics.

    Output shape: ``(..., 2, lmax_idx, mmax_idx)`` complex with
    ``lmax_idx = mmax_idx = l_max + 1``. Only m >= 0 is stored explicitly; the library
    reconstructs m < 0 from Hermitian symmetry (see R1 in the research manifest).

    Parameters
    ----------
    packed : np.ndarray
        Real array of shape ``(..., 4 K)``.
    l_max : int
        Truncation order (default ``L_MAX``).

    Returns
    -------
    np.ndarray
        Complex array of shape ``(..., 2, l_max + 1, l_max + 1)`` (channels = E, M).
    """
    if packed.shape[-1] != 4 * l_max * (l_max + 2):
        raise ValueError(
            f"packed trailing dim must be {4 * l_max * (l_max + 2)} for L={l_max}, "
            f"got {packed.shape[-1]}"
        )
    a_e, a_m = unpack_coefficients(packed)
    batch_shape = a_e.shape[:-1]
    out = np.zeros((*batch_shape, 2, l_max + 1, l_max + 1), dtype=np.complex64)
    for k, (l, m) in enumerate(_CANONICAL_LM):
        if m < 0:
            continue
        out[..., 0, l, m] = a_e[..., k]
        out[..., 1, l, m] = a_m[..., k]
    return out


def zero_above_band(packed, k: int, l_max: int):
    """Return a copy of ``packed`` with every coefficient at order ``l > k`` zeroed.

    The packed layout is the standard four-quarter ``[Re a^E, Im a^E, Re a^M, Im a^M]``
    of length ``4 K`` with ``K = l_max (l_max + 2)``. Inside each quarter the modes
    are ordered ``l = 1..l_max`` ascending, ``m = -l..+l`` ascending; the count of
    modes with ``l ≤ k`` is exactly ``k (k + 2)``. We therefore zero the index range
    ``[k (k + 2), K)`` inside each of the four quarters and leave the
    ``l ∈ {1, …, k}`` block untouched.

    Works for both ``np.ndarray`` and ``torch.Tensor`` because the slice-assignment
    and copy contract is identical (``.clone()`` for torch, ``.copy()`` for numpy).

    Parameters
    ----------
    packed : np.ndarray | torch.Tensor
        Trailing dim ``4 K``; arbitrary leading batch dims.
    k : int
        Inclusive band cutoff; modes with ``l ≤ k`` are kept, ``l > k`` are zeroed.
        ``k = l_max`` is a no-op (returns a copy of the input).
    l_max : int
        Truncation order of the input layout.

    Raises
    ------
    ValueError
        If ``k`` is out of range or the trailing dim disagrees with ``4 K``.
    """
    K = l_max * (l_max + 2)
    if packed.shape[-1] != 4 * K:
        raise ValueError(
            f"packed trailing dim must be {4 * K} for l_max={l_max}; got {packed.shape[-1]}"
        )
    if not (0 <= k <= l_max):
        raise ValueError(f"k={k} out of range [0, {l_max}]")
    out = packed.clone() if hasattr(packed, "clone") else packed.copy()
    if k == l_max:
        return out
    boundary = k * (k + 2)
    for q in range(4):
        start = q * K + boundary
        end = (q + 1) * K
        out[..., start:end] = 0
    return out


def unpack_from_sht_grid(grid: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    """Inverse of :func:`pack_to_sht_grid`: pull the m >= 0 entries back into the
    packed real layout.

    The m < 0 slots in the packed vector are reconstructed from Hermitian symmetry:
    ``a_{l,-m} = (-1)^m conj(a_{l,m})`` for the standard (Condon-Shortley) convention.
    This relation is needed because the packed layout stores both signs of m
    explicitly, while the SHT grid stores only ``m >= 0``.

    Parameters
    ----------
    grid : np.ndarray
        Complex array of shape ``(..., 2, l_max + 1, l_max + 1)``.
    l_max : int
        Truncation order.

    Returns
    -------
    np.ndarray
        Real array of shape ``(..., 4 K)``.
    """
    if grid.shape[-3:] != (2, l_max + 1, l_max + 1):
        raise ValueError(
            f"grid trailing dims must be (2, {l_max + 1}, {l_max + 1}), got {grid.shape[-3:]}"
        )
    K = l_max * (l_max + 2)
    a_e = np.zeros((*grid.shape[:-3], K), dtype=np.complex64)
    a_m = np.zeros_like(a_e)
    for k, (l, m) in enumerate(_CANONICAL_LM):
        if m >= 0:
            a_e[..., k] = grid[..., 0, l, m]
            a_m[..., k] = grid[..., 1, l, m]
        else:
            sign = (-1.0) ** m
            a_e[..., k] = sign * np.conj(grid[..., 0, l, -m])
            a_m[..., k] = sign * np.conj(grid[..., 1, l, -m])
    return pack_coefficients(a_e, a_m)
