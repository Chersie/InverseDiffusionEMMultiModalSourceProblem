"""Per-mode (``(l, m, family)``) coefficient metrics."""

from __future__ import annotations

import numpy as np

from mpinv.core.packing import iter_modes


def per_lm_mse(
    pred: np.ndarray, target: np.ndarray, l_max: int
) -> dict[tuple[int, int, str], float]:
    """Per-(l, m, family) MSE in coefficient space.

    Returns a dict ``{(l, m, family): mse}`` where ``family`` is one of
    ``'aE_real', 'aE_imag', 'aM_real', 'aM_imag'``.
    """
    K = l_max * (l_max + 2)
    blocks = ("aE_real", "aE_imag", "aM_real", "aM_imag")
    out: dict[tuple[int, int, str], float] = {}
    for block_idx, block_name in enumerate(blocks):
        diff = (
            pred[:, block_idx * K : (block_idx + 1) * K]
            - target[:, block_idx * K : (block_idx + 1) * K]
        )
        for k, (l, m) in enumerate(iter_modes(l_max)):
            out[(l, m, block_name)] = float((diff[:, k] ** 2).mean())
    return out


def reflected_conjugate_aware_loss(pred: np.ndarray, target: np.ndarray, l_max: int) -> float:
    """Coefficient MSE that takes the minimum over the §1.7 reflected-conjugate orbit.

    The §1.7 ambiguity says ``a' = (-1)^m * conj(a_{l, -m})`` produces the same
    power pattern as ``a``, so a model that learns the mirror image is not actually
    wrong on the field. This metric reports
    ``min(|| pred - target ||^2, || pred - target' ||^2)``.
    """
    K = l_max * (l_max + 2)
    if pred.shape[-1] != 4 * K:
        raise ValueError(f"expected packed dim 4 K = {4 * K}, got {pred.shape[-1]}")
    re_e, im_e, re_m, im_m = np.split(target, 4, axis=-1)
    a_e = re_e + 1j * im_e
    a_m = re_m + 1j * im_m
    # build the reflected-conjugate target
    mode_index = {(l, m): k for k, (l, m) in enumerate(iter_modes(l_max))}
    a_e_p = np.zeros_like(a_e)
    a_m_p = np.zeros_like(a_m)
    for k, (l, m) in enumerate(iter_modes(l_max)):
        k_neg = mode_index[(l, -m)]
        sign = (-1.0) ** m
        a_e_p[..., k] = sign * np.conj(a_e[..., k_neg])
        a_m_p[..., k] = sign * np.conj(a_m[..., k_neg])
    target_p = np.concatenate((a_e_p.real, a_e_p.imag, a_m_p.real, a_m_p.imag), axis=-1).astype(
        target.dtype
    )

    err = ((pred - target) ** 2).mean()
    err_p = ((pred - target_p) ** 2).mean()
    return float(min(err, err_p))
