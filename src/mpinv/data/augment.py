"""Augmentation library for the synthetic + holdout data pipeline.

Five augmentations, each documented with its physical-consistency contract:

1. ``coef_phase_rotation``: per-sample global phase ``a -> exp(i alpha) a`` with
   ``alpha ~ U[0, 2 pi)``. The power pattern ``P = |E|^2`` is *invariant* under
   this map (presentation/ch1_full.md §1.6, global U(1)). Only ``packed`` updates;
   ``P`` is left unchanged. Free augmentation.
2. ``coef_additive_noise``: ``a -> a + sigma_a * eps`` with ``eps`` complex
   standard normal. Changes ``P``. Re-synthesises ``P`` from the perturbed
   coefficients to keep ``(P, packed)`` consistent.
3. ``coef_mode_dropout``: zero each ``(l, m)`` mode (independently per family
   and per sample) with probability ``dropout_prob``. Changes ``P``;
   re-synthesises ``P`` from the masked coefficients to keep ``(P, packed)``
   consistent. Mirrors the synthetic generator's ``mode_dropout_prob`` knob but
   applies it as a post-hoc augmentation, so it works on real-antenna samples
   too.
4. ``field_additive_noise``: ``P -> max(P + sigma_P * eps, 0)`` with ``eps``
   real standard normal scaled by the per-sample max of ``P``. Targets are kept
   unchanged: this is an explicit *noisy-input vs. clean-target* dataset
   contract that mirrors the robustness setup of Schmid et al. 2025.
5. ``field_phi_roll``: random integer roll ``k`` along the azimuth, applied
   *both* to ``P`` (``np.roll`` along axis -1) and to ``packed`` via the
   coefficient-domain equivalent ``a_{l,m} -> exp(i m phi_k) a_{l,m}``. The two
   are guaranteed consistent up to discretisation error because the project's
   1-degree azimuth grid samples ``phi_k = 2 pi k / n_phi`` exactly.

Each augmentation is parameterised by a small, immutable dataclass and exposed
through :func:`apply_augmentation`. The protocol is deliberately simple: take
``(P, packed)`` numpy arrays, return new ``(P, packed)`` arrays of the same
shape and dtype.

The augmentations need the VSH basis tensor only when re-synthesis is required
(noise or dropout on coefficients). Other augmentations are basis-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mpinv.core.packing import iter_modes, pack_coefficients, unpack_coefficients
from mpinv.data._basis_cache import VSHBasis


@dataclass(slots=True, frozen=True)
class CoefPhaseRotationConfig:
    """Per-sample uniform global phase rotation in coefficient space."""

    name: Literal["coef_phase_rotation"] = "coef_phase_rotation"


@dataclass(slots=True, frozen=True)
class CoefAdditiveNoiseConfig:
    """Per-sample additive complex Gaussian noise on coefficients.

    ``sigma`` is the per-coefficient noise standard deviation (applied to both
    real and imaginary parts independently after dividing by ``sqrt(2)``).
    """

    sigma: float = 0.05
    name: Literal["coef_additive_noise"] = "coef_additive_noise"


@dataclass(slots=True, frozen=True)
class CoefModeDropoutConfig:
    """Per-mode independent dropout on coefficients.

    Each ``(l, m)`` mode in each family is zeroed with probability
    ``dropout_prob``, independently per sample. ``P`` is re-synthesised from
    the masked coefficients so ``(P, packed)`` stays consistent. Identical
    statistical recipe to ``SyntheticGeneratorConfig.mode_dropout_prob`` but
    applied post-hoc, so it works on real-antenna samples.
    """

    dropout_prob: float = 0.1
    name: Literal["coef_mode_dropout"] = "coef_mode_dropout"


@dataclass(slots=True, frozen=True)
class FieldAdditiveNoiseConfig:
    """Per-sample additive Gaussian noise on the power pattern.

    ``relative_sigma`` is multiplied by each sample's max ``P`` to obtain the
    actual standard deviation. Result is clipped at zero (``P`` is non-negative).
    """

    relative_sigma: float = 0.02
    name: Literal["field_additive_noise"] = "field_additive_noise"


@dataclass(slots=True, frozen=True)
class FieldPhiRollConfig:
    """Per-sample random azimuthal roll, applied consistently to P and packed."""

    name: Literal["field_phi_roll"] = "field_phi_roll"


AugmentationConfig = (
    CoefPhaseRotationConfig
    | CoefAdditiveNoiseConfig
    | CoefModeDropoutConfig
    | FieldAdditiveNoiseConfig
    | FieldPhiRollConfig
)


def _coef_phase_rotation(
    P: np.ndarray, packed: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``a -> exp(i alpha) a`` per sample. P invariant; packed updates."""
    n = packed.shape[0]
    a_e, a_m = unpack_coefficients(packed)
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1)).astype(np.float64)
    rot = np.exp(1j * alpha).astype(np.complex64)
    a_e = a_e * rot
    a_m = a_m * rot
    return P, pack_coefficients(a_e, a_m)


def _synthesize(
    a_e: np.ndarray, a_m: np.ndarray, basis: VSHBasis
) -> np.ndarray:
    """Re-synthesise P from coefficients via the cached VSH basis."""
    E_e = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E = E_e + E_m
    return (E.real**2 + E.imag**2).sum(axis=1).astype(np.float32)


def _coef_additive_noise(
    P: np.ndarray,
    packed: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    basis: VSHBasis | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``a -> a + sigma * eps`` with re-synthesis of P (consistent (P, packed))."""
    if basis is None:
        raise ValueError("coef_additive_noise requires a VSH basis to re-synthesise P")
    a_e, a_m = unpack_coefficients(packed)
    n, K = a_e.shape
    eps_e = (
        rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
    ).astype(np.complex64) / np.sqrt(2.0)
    eps_m = (
        rng.standard_normal(size=(n, K)) + 1j * rng.standard_normal(size=(n, K))
    ).astype(np.complex64) / np.sqrt(2.0)
    a_e_p = (a_e + sigma * eps_e).astype(np.complex64)
    a_m_p = (a_m + sigma * eps_m).astype(np.complex64)
    P_p = _synthesize(a_e_p, a_m_p, basis)
    return P_p, pack_coefficients(a_e_p, a_m_p)


def _coef_mode_dropout(
    P: np.ndarray,
    packed: np.ndarray,
    rng: np.random.Generator,
    dropout_prob: float,
    basis: VSHBasis | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero each mode independently with probability ``dropout_prob``.

    ``dropout_prob = 0`` is the identity. ``dropout_prob >= 1`` zeros every
    mode and yields ``P_p = 0``. The mask is sampled independently per family
    (E vs M) and per sample, matching the generator's ``mode_dropout_prob``
    semantics.
    """
    if basis is None:
        raise ValueError("coef_mode_dropout requires a VSH basis to re-synthesise P")
    if not (0.0 <= dropout_prob <= 1.0):
        raise ValueError(f"dropout_prob must lie in [0, 1]; got {dropout_prob}")
    if dropout_prob == 0.0:
        return P, packed
    a_e, a_m = unpack_coefficients(packed)
    n, K = a_e.shape
    keep_e = rng.uniform(size=(n, K)) >= dropout_prob
    keep_m = rng.uniform(size=(n, K)) >= dropout_prob
    a_e_p = (a_e * keep_e).astype(np.complex64)
    a_m_p = (a_m * keep_m).astype(np.complex64)
    P_p = _synthesize(a_e_p, a_m_p, basis)
    return P_p, pack_coefficients(a_e_p, a_m_p)


def _field_additive_noise(
    P: np.ndarray,
    packed: np.ndarray,
    rng: np.random.Generator,
    relative_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive Gaussian noise on P; targets unchanged (input-noise contract)."""
    if relative_sigma <= 0:
        return P, packed
    n = P.shape[0]
    p_max = P.reshape(n, -1).max(axis=1)
    sigma = (relative_sigma * p_max).astype(np.float32)
    eps = rng.standard_normal(size=P.shape).astype(np.float32)
    P_p = P + sigma[:, None, None] * eps
    np.maximum(P_p, 0.0, out=P_p)
    return P_p, packed


def _field_phi_roll(
    P: np.ndarray, packed: np.ndarray, rng: np.random.Generator, l_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random per-sample roll along phi, with consistent coefficient update.

    ``np.roll(P, k, axis=-1)`` produces ``P_new[..., j] = P[..., (j - k) mod n_phi]``.
    Translating to angles, this is ``P_new(phi) = P(phi - phi_k)`` with
    ``phi_k = 2 pi k / n_phi``. Substituting into the VSH expansion gives
    ``Y_l^m(theta, phi - phi_k) = exp(-i m phi_k) Y_l^m(theta, phi)``, so the
    coefficient-domain equivalent is ``a_{l,m} -> exp(-i m phi_k) a_{l,m}``.
    The vectorial angular operators do not change ``m``, so the same sign holds
    for both VSH families.
    """
    n_phi = P.shape[-1]
    n = P.shape[0]
    ks = rng.integers(0, n_phi, size=n)
    P_p = np.empty_like(P)
    for i in range(n):
        P_p[i] = np.roll(P[i], int(ks[i]), axis=-1)
    a_e, a_m = unpack_coefficients(packed)
    K = a_e.shape[-1]
    m_per_mode = np.array([m for _, m in iter_modes(l_max)], dtype=np.float64)
    if K != m_per_mode.size:
        raise ValueError(
            f"packed K={K} disagrees with l_max={l_max} (expected K={m_per_mode.size})"
        )
    phi_k = (2.0 * np.pi / n_phi) * ks.astype(np.float64)
    phase = np.exp(-1j * np.outer(phi_k, m_per_mode)).astype(np.complex64)
    a_e_p = (a_e * phase).astype(np.complex64)
    a_m_p = (a_m * phase).astype(np.complex64)
    return P_p, pack_coefficients(a_e_p, a_m_p)


def apply_augmentation(
    P: np.ndarray,
    packed: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
    basis: VSHBasis | None = None,
    l_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a single augmentation to a batch of (P, packed) pairs.

    Returns new ``(P, packed)`` arrays. Inputs are not modified in place.
    """
    if isinstance(cfg, CoefPhaseRotationConfig):
        return _coef_phase_rotation(P, packed, rng)
    if isinstance(cfg, CoefAdditiveNoiseConfig):
        return _coef_additive_noise(P, packed, rng, sigma=cfg.sigma, basis=basis)
    if isinstance(cfg, CoefModeDropoutConfig):
        return _coef_mode_dropout(
            P, packed, rng, dropout_prob=cfg.dropout_prob, basis=basis
        )
    if isinstance(cfg, FieldAdditiveNoiseConfig):
        return _field_additive_noise(P, packed, rng, relative_sigma=cfg.relative_sigma)
    if isinstance(cfg, FieldPhiRollConfig):
        if l_max is None:
            raise ValueError("field_phi_roll needs l_max")
        return _field_phi_roll(P, packed, rng, l_max=l_max)
    raise TypeError(f"unknown augmentation config: {type(cfg).__name__}")


def build_augmentation(spec: Mapping[str, Any] | None) -> AugmentationConfig | None:
    """Build an augmentation config from a plain mapping (e.g. Hydra dict).

    Returns ``None`` if ``spec`` is ``None``, falsy, or has ``name`` set to one
    of ``"none"``/``"off"``.
    """
    if not spec:
        return None
    name = spec.get("name", None)
    if name in (None, "none", "off"):
        return None
    if name == "coef_phase_rotation":
        return CoefPhaseRotationConfig()
    if name == "coef_additive_noise":
        return CoefAdditiveNoiseConfig(sigma=float(spec.get("sigma", 0.05)))
    if name == "coef_mode_dropout":
        return CoefModeDropoutConfig(
            dropout_prob=float(spec.get("dropout_prob", 0.1))
        )
    if name == "field_additive_noise":
        return FieldAdditiveNoiseConfig(
            relative_sigma=float(spec.get("relative_sigma", 0.02))
        )
    if name == "field_phi_roll":
        return FieldPhiRollConfig()
    raise ValueError(f"unknown augmentation name: {name!r}")
