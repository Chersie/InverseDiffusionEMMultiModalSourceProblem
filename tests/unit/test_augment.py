"""Tests for the augmentation library.

Each augmentation is verified against its physical-consistency contract.
"""

from __future__ import annotations

import numpy as np

from mpinv.data.augment import (
    CoefAdditiveNoiseConfig,
    CoefModeDropoutConfig,
    CoefPhaseRotationConfig,
    FieldAdditiveNoiseConfig,
    FieldPhiRollConfig,
    apply_augmentation,
    build_augmentation,
)
from mpinv.data.synthetic_generator import SyntheticGenerator


def _resynthesize(packed: np.ndarray, basis_basis: np.ndarray) -> np.ndarray:
    from mpinv.core.packing import unpack_coefficients

    a_e, a_m = unpack_coefficients(packed)
    E_e = np.einsum("nk,kctp->nctp", a_e, basis_basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis_basis[:, 1])
    E = E_e + E_m
    return (E.real**2 + E.imag**2).sum(axis=1).astype(np.float32)


def test_coef_phase_rotation_keeps_P_invariant(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(0)
    P, packed = tiny_generator.generate_batch(8, rng)
    rng_aug = np.random.default_rng(123)
    P2, packed2 = apply_augmentation(
        P, packed, cfg=CoefPhaseRotationConfig(), rng=rng_aug, basis=tiny_generator.basis
    )
    # P unchanged
    assert np.array_equal(P, P2)
    # packed should differ (with overwhelming probability for random alpha)
    assert not np.array_equal(packed, packed2)
    # Physical check: re-synthesising from packed2 must yield the same P (within fp tol)
    P_resyn = _resynthesize(packed2, tiny_generator.basis.basis)
    np.testing.assert_allclose(P, P_resyn, rtol=1e-4, atol=1e-5)


def test_coef_additive_noise_resynthesis_consistent(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(1)
    P, packed = tiny_generator.generate_batch(4, rng)
    rng_aug = np.random.default_rng(456)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefAdditiveNoiseConfig(sigma=0.1),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    # P must change (sigma > 0)
    assert not np.allclose(P, P2)
    # packed must change
    assert not np.allclose(packed, packed2)
    # P2 must match the re-synthesis of packed2 to floating-point tolerance
    P_resyn = _resynthesize(packed2, tiny_generator.basis.basis)
    np.testing.assert_allclose(P2, P_resyn, rtol=1e-4, atol=1e-5)


def test_coef_additive_noise_zero_sigma_is_identity(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(2)
    P, packed = tiny_generator.generate_batch(4, rng)
    rng_aug = np.random.default_rng(0)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefAdditiveNoiseConfig(sigma=0.0),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    # With sigma=0 the perturbed coefficients are identical
    np.testing.assert_allclose(packed, packed2, atol=1e-7)
    # And re-synthesised P matches the original (within fp tol of einsum cast)
    np.testing.assert_allclose(P, P2, rtol=1e-4, atol=1e-5)


def test_coef_mode_dropout_zero_prob_is_identity(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(0)
    P, packed = tiny_generator.generate_batch(4, rng)
    rng_aug = np.random.default_rng(11)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefModeDropoutConfig(dropout_prob=0.0),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    np.testing.assert_array_equal(P, P2)
    np.testing.assert_array_equal(packed, packed2)


def test_coef_mode_dropout_full_prob_zeros_everything(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(1)
    P, packed = tiny_generator.generate_batch(3, rng)
    rng_aug = np.random.default_rng(13)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefModeDropoutConfig(dropout_prob=1.0),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    np.testing.assert_allclose(packed2, 0.0, atol=1e-7)
    np.testing.assert_allclose(P2, 0.0, atol=1e-7)


def test_coef_mode_dropout_resynthesis_consistent(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(2)
    P, packed = tiny_generator.generate_batch(8, rng)
    rng_aug = np.random.default_rng(17)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefModeDropoutConfig(dropout_prob=0.3),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    P_resyn = _resynthesize(packed2, tiny_generator.basis.basis)
    np.testing.assert_allclose(P2, P_resyn, rtol=1e-4, atol=1e-5)
    # Some coefficients must be exactly zero (mask hits with overwhelming probability
    # for 8 * 4K trials at p=0.3).
    assert (packed2 == 0.0).any()


def test_coef_mode_dropout_rate_within_tolerance(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(3)
    P, packed = tiny_generator.generate_batch(64, rng)
    rng_aug = np.random.default_rng(19)
    _, packed2 = apply_augmentation(
        P,
        packed,
        cfg=CoefModeDropoutConfig(dropout_prob=0.25),
        rng=rng_aug,
        basis=tiny_generator.basis,
    )
    # Exactly-zero entries should be a fraction close to 0.25 across all packed
    # samples and components (binomial CI ~ 0.25 +/- 5% at this n).
    zero_frac = float((packed2 == 0.0).mean())
    assert 0.20 <= zero_frac <= 0.30, f"empirical dropout rate out of range: {zero_frac}"


def test_field_additive_noise_targets_unchanged(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(3)
    P, packed = tiny_generator.generate_batch(8, rng)
    rng_aug = np.random.default_rng(111)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=FieldAdditiveNoiseConfig(relative_sigma=0.1),
        rng=rng_aug,
    )
    # Targets unchanged (input-noise contract)
    assert np.array_equal(packed, packed2)
    # P changed but stays non-negative
    assert not np.array_equal(P, P2)
    assert (P2 >= 0.0).all()


def test_field_phi_roll_consistent_with_coef_rotation(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(4)
    P, packed = tiny_generator.generate_batch(6, rng)
    rng_aug = np.random.default_rng(99)
    P2, packed2 = apply_augmentation(
        P,
        packed,
        cfg=FieldPhiRollConfig(),
        rng=rng_aug,
        l_max=tiny_generator.cfg.l_max,
    )
    # P must be the re-synthesis of packed2 (the augmentation rotates both
    # consistently). On the 1-degree azimuth grid the rotation is exact.
    P_resyn = _resynthesize(packed2, tiny_generator.basis.basis)
    np.testing.assert_allclose(P2, P_resyn, rtol=1e-4, atol=1e-5)


def test_field_phi_roll_preserves_total_energy(tiny_generator: SyntheticGenerator):
    rng = np.random.default_rng(5)
    P, packed = tiny_generator.generate_batch(4, rng)
    rng_aug = np.random.default_rng(7)
    P2, _ = apply_augmentation(
        P,
        packed,
        cfg=FieldPhiRollConfig(),
        rng=rng_aug,
        l_max=tiny_generator.cfg.l_max,
    )
    # np.roll is a permutation; total sum is preserved exactly per sample
    np.testing.assert_allclose(P.sum(axis=(-2, -1)), P2.sum(axis=(-2, -1)), rtol=1e-6)


def test_build_augmentation_dispatch():
    assert build_augmentation(None) is None
    assert build_augmentation({"name": "none"}) is None
    assert isinstance(build_augmentation({"name": "coef_phase_rotation"}), CoefPhaseRotationConfig)
    cfg = build_augmentation({"name": "coef_additive_noise", "sigma": 0.2})
    assert isinstance(cfg, CoefAdditiveNoiseConfig)
    assert cfg.sigma == 0.2
    cfg2 = build_augmentation({"name": "field_additive_noise", "relative_sigma": 0.05})
    assert isinstance(cfg2, FieldAdditiveNoiseConfig)
    assert cfg2.relative_sigma == 0.05
    cfg3 = build_augmentation({"name": "coef_mode_dropout", "dropout_prob": 0.2})
    assert isinstance(cfg3, CoefModeDropoutConfig)
    assert cfg3.dropout_prob == 0.2
    assert isinstance(build_augmentation({"name": "field_phi_roll"}), FieldPhiRollConfig)
