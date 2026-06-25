"""Project a tangential complex field onto the project's VSH basis.

The forward operator (:mod:`mpinv.data.synthetic_generator` /
:mod:`mpinv.losses.differentiable_field`) takes coefficients in the project's
basis (defined in :mod:`mpinv.data._basis_cache`, Jackson §9 convention with
strict orthonormal Condon-Shortley) and produces a complex tangential field

    E_a, b(theta, phi) = sum_k a_k Psi_k(theta, phi)        (a, b in {theta, phi})

The *inverse* operation — given a measured ``E_theta, E_phi`` on the project's
1° equiangular grid (poles excluded), recover the coefficients
``(a^E, a^M)`` — is needed when ingesting external data whose multipole
files were written in a *different* VSH convention (e.g. the legacy
``~/Desktop/diplom`` corpus, which applies an extra ``-(1j)^(l+1)`` global
phase per mode plus a sign flip on the electric family). Trying to reuse those
coefficients with the project's basis silently produces inconsistent
``(P, packed)`` pairs — verified empirically: ``||P_meas|| / ||P_resyn||`` is
~250 on the holdout corpus despite L=5 capturing > 99.5% of the multipole
energy.

This module computes the orthogonal projection on the discrete grid using the
basis's defining inner product

    <f, g> = int_S2 f . conj(g) sin(theta) d theta d phi

approximated as a Riemann sum with weights ``sin(theta_i) * d_theta * d_phi``.
The project's basis is orthonormal under that inner product (verified by the
companion test ``tests/unit/test_basis_decomposer.py``), so the projection is
exact up to discretisation and bandlimit-truncation error: re-synthesising
``P`` from the resulting ``packed`` reproduces the input ``P`` to within the
energy fraction lost outside ``l <= l_max``.

Public API
----------

* :func:`decompose_field_to_packed` — single (or batched) E -> packed.
* :func:`decomposition_residual` — diagnostic ``||E - E_resyn|| / ||E||``.
"""

from __future__ import annotations

import numpy as np

from mpinv.core.grid import GridSpec
from mpinv.core.packing import pack_coefficients
from mpinv.data._basis_cache import VSHBasis


def _quadrature_weights(grid: GridSpec) -> np.ndarray:
    """``sin(theta_i) * d_theta * d_phi`` shaped ``(n_theta,)``.

    The d_phi factor is constant across phi, so we collapse it into the
    theta-axis weight to keep the einsum 4-axis.
    """
    theta = grid.theta_axis()
    return (np.sin(theta) * grid.dtheta * grid.dphi).astype(np.float64)


def decompose_field_to_packed(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    *,
    basis: VSHBasis,
    grid: GridSpec,
) -> np.ndarray:
    """Project a tangential complex field onto the project's VSH basis.

    Parameters
    ----------
    E_theta, E_phi : np.ndarray
        Complex arrays of shape ``(n_theta, n_phi)`` *or* ``(N, n_theta, n_phi)``
        for batched input. Shapes must match.
    basis : VSHBasis
        Pre-computed basis tensor (``(K, 2, 2, n_theta, n_phi)``). Must be on the
        same grid and at the same ``l_max`` you want to recover.
    grid : GridSpec
        Grid spec; used for the quadrature weights.

    Returns
    -------
    np.ndarray
        Packed coefficient vector(s) of shape ``(4 K,)`` or ``(N, 4 K)``,
        following :func:`mpinv.core.packing.pack_coefficients`.
    """
    if E_theta.shape != E_phi.shape:
        raise ValueError(
            f"E_theta and E_phi shape mismatch: {E_theta.shape} vs {E_phi.shape}"
        )
    if E_theta.shape[-2:] != (grid.n_theta, grid.n_phi):
        raise ValueError(
            f"E shape {E_theta.shape[-2:]} does not match grid "
            f"({grid.n_theta}, {grid.n_phi})"
        )
    if basis.basis.shape[-2:] != (grid.n_theta, grid.n_phi):
        raise ValueError(
            f"basis shape {basis.basis.shape[-2:]} does not match grid "
            f"({grid.n_theta}, {grid.n_phi})"
        )

    batched = E_theta.ndim == 3
    if not batched:
        E_theta_b = E_theta[None, ...]
        E_phi_b = E_phi[None, ...]
    else:
        E_theta_b = E_theta
        E_phi_b = E_phi

    weight = _quadrature_weights(grid)
    psi_conj = np.conj(basis.basis)
    a = np.einsum(
        "kfctp,nctp,t->nkf",
        psi_conj,
        np.stack([E_theta_b, E_phi_b], axis=1).astype(np.complex128),
        weight,
        optimize=True,
    )
    a_e = a[..., 0].astype(np.complex64)
    a_m = a[..., 1].astype(np.complex64)
    packed = pack_coefficients(a_e, a_m)
    if not batched:
        return packed[0]
    return packed


def decomposition_residual(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    *,
    basis: VSHBasis,
    grid: GridSpec,
) -> dict[str, float]:
    """Diagnose how much of the field is captured by the projection at ``l_max``.

    Returns a dict with four diagnostic scalars (averaged over the batch):

    * ``e_rel_residual`` — ``||E - E_resyn||_w / ||E||_w`` (norm under the same
      sin-theta-weighted inner product used for the projection).
    * ``p_rel_rmse`` — ``||P_meas - P_resyn||_2 / ||P_meas||_2`` (unweighted L2
      on P = |E|^2). This is the metric the model's report uses.
    * ``e_norm_w`` — ``||E||_w`` for context.
    * ``p_norm`` — ``||P_meas||_2`` for context.
    """
    packed = decompose_field_to_packed(E_theta, E_phi, basis=basis, grid=grid)
    if packed.ndim == 1:
        packed = packed[None, :]
        E_theta_b = E_theta[None, ...]
        E_phi_b = E_phi[None, ...]
    else:
        E_theta_b = E_theta
        E_phi_b = E_phi

    from mpinv.core.packing import unpack_coefficients

    a_e, a_m = unpack_coefficients(packed)
    E_e = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E_resyn = E_e + E_m

    weight = _quadrature_weights(grid)

    def _w_norm(field: np.ndarray) -> np.ndarray:
        sq = (field.real**2 + field.imag**2).sum(axis=1)
        return np.sqrt((sq * weight[None, :, None]).sum(axis=(-2, -1)))

    e_norm = _w_norm(np.stack([E_theta_b, E_phi_b], axis=1).astype(np.complex128))
    e_resid_norm = _w_norm(
        np.stack([E_theta_b, E_phi_b], axis=1).astype(np.complex128) - E_resyn
    )
    e_rel = e_resid_norm / np.maximum(e_norm, 1e-30)

    P_meas = (np.abs(E_theta_b) ** 2 + np.abs(E_phi_b) ** 2).astype(np.float64)
    P_resyn = (E_resyn.real**2 + E_resyn.imag**2).sum(axis=1)
    p_norm = np.linalg.norm(P_meas.reshape(P_meas.shape[0], -1), axis=1)
    p_resid = np.linalg.norm(
        (P_meas - P_resyn).reshape(P_meas.shape[0], -1), axis=1
    )
    p_rel = p_resid / np.maximum(p_norm, 1e-30)

    return {
        "e_rel_residual": float(e_rel.mean()),
        "p_rel_rmse": float(p_rel.mean()),
        "e_norm_w": float(e_norm.mean()),
        "p_norm": float(p_norm.mean()),
    }
