"""
Fast multipole library generator: E_lm and M_lm field files on a spherical grid.
Refactored and vectorized (NumPy) for speed. Deterministic (no random).

Aligned with MPField_Spherical_Write_Updated.py:
  - Grid: same loop (theta[i,j], phi[i,j]), same formula.
  - Y: sph_harm(m, l, phi, theta) — prefer sph_harm when available.
  - X: th_1/ph_1 zero at poles (no div by sin); ph_2 computed everywhere (incl. poles).
  - Amplitude: -(1j)**(l+1) * [X_phi, -X_theta] for E; pole correction /sqrt(2).
  - Phase: we write np.angle(-amplitude) so phase matches original (compensates sign from vectorized vs scalar).
  - Output: same columns, same header; data fmt %.17g.
"""
import os
import numpy as np

# Reproducibility: no random in this script; fix seed in case of future changes
np.random.seed(0)

# Prefer sph_harm (same as original); use sph_harm_y when sph_harm is unavailable (SciPy 1.15+ / Python 3.14)
try:
    from scipy.special import sph_harm as _sph_harm_impl
    _use_sph_harm_y = False
except (ImportError, AttributeError):
    try:
        from scipy.special import sph_harm_y as _sph_harm_impl
        _use_sph_harm_y = True
    except ImportError as e:
        raise ImportError(
            "Neither scipy.special.sph_harm nor sph_harm_y is available. "
            "Install an older scipy (e.g. 1.11) for sph_harm, or a newer one with sph_harm_y."
        ) from e

# -----------------------------------------------------------------------------
# Config (canonical decomposition grid)
# -----------------------------------------------------------------------------
CALC_STEP = 1  # degrees (original: calc_step = 1)
MINORDER = 1
MAXORDER = 15
OUTPUT_DIR = "FieldsFast0.5"
# Grid: n_phi = int(360/CALC_STEP), n_theta = int(180/CALC_STEP)-1 -> (360, 179) for CALC_STEP=1

# -----------------------------------------------------------------------------
# Vectorized spherical harmonics and vector functions (array-friendly)
# -----------------------------------------------------------------------------


def Y_vec(l, m, theta, phi):
    """Spherical harmonic Y_lm. Same convention as original: sph_harm(m, l, phi, theta)."""
    if _use_sph_harm_y:
        return _sph_harm_impl(l, m, theta, phi)
    return _sph_harm_impl(m, l, phi, theta)


def X_vec(l, m, theta, phi):
    """
    Vector harmonic X_lm (magnetic-type) in spherical components [r, theta, phi].
    theta, phi: arrays of same shape. Returns shape (..., 3), complex.
    """
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    Y_lm = Y_vec(l, m, theta, phi)
    # Avoid division by zero at poles
    safe = np.where(sin_th > 1e-14, sin_th, 1.0)
    th_1 = np.where(sin_th > 1e-14, -m * Y_lm / safe, 0.0)
    ph_1 = np.where(sin_th > 1e-14, -1j * m * cos_th * Y_lm / safe, 0.0)
    ph_2 = np.zeros_like(Y_lm, dtype=complex)
    if m + 1 <= l:
        ph_2 = (
            -1j
            * np.sqrt((l - m) * (l + m + 1))
            * np.exp(-1j * phi)
            * Y_vec(l, m + 1, theta, phi)
        )
    norm = np.sqrt(l * (l + 1))
    zero = np.zeros_like(Y_lm)
    return np.stack([zero, (th_1 + zero) / norm, (ph_1 + ph_2) / norm], axis=-1)


def Xn_vec(l, m, theta, phi):
    """Vector harmonic Xn_lm (electric-type): [0, X_phi, -X_theta]. Shape (..., 3)."""
    x = X_vec(l, m, theta, phi)
    return np.stack([x[..., 0], x[..., 2], -x[..., 1]], axis=-1)


def X_from_Y_batch(Y_k, Y_next, l, m, theta, phi):
    """
    Compute X_lm (3 components) from precomputed Y_lm and Y_{l,m+1}.
    Scalar version: Y_k, Y_next shape (n_phi, n_theta); returns (n_phi, n_theta, 3).
    Batched version: Y_k, Y_next shape (n_lm, n_phi, n_theta); l, m shape (n_lm,) or (n_lm,1,1);
      theta, phi (n_phi, n_theta). Returns (n_lm, n_phi, n_theta, 3).
    """
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    safe = np.where(sin_th > 1e-14, sin_th, 1.0)
    th_1 = np.where(sin_th > 1e-14, -m * Y_k / safe, 0.0)
    ph_1 = np.where(sin_th > 1e-14, -1j * m * cos_th * Y_k / safe, 0.0)
    coef = -1j * np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * phi)
    # ph_2 has no 1/sin(theta); compute everywhere to match original (including at poles)
    ph_2 = coef * Y_next
    norm = np.sqrt(l * (l + 1))
    zero = np.zeros_like(Y_k)
    return np.stack(
        [zero, (th_1 + zero) / norm, (ph_1 + ph_2) / norm],
        axis=-1,
    )


def X_flat_from_Y_flat(Y_flat, next_k, L_flat, M_flat, theta, phi):
    """
    Vectorized over all multipole indices k. No Python loop over (l, m).
    Y_flat: (n_lm, n_phi, n_theta), next_k: (n_lm,) -> indices into Y_flat for Y_{l,m+1}.
    Returns X_flat (n_lm, n_phi, n_theta, 3).
    """
    n_lm, n_phi, n_theta = Y_flat.shape
    valid = next_k >= 0
    idx = np.where(valid, next_k, 0)
    Y_next = np.where(
        valid[:, np.newaxis, np.newaxis],
        Y_flat[idx, :, :],
        0.0,
    )
    # Broadcast l, m to (n_lm, 1, 1) for use with (n_lm, n_phi, n_theta)
    l = L_flat[:, np.newaxis, np.newaxis]
    m = M_flat[:, np.newaxis, np.newaxis]
    return X_from_Y_batch(Y_flat, Y_next, l, m, theta, phi)


# -----------------------------------------------------------------------------
# Grid and field computation
# -----------------------------------------------------------------------------


def build_grid(step_deg):
    """Build canonical theta/phi grid in radians (phi, theta) without poles."""
    n_phi = int(360 / step_deg)
    n_theta = int(180 / step_deg) - 1
    phi_deg = np.arange(n_phi, dtype=float) * step_deg
    theta_deg = (np.arange(n_theta, dtype=float) + 1.0) * step_deg
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    phi = np.broadcast_to(phi_rad[:, np.newaxis], (n_phi, n_theta))
    theta = np.broadcast_to(theta_rad[np.newaxis, :], (n_phi, n_theta))
    return theta, phi


def _apply_pole_correction_if_present(amplitude, theta):
    """Apply legacy 1/sqrt(2) correction only when poles are present in theta grid."""
    if theta.shape[1] == 0:
        return amplitude
    if np.isclose(theta[0, 0], 0.0):
        amplitude[:, 0, :] /= np.sqrt(2)
    if np.isclose(theta[0, -1], np.pi):
        amplitude[:, -1, :] /= np.sqrt(2)
    return amplitude


def field_for_multipole(l, m, theta, phi, electric=True):
    """
    Get E-field (theta, phi components only) on the grid for a single multipole
    with coefficient 1. electric=True -> E_lm (Xn), else M_lm (X).
    Returns amplitude shape (n_phi, n_theta, 2), complex.
    """
    if electric:
        vec = Xn_vec(l, m, theta, phi)
    else:
        vec = X_vec(l, m, theta, phi)
    # vec[..., 1] = theta component, vec[..., 2] = phi component
    amp = -(1j) ** (l + 1) * vec[..., 1:3].copy()
    amp = _apply_pole_correction_if_present(amp, theta)
    return amp


def power_from_amplitude(amplitude):
    """|E|^2 on grid. amplitude shape (..., 2)."""
    return np.sum(np.abs(amplitude) ** 2, axis=-1)


def write_field_file(path, theta, phi, power, amplitude, l, m, kind):
    """Write one E_lm or M_lm file with header and data block."""
    # Header: consumer (e.g. 3 FieldsToMultipoles.py) skips startline=43
    header_lines = (
        ["Electric multipoles:\n"]
        + ["{}: {}\n".format(ll, {mm: 0.0 for mm in range(-ll, ll + 1)}) for ll in range(1, 16)]
        + ["Magnetic multipoles:\n"]
        + ["{}: {}\n".format(ll, {mm: 0.0 for mm in range(-ll, ll + 1)}) for ll in range(1, 16)]
        + ["#\n"] * 10
        + ["Theta Phi Abs(Power) Abs(Theta) Phase(Theta) Abs(Phi) Phase(Phi)\n"]
    )
    # Data: row order as original (for i in range(n_phi): for j in range(n_theta): theta[i,j], phi[i,j], ...)
    th_flat = np.rad2deg(theta).ravel()
    ph_flat = np.rad2deg(phi).ravel()
    pwr_flat = power.ravel()
    abs_th = np.abs(amplitude[:, :, 0]).ravel()
    abs_ph = np.abs(amplitude[:, :, 1]).ravel()
    # Phase: match original. When both use sph_harm_y, use angle(amplitude); else angle(-amplitude) for legacy.
    PHASE_EPS = 1e-14
    if _use_sph_harm_y:
        phs_th = np.angle(amplitude[:, :, 0]).ravel()
        phs_ph = np.angle(amplitude[:, :, 1]).ravel()
    else:
        phs_th = np.angle(-amplitude[:, :, 0]).ravel()
        phs_ph = np.angle(-amplitude[:, :, 1]).ravel()
    phs_th[abs_th < PHASE_EPS] = 0.0
    phs_ph[abs_ph < PHASE_EPS] = 0.0
    data = np.column_stack(
        (th_flat, ph_flat, pwr_flat, abs_th, phs_th, abs_ph, phs_ph)
    )
    base = os.path.dirname(path)
    if base:
        os.makedirs(base, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(header_lines)
        np.savetxt(f, data, fmt="%.17g")
    return path


# -----------------------------------------------------------------------------
# (l, m) cycle logic — multipole index set
# -----------------------------------------------------------------------------
#
# We iterate over all valid (l, m) pairs for the multipole expansion:
#
#   - l (order):  minorder .. maxorder  (e.g. 1 .. 15)
#   - m (azimuthal):  for each l,  m = -l .. +l  (2l+1 values)
#
# So the set is: (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2), ...
# Total number of pairs:  n_lm = sum(2l+1 for l in 1..maxorder)
#                        = maxorder*(maxorder+2)  [since sum_{l=1}^L (2l+1) = L(L+2)]
#
# Flat index k in 0 .. n_lm-1  corresponds to (l, m) with:
#   k = (l-1)*l + (m+l)  =>  l = floor((sqrt(1+4*k)-1)/2)+1,  m = k - l*(l-1) - l
# or we build L_flat[k], M_flat[k] explicitly.
#
# For each (l, m) we compute X_vec(l, m, theta, phi), which uses Y_lm and Y_{l,m+1}.
# So we can batch: precompute all Y_lm into Y_flat[k], then compute X for all k
# in one vectorized step (no loop over k for the heavy math).
# -----------------------------------------------------------------------------


def build_multipole_indices(minorder, maxorder):
    """
    Return flat arrays of (l, m) and the index of (l, m+1) for each k.
    Returns:
        L_flat, M_flat: int arrays shape (n_lm,)
        next_k: int array shape (n_lm,) — index of (l, m+1), or -1 if m+1 > l
    """
    pairs = [
        (l, m)
        for l in range(minorder, maxorder + 1)
        for m in range(-l, l + 1)
    ]
    L_flat = np.array([p[0] for p in pairs], dtype=np.intp)
    M_flat = np.array([p[1] for p in pairs], dtype=np.intp)
    # (l, m+1) is next in the same l-block when m < l; else no successor
    next_k = np.where(
        M_flat < L_flat,
        np.arange(len(pairs), dtype=np.intp) + 1,
        -1,
    )
    return L_flat, M_flat, next_k


def generate_library(
    output_dir=OUTPUT_DIR,
    calc_step=CALC_STEP,
    minorder=MINORDER,
    maxorder=MAXORDER,
):
    """
    Generate E_lm and M_lm field files for all (l, m).
    Processes one (l, m) at a time to keep memory low (avoids OOM on large grids).
    """
    theta, phi = build_grid(calc_step)
    n_pairs = sum(2 * l + 1 for l in range(minorder, maxorder + 1))
    total = 0
    current = 0
    for l in range(minorder, maxorder + 1):
        for m in range(-l, l + 1):
            current += 1
            print("Processing (l, m) = ({}, {}) [{}/{}]".format(l, m, current, n_pairs), flush=True)
            # Y for this (l, m) and for (l, m+1) if needed for X
            Y_k = Y_vec(l, m, theta, phi)
            if m + 1 <= l:
                Y_next = Y_vec(l, m + 1, theta, phi)
            else:
                Y_next = np.zeros_like(Y_k)
            X = X_from_Y_batch(Y_k, Y_next, l, m, theta, phi)  # (n_phi, n_theta, 3)
            factor = -(1j) ** (l + 1)
            amp_e = factor * np.stack([X[..., 2], -X[..., 1]], axis=-1)
            amp_m = factor * X[..., 1:3].copy()
            amp_e = _apply_pole_correction_if_present(amp_e, theta)
            amp_m = _apply_pole_correction_if_present(amp_m, theta)
            power_e = power_from_amplitude(amp_e)
            power_m = power_from_amplitude(amp_m)
            write_field_file(
                os.path.join(output_dir, "E_l{:d}_m{:d}.txt".format(l, m)),
                theta, phi, power_e, amp_e, l, m, "E",
            )
            write_field_file(
                os.path.join(output_dir, "M_l{:d}_m{:d}.txt".format(l, m)),
                theta, phi, power_m, amp_m, l, m, "M",
            )
            total += 2
    return total


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    n = generate_library()
    t1 = time.perf_counter()
    print("Wrote {} field files in {:.2f} s".format(n, t1 - t0))
