"""
Physics-aware evaluation metrics for radiation pattern prediction.

All public functions accept batched arrays:
    p_true, p_pred : (N, n_points) float32/float64
    sin_theta      : (n_points,) float32  — spherical area weight for each grid point

Grid convention (matches build_dataset):
    flat index = phi_idx * 179 + theta_idx
    theta_idx ∈ 0..178  →  theta = (theta_idx + 1) degrees  (1° … 179°)
    phi_idx   ∈ 0..359  →  phi   = phi_idx degrees           (0° … 359°)
    reshape to (360, 179) = (n_phi, n_theta)

See docs/metrics_notes.md for the design rationale behind each metric.
"""
from __future__ import annotations

import numpy as np

_N_PHI = 360
_N_THETA = 179
_DEG2RAD = np.pi / 180.0

# 1-degree steps for both axes (matching the canonical grid).
_THETA_DEG = np.arange(1, 180, dtype=np.float64)          # shape (179,)
_PHI_DEG = np.arange(0, 360, dtype=np.float64)            # shape (360,)
_THETA_RAD = _THETA_DEG * _DEG2RAD
_PHI_RAD = _PHI_DEG * _DEG2RAD


def _to_2d(flat: np.ndarray) -> np.ndarray:
    """Reshape (n_points,) → (n_phi=360, n_theta=179)."""
    return flat.reshape(_N_PHI, _N_THETA)


# ---------------------------------------------------------------------------
# Idea 1 & 3 — Area-weighted metrics (sin θ pole correction)
# ---------------------------------------------------------------------------

def weighted_mse(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    sin_theta: np.ndarray,
) -> float:
    """
    Area-weighted MSE on the spherical grid.

    Each grid point is weighted by sin(θ) — proportional to the solid angle
    it represents — so polar oversampling does not dominate the error.

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays
    sin_theta      : (n_points,) weights (sin of polar angle for each point)

    Returns
    -------
    Scalar float.
    """
    w = sin_theta.astype(np.float64)
    w_sum = w.sum()
    diff2 = (p_true.astype(np.float64) - p_pred.astype(np.float64)) ** 2
    return float((diff2 * w).sum() / (p_true.shape[0] * w_sum))


def total_power_log_mse(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    sin_theta: np.ndarray,
) -> float:
    """
    Mean squared error of log(P), with P the area-weighted total power per sample.

    P = Σ_j w_j p_j,  w_j ∝ sin(θ_j),  w normalised to sum to 1 — matches the
    physics trainer's amplitude term (evaluation, not training).
    """
    w = sin_theta.astype(np.float64)
    w = w / w.sum()
    wr = w[np.newaxis, :]
    pt = p_true.astype(np.float64)
    pp = p_pred.astype(np.float64)
    p_true_tot = (pt * wr).sum(axis=1).clip(1e-12)
    p_pred_tot = (pp * wr).sum(axis=1).clip(1e-12)
    return float(np.mean((np.log(p_pred_tot) - np.log(p_true_tot)) ** 2))


def weighted_rel_l2(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    sin_theta: np.ndarray,
) -> float:
    """
    Area-weighted relative L2 error, averaged over samples.

    rel_l2_i = sqrt( Σ_j w_j (pred_ij - true_ij)² ) / sqrt( Σ_j w_j true_ij² )

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays
    sin_theta      : (n_points,) weights

    Returns
    -------
    Mean relative L2 over samples (scalar float).
    """
    w = sin_theta.astype(np.float64)
    pt = p_true.astype(np.float64)
    pp = p_pred.astype(np.float64)
    num = np.sqrt(((pt - pp) ** 2 * w).sum(axis=1))
    den = np.sqrt((pt ** 2 * w).sum(axis=1)) + 1e-12
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Idea 4 — dB-scale RMSE
# ---------------------------------------------------------------------------

def db_rmse(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    floor_db: float = -40.0,
) -> float:
    """
    RMSE in dB scale, normalised per sample to its own peak.

    Converts each sample's power pattern to dB relative to its maximum:
        P_dB(i) = 10 · log10( P(i) / max(P) + ε )
    then clips at floor_db and computes RMSE over all samples and grid points.

    This gives equal weight to sidelobe accuracy instead of being dominated
    by the main-beam magnitude.

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays
    floor_db       : dB floor applied before RMSE (default -40 dB)

    Returns
    -------
    Scalar float (dB).
    """
    eps = 1e-12
    pt = p_true.astype(np.float64)
    pp = p_pred.astype(np.float64)

    # Normalise each sample by its own true-field peak.
    peak = pt.max(axis=1, keepdims=True).clip(min=eps)
    pt_db = 10.0 * np.log10(pt / peak + eps)
    pp_db = 10.0 * np.log10(pp / peak + eps)

    pt_db = np.clip(pt_db, floor_db, 0.0)
    pp_db = np.clip(pp_db, floor_db, 0.0)

    return float(np.sqrt(np.mean((pp_db - pt_db) ** 2)))


# ---------------------------------------------------------------------------
# Idea 2 — Main-beam pointing error
# ---------------------------------------------------------------------------

def beam_pointing_error_deg(
    p_true: np.ndarray,
    p_pred: np.ndarray,
) -> float:
    """
    Mean great-circle angular distance between the main-beam peaks.

    For each sample, find the flat argmax of p_true and p_pred, convert to
    spherical coordinates, then compute the great-circle distance via the
    spherical law of cosines.

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays

    Returns
    -------
    Mean angular error in degrees (scalar float).
    """
    idx_true = np.argmax(p_true, axis=1)   # (N,)
    idx_pred = np.argmax(p_pred, axis=1)   # (N,)

    # Convert flat index → (theta_idx, phi_idx)
    # flat = phi_idx * 179 + theta_idx
    theta_idx_t = idx_true % _N_THETA
    phi_idx_t = idx_true // _N_THETA
    theta_idx_p = idx_pred % _N_THETA
    phi_idx_p = idx_pred // _N_THETA

    # Degrees → radians
    th_t = (theta_idx_t + 1).astype(np.float64) * _DEG2RAD
    th_p = (theta_idx_p + 1).astype(np.float64) * _DEG2RAD
    ph_t = phi_idx_t.astype(np.float64) * _DEG2RAD
    ph_p = phi_idx_p.astype(np.float64) * _DEG2RAD

    # Spherical law of cosines: cos d = sin θ₁ sin θ₂ cos Δφ + cos θ₁ cos θ₂
    cos_d = (
        np.sin(th_t) * np.sin(th_p) * np.cos(ph_p - ph_t)
        + np.cos(th_t) * np.cos(th_p)
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    d_deg = np.degrees(np.arccos(cos_d))
    return float(np.mean(d_deg))


# ---------------------------------------------------------------------------
# Idea 5 — Gradient MAE (derivative-based metric)
# ---------------------------------------------------------------------------

def gradient_mae(
    p_true: np.ndarray,
    p_pred: np.ndarray,
) -> float:
    """
    MAE of the estimated gradient magnitude on the sphere surface.

    Reshapes each sample to (n_phi=360, n_theta=179) and applies
    numpy.gradient along both axes (central finite differences).
    Computes the gradient magnitude |∇P| = sqrt((∂P/∂θ)² + (∂P/∂φ)²)
    and returns the mean absolute error across samples and grid points.

    This captures whether the predicted field has the correct spatial
    structure: sharp beam edges, correct sidelobe transitions.

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays

    Returns
    -------
    Scalar float (same units as input power / degree).
    """
    N = p_true.shape[0]
    total_err = 0.0
    for i in range(N):
        pt_2d = _to_2d(p_true[i].astype(np.float64))
        pp_2d = _to_2d(p_pred[i].astype(np.float64))

        # np.gradient returns [grad_axis0, grad_axis1] = [∂/∂phi, ∂/∂theta]
        gt_phi, gt_th = np.gradient(pt_2d)
        gp_phi, gp_th = np.gradient(pp_2d)

        mag_t = np.sqrt(gt_phi ** 2 + gt_th ** 2)
        mag_p = np.sqrt(gp_phi ** 2 + gp_th ** 2)
        total_err += float(np.mean(np.abs(mag_p - mag_t)))

    return total_err / N


# ---------------------------------------------------------------------------
# Bonus — Fractions Skill Score (spatial beam-shape overlap)
# ---------------------------------------------------------------------------

def fss(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    threshold_db: float = -3.0,
    window: int = 5,
) -> float:
    """
    Fractions Skill Score (Roberts & Lean 2008) adapted for radiation patterns.

    Applies a threshold at (peak + threshold_db) dB relative to the true
    pattern peak, creating binary "beam present / absent" maps.  Then
    computes the fraction of above-threshold pixels within every sliding
    window of size (window × window) and evaluates how closely the predicted
    fractions match the true fractions.

    FSS = 1 − MSE(frac_pred, frac_true) / MSE_ref
    where MSE_ref = (mean(frac_pred²) + mean(frac_true²)) / 2  [worst-case]

    A score of 1 is perfect; 0 means no better than random spatial placement.
    0.5 is considered the threshold of "useful" skill in meteorology.

    Parameters
    ----------
    p_true, p_pred   : (N, n_points) float arrays
    threshold_db     : dB threshold relative to true peak (default -3 dB = HPBW)
    window           : sliding window size in grid cells (default 5)

    Returns
    -------
    Mean FSS across samples (scalar float, range 0–1).
    """
    eps = 1e-12
    scores: list[float] = []

    for i in range(p_true.shape[0]):
        pt = _to_2d(p_true[i].astype(np.float64))
        pp = _to_2d(p_pred[i].astype(np.float64))

        peak = float(pt.max()) + eps
        thresh = peak * (10.0 ** (threshold_db / 10.0))

        bt = (pt >= thresh).astype(np.float64)
        bp = (pp >= thresh).astype(np.float64)

        # Compute sliding-window fraction via 2-D cumulative sum (integral image).
        def _window_fractions(binary: np.ndarray) -> np.ndarray:
            cs = np.cumsum(np.cumsum(binary, axis=0), axis=1)
            # Pad with zeros for boundary handling.
            cs_pad = np.zeros((cs.shape[0] + 1, cs.shape[1] + 1))
            cs_pad[1:, 1:] = cs
            half = window // 2
            n_phi, n_theta = binary.shape
            # Use view-based sliding sum: clip coordinates to valid range.
            r1 = np.arange(n_phi)
            r2 = np.arange(n_theta)
            # Row ranges (clamped).
            row_lo = np.clip(r1 - half - 1, 0, n_phi).reshape(-1, 1)
            row_hi = np.clip(r1 + half, 0, n_phi).reshape(-1, 1)
            col_lo = np.clip(r2 - half - 1, 0, n_theta).reshape(1, -1)
            col_hi = np.clip(r2 + half, 0, n_theta).reshape(1, -1)
            win_sum = (
                cs_pad[row_hi, col_hi]
                - cs_pad[row_lo, col_hi]
                - cs_pad[row_hi, col_lo]
                + cs_pad[row_lo, col_lo]
            )
            win_area = (row_hi - row_lo) * (col_hi - col_lo) + eps
            return win_sum / win_area

        ft = _window_fractions(bt)
        fp = _window_fractions(bp)

        mse_num = float(np.mean((fp - ft) ** 2))
        mse_ref = float((np.mean(fp ** 2) + np.mean(ft ** 2)) / 2.0) + eps
        scores.append(1.0 - mse_num / mse_ref)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_all(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    sin_theta: np.ndarray,
) -> dict[str, float]:
    """
    Compute all physics-aware metrics and the legacy linear metrics.

    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays — true and predicted power patterns
    sin_theta      : (n_points,) float array — sin(θ) area weights from basis cache

    Returns
    -------
    Dict of scalar metric values ready for MLflow logging.
    """
    pt = p_true.astype(np.float32)
    pp = p_pred.astype(np.float32)

    # Legacy linear metrics (kept for continuity).
    diff = pt - pp
    p_mse = float(np.mean(diff ** 2))
    p_mae = float(np.mean(np.abs(diff)))
    num = np.sqrt(np.sum(diff ** 2, axis=1))
    den = np.sqrt(np.sum(pt ** 2, axis=1)) + 1e-12
    p_rel_l2_mean = float(np.mean(num / den))

    return {
        # Legacy
        "p_mse": p_mse,
        "p_mae": p_mae,
        "p_rel_l2_mean": p_rel_l2_mean,
        # Amplitude of total power (same definition as physics loss amplitude term)
        "total_power_log_mse": total_power_log_mse(pt, pp, sin_theta),
        # Area-weighted (pole-corrected)
        "weighted_mse": weighted_mse(pt, pp, sin_theta),
        "weighted_rel_l2": weighted_rel_l2(pt, pp, sin_theta),
        # dB-scale
        "db_rmse": db_rmse(pt, pp),
        # Beam pointing
        "beam_pointing_error_deg": beam_pointing_error_deg(pt, pp),
        # Gradient / derivative
        "gradient_mae": gradient_mae(pt, pp),
        # Spatial skill score
        "fss": fss(pt, pp),
    }
