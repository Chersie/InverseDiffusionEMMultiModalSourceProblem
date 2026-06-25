"""Field-space metrics."""

from __future__ import annotations

import numpy as np

from mpinv.core.area_weights import normalised_area_weights
from mpinv.core.grid import GRID_DEFAULT, GridSpec


def weighted_mse_P(P_pred: np.ndarray, P_true: np.ndarray, grid: GridSpec = GRID_DEFAULT) -> float:
    """``sin θ``-weighted MSE between predicted and true power patterns."""
    w = normalised_area_weights(grid)
    return float((((P_pred - P_true) ** 2) * w).mean())


def weighted_nrmse_P(
    P_pred: np.ndarray, P_true: np.ndarray, grid: GridSpec = GRID_DEFAULT
) -> float:
    """Normalised RMSE of P relative to ``||P_true||_w``."""
    w = normalised_area_weights(grid)
    num = float((((P_pred - P_true) ** 2) * w).sum())
    den = float(((P_true**2) * w).sum())
    return float(np.sqrt(num / max(den, 1e-12)))


def per_sample_weighted_nrmse_P(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    grid: GridSpec = GRID_DEFAULT,
    eps: float = 1e-30,
    constant_sentinel: float = float("nan"),
) -> np.ndarray:
    """Per-sample sin-theta-weighted normalised RMSE on power patterns.

    For each sample ``i``::

        NRMSE_w(i) = sqrt( sum_pix w * (P_pred[i] - P_true[i])^2
                         / sum_pix w * P_true[i]^2 )

    Returns a ``(B,)`` array. Lower is better; ``0`` is perfect, ``1`` means
    residuals on the order of the target itself, ``> 1`` is worse than
    predicting zero. Samples whose target is effectively zero (so the
    denominator collapses) are returned as ``constant_sentinel`` (``NaN`` by
    default; typical caller is the violin plot, which drops NaN).
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    if P_pred.ndim != 3:
        raise ValueError(f"expected (B, n_theta, n_phi); got {P_pred.shape}")
    w = normalised_area_weights(grid).astype(np.float64)
    diff = P_pred.astype(np.float64) - P_true.astype(np.float64)
    num = (diff * diff * w).sum(axis=(-2, -1))
    den = (P_true.astype(np.float64) ** 2 * w).sum(axis=(-2, -1))
    out = np.full_like(num, constant_sentinel, dtype=np.float64)
    valid = den >= eps
    out[valid] = np.sqrt(num[valid] / den[valid])
    return out


def per_sample_weighted_r2_P(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    grid: GridSpec = GRID_DEFAULT,
    eps: float = 1e-30,
    constant_sentinel: float = float("-inf"),
) -> np.ndarray:
    """Per-sample sin-θ-weighted coefficient of determination on power patterns.

    For each sample ``i`` (axis 0)::

        R²_w(i) = 1 − SS_res_w / SS_tot_w
        SS_res_w = Σ_pixels w_pixel · (P_pred[i] − P_true[i])²
        SS_tot_w = Σ_pixels w_pixel · (P_true[i] − ⟨P_true[i]⟩_w)²
        ⟨P_true[i]⟩_w = (Σ w · P_true[i]) / (Σ w)

    where ``w`` is the project's sin-θ-normalised area weight on the
    full ``(n_theta, n_phi)`` grid.

    This matches the definition sklearn's ``r2_score`` would give if you fed
    each sample's flattened pattern as a separate y-vector and asked for
    per-sample R² (sklearn's ``multioutput='raw_values'`` on transposed
    input). The "weighted" qualifier just changes the inner products from
    plain sums to sin-θ-weighted sums.

    Parameters
    ----------
    P_pred, P_true : np.ndarray
        Real arrays of shape ``(B, n_theta, n_phi)``. Shapes must match.
    grid : GridSpec
        Grid spec; controls the area weights.
    eps : float
        Numerical floor for ``SS_tot_w`` so we don't divide by exact zero.
    constant_sentinel : float
        Returned in entries where ``SS_tot_w < eps`` (i.e. the target P is
        effectively constant across the sphere — a "silent" antenna). The
        default is ``-inf`` so that callers that sort by R² treat such samples
        as the worst possible fit. Use ``np.nan`` if you want to filter them
        out instead.

    Returns
    -------
    np.ndarray
        Array of shape ``(B,)`` with one R² per sample. Higher is better;
        ``1.0`` is perfect, ``0.0`` is "predict each sample's own weighted
        mean", negative values are worse than that baseline.
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    if P_pred.ndim != 3:
        raise ValueError(f"expected (B, n_theta, n_phi); got {P_pred.shape}")
    w = normalised_area_weights(grid).astype(np.float64)  # (n_theta, n_phi)
    w_sum = float(w.sum())
    diff = (P_pred.astype(np.float64) - P_true.astype(np.float64))
    ss_res = (diff * diff * w).sum(axis=(-2, -1))
    P_true_64 = P_true.astype(np.float64)
    P_mean_w = (P_true_64 * w).sum(axis=(-2, -1)) / w_sum  # (B,)
    centred = P_true_64 - P_mean_w[:, None, None]
    ss_tot = (centred * centred * w).sum(axis=(-2, -1))
    out = np.full_like(ss_res, constant_sentinel, dtype=np.float64)
    valid = ss_tot >= eps
    out[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return out


def weighted_r2_P(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    grid: GridSpec = GRID_DEFAULT,
) -> float:
    """Batch-aggregated sin-θ-weighted R² over power patterns.

    Mean of ``per_sample_weighted_r2_P`` over the batch axis (sklearn's
    ``multioutput='uniform_average'`` style), with samples whose target is
    effectively constant (R² undefined) excluded from the mean. If every
    sample is degenerate the function returns ``nan``.
    """
    per_sample = per_sample_weighted_r2_P(P_pred, P_true, grid=grid)
    finite = per_sample[np.isfinite(per_sample)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def per_theta_band_error(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    n_bands: int = 9,
) -> np.ndarray:
    """Mean absolute error in each of ``n_bands`` equal-θ bands.

    Returns a 1-D array of length ``n_bands``.
    """
    n_theta = P_pred.shape[-2]
    edges = np.linspace(0, n_theta, n_bands + 1, dtype=int)
    out = np.zeros(n_bands)
    for b in range(n_bands):
        s, e = edges[b], edges[b + 1]
        out[b] = float(np.abs(P_pred[..., s:e, :] - P_true[..., s:e, :]).mean())
    return out


# -- rank- and bin-based metrics --------------------------------------------
#
# These are dataset-level summaries of how well the model preserves the
# *rank order* of pixels in P, rather than absolute amplitudes. They are
# the natural evaluation companions of the soft rank-bin training loss in
# :mod:`mpinv.losses.rank_bin` (which uses the same ``n_bins = 2*l_max+1``
# convention). All these metrics use *hard* sorts and bin assignments — no
# sigmoid soft-binning — because evaluation does not need to be
# differentiable.


def _hard_bin_indices_per_sample(P_flat: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-sample hard quantile bin index for each pixel, in ``[0, n_bins-1]``.

    ``P_flat`` has shape ``(B, n_pixels)``. The returned int64 array has the
    same shape; entry ``[i, j]`` is the rank-bin of pixel ``j`` of sample
    ``i`` under sample-``i``'s own ranking. Ties (rare for float P) follow
    NumPy's stable argsort, so equal values land in adjacent bins
    deterministically.
    """
    if P_flat.ndim != 2:
        raise ValueError(f"expected (B, n_pixels); got {P_flat.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    B, N = P_flat.shape
    # Double-argsort gives the rank of each entry per sample.
    ranks = np.argsort(np.argsort(P_flat, axis=-1), axis=-1)
    return np.minimum(((ranks * n_bins) // N).astype(np.int64), n_bins - 1)


def per_sample_spearman_rho_P(
    P_pred: np.ndarray, P_true: np.ndarray
) -> np.ndarray:
    """Per-sample Spearman rank correlation between predicted and true P.

    Computed by Pearson-correlating the per-sample double-argsort ranks.
    Returns a ``(B,)`` array in ``[-1, 1]``: ``+1`` perfect rank agreement,
    ``-1`` reversed, ``0`` no rank correlation. Returns ``NaN`` for samples
    where either ``P_pred`` or ``P_true`` is constant (because the rank of
    a constant vector is arbitrary — argsort would still produce a
    deterministic ordering, but it carries no real information so the
    correlation against it would be misleading).
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    if P_pred.ndim != 3:
        raise ValueError(f"expected (B, n_theta, n_phi); got {P_pred.shape}")
    B = P_pred.shape[0]
    f_pred = P_pred.reshape(B, -1).astype(np.float64)
    f_true = P_true.reshape(B, -1).astype(np.float64)
    # Detect samples whose pixel values are constant (or numerically so) —
    # ``argsort`` would still give them an arbitrary ordering but Spearman is
    # undefined in that case.
    eps = 1e-12
    const_pred = (f_pred.std(axis=1) <= eps)
    const_true = (f_true.std(axis=1) <= eps)
    degenerate = const_pred | const_true

    rk_pred = np.argsort(np.argsort(f_pred, axis=1), axis=1).astype(np.float64)
    rk_true = np.argsort(np.argsort(f_true, axis=1), axis=1).astype(np.float64)
    rk_pred -= rk_pred.mean(axis=1, keepdims=True)
    rk_true -= rk_true.mean(axis=1, keepdims=True)
    num = (rk_pred * rk_true).sum(axis=1)
    den_p = (rk_pred * rk_pred).sum(axis=1)
    den_t = (rk_true * rk_true).sum(axis=1)
    den = np.sqrt(den_p * den_t)
    rho = np.full(B, np.nan, dtype=np.float64)
    valid = (den > 1e-30) & ~degenerate
    rho[valid] = num[valid] / den[valid]
    return rho


def spearman_rho_P(P_pred: np.ndarray, P_true: np.ndarray) -> float:
    """Mean Spearman ρ over the batch (NaN samples — degenerate ranks — are
    excluded from the mean). Single-number summary of dataset-level rank
    agreement; ``+1`` is perfect, ``0`` is no relationship, ``-1`` is anti-
    correlated. Returns ``nan`` if every sample is degenerate.
    """
    rho = per_sample_spearman_rho_P(P_pred, P_true)
    finite = rho[np.isfinite(rho)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def per_sample_bin_accuracy_P(
    P_pred: np.ndarray, P_true: np.ndarray, n_bins: int
) -> np.ndarray:
    """Fraction of pixels assigned to the same hard rank-bin per sample."""
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    B = P_pred.shape[0]
    bp = _hard_bin_indices_per_sample(P_pred.reshape(B, -1), n_bins)
    bt = _hard_bin_indices_per_sample(P_true.reshape(B, -1), n_bins)
    return (bp == bt).mean(axis=1).astype(np.float64)


def bin_accuracy_P(
    P_pred: np.ndarray, P_true: np.ndarray, n_bins: int
) -> float:
    """Mean per-pixel exact-bin agreement across the dataset.

    The most direct evaluation companion of
    :func:`mpinv.losses.rank_bin.rank_bin_mse`: under hard binning, the
    soft loss collapses onto ``E[(bin_pred - bin_true)²]`` and the metric
    here is ``E[bin_pred == bin_true]`` over all pixels and all samples.
    """
    return float(per_sample_bin_accuracy_P(P_pred, P_true, n_bins).mean())


def per_sample_bin_within_k_accuracy_P(
    P_pred: np.ndarray, P_true: np.ndarray, n_bins: int, k: int = 1
) -> np.ndarray:
    """Per-sample fraction of pixels whose pred bin is within ``k`` of true."""
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    B = P_pred.shape[0]
    bp = _hard_bin_indices_per_sample(P_pred.reshape(B, -1), n_bins)
    bt = _hard_bin_indices_per_sample(P_true.reshape(B, -1), n_bins)
    return (np.abs(bp - bt) <= k).mean(axis=1).astype(np.float64)


def bin_within_k_accuracy_P(
    P_pred: np.ndarray, P_true: np.ndarray, n_bins: int, k: int = 1
) -> float:
    """Mean within-``k``-bins agreement across the dataset (default ``k=1``).

    Less brittle than exact-bin agreement: ``k=1`` says "off by one bin is
    acceptable", which is a more honest target when bins are close in P.
    """
    return float(per_sample_bin_within_k_accuracy_P(P_pred, P_true, n_bins, k).mean())


def hard_rank_bin_mse_P(
    P_pred: np.ndarray, P_true: np.ndarray, n_bins: int
) -> float:
    """Hard-binned companion to :func:`mpinv.losses.rank_bin.rank_bin_mse`.

    Computes ``E[(bin_pred - bin_true)²]`` where bins are exact quantile
    indices (no sigmoid). Use this as the *evaluation* metric tied to the
    rank-bin training loss; smaller is better.
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    B = P_pred.shape[0]
    bp = _hard_bin_indices_per_sample(P_pred.reshape(B, -1), n_bins).astype(np.float64)
    bt = _hard_bin_indices_per_sample(P_true.reshape(B, -1), n_bins).astype(np.float64)
    return float(((bp - bt) ** 2).mean())
