"""
Polarization-aware evaluation metrics for electromagnetic field prediction.

These metrics analyze E_θ and E_φ components separately, with logarithmic power weighting
as requested by the advisor: points with higher total power P get proportionally more
attention, reflecting real-world priorities where clean polarization in the main beam
matters more than errors in low-power regions.

Logarithmic weighting: P increases by 2x → ~3 additional weight points.
"""
from __future__ import annotations

import numpy as np

# Grid constants (matching other metrics)
_N_PHI = 360
_N_THETA = 179
_DEG2RAD = np.pi / 180.0


def _to_2d(flat: np.ndarray) -> np.ndarray:
    """Reshape (n_points,) → (n_phi=360, n_theta=179)."""
    return flat.reshape(_N_PHI, _N_THETA)


def power_weighted_polarization_error(
    e_theta_true: np.ndarray,
    e_phi_true: np.ndarray,
    e_theta_pred: np.ndarray,
    e_phi_pred: np.ndarray,
    sin_theta: np.ndarray,
    log_power_weight: float = 1.0,
    min_power_db: float = -40.0,
) -> dict[str, float]:
    """
    Power-weighted polarization component errors with logarithmic weighting.
    
    The total power P = |E_θ|² + |E_φ|² determines the importance of each point:
    higher power regions get more weight in the error calculation.
    
    Weighting formula: w_i = α × log₂(P_i/P_min + 1) × sin(θ_i)
    where P_min is derived from min_power_db relative to the peak.
    
    Parameters
    ----------
    e_theta_true, e_phi_true : (N, n_points) float arrays
        True |E_θ|² and |E_φ|² power components
    e_theta_pred, e_phi_pred : (N, n_points) float arrays  
        Predicted |E_θ|² and |E_φ|² power components
    sin_theta : (n_points,) float array
        Area weighting factors (sin of polar angle)
    log_power_weight : float
        Scaling factor for logarithmic power weighting
    min_power_db : float
        Minimum power level in dB for logarithmic scaling floor
        
    Returns
    -------
    dict
        Metrics including component MSEs, weighted errors, and polarization balance
    """
    # Convert to float64 for numerical stability
    et_t = e_theta_true.astype(np.float64)
    ep_t = e_phi_true.astype(np.float64)
    et_p = e_theta_pred.astype(np.float64)
    ep_p = e_phi_pred.astype(np.float64)
    
    # Total power per point
    p_true = et_t + ep_t  # (N, n_points)
    p_pred = et_p + ep_p
    
    # Area weighting (standard spherical integration)
    w_area = sin_theta.astype(np.float64)  # (n_points,)
    w_area = w_area / w_area.sum()  # Normalize to sum to 1
    
    # Logarithmic power weighting per sample
    # P increases by 2x → log₂(2) = 1 additional point → with scaling factor
    results = {}
    
    for i in range(len(p_true)):
        # Power weighting for this sample
        p_sample = p_true[i]  # (n_points,)
        p_peak = p_sample.max()
        p_min = p_peak * (10.0 ** (min_power_db / 10.0))  # Floor in linear scale
        
        # Logarithmic power weights: w_power = log₂(P/P_min + 1)
        # This gives 0 weight at P=P_min, 1 at P=2*P_min, 2 at P=4*P_min, etc.
        w_power = np.log2(p_sample / p_min + 1.0)
        w_power = np.maximum(w_power, 0.0)  # No negative weights
        
        # Combined weighting: area × logarithmic power
        w_combined = w_area * w_power * log_power_weight
        w_sum = w_combined.sum() + 1e-12
        w_normalized = w_combined / w_sum  # Normalize to sum to 1
        
        # Component errors for this sample
        et_error = et_p[i] - et_t[i]  # (n_points,)
        ep_error = ep_p[i] - ep_t[i]
        
        # Power-weighted component MSEs
        et_mse_weighted = (et_error ** 2 * w_normalized).sum()
        ep_mse_weighted = (ep_error ** 2 * w_normalized).sum()
        
        # Store per-sample results
        if i == 0:  # Initialize accumulators
            results.update({
                'theta_mse_power_weighted': et_mse_weighted,
                'phi_mse_power_weighted': ep_mse_weighted,
                'total_mse_power_weighted': et_mse_weighted + ep_mse_weighted,
            })
        else:  # Accumulate
            results['theta_mse_power_weighted'] += et_mse_weighted
            results['phi_mse_power_weighted'] += ep_mse_weighted
            results['total_mse_power_weighted'] += et_mse_weighted + ep_mse_weighted
    
    # Average over samples
    n_samples = len(p_true)
    for key in ['theta_mse_power_weighted', 'phi_mse_power_weighted', 'total_mse_power_weighted']:
        results[key] /= n_samples
    
    # Additional polarization balance metrics
    results.update(_polarization_balance_metrics(et_t, ep_t, et_p, ep_p, w_area))
    
    return {k: float(v) for k, v in results.items()}


def _polarization_balance_metrics(
    e_theta_true: np.ndarray,
    e_phi_true: np.ndarray, 
    e_theta_pred: np.ndarray,
    e_phi_pred: np.ndarray,
    w_area: np.ndarray,
) -> dict[str, float]:
    """
    Compute polarization balance and ratio metrics.
    
    Analyzes how well the predicted model maintains the correct balance
    between E_θ and E_φ components relative to the true field.
    """
    # Total power per sample
    p_true = e_theta_true + e_phi_true  # (N, n_points)
    p_pred = e_theta_pred + e_phi_pred
    
    # Avoid division by zero
    eps = 1e-12
    p_true_safe = p_true + eps
    p_pred_safe = p_pred + eps
    
    # Polarization ratios: θ-component fraction
    theta_frac_true = e_theta_true / p_true_safe  # (N, n_points)
    theta_frac_pred = e_theta_pred / p_pred_safe
    
    # Area-weighted errors in polarization fraction
    frac_error = theta_frac_pred - theta_frac_true  # (N, n_points)
    frac_mse = ((frac_error ** 2 * w_area).sum(axis=1)).mean()  # Weighted over grid, averaged over samples
    
    # Cross-polarization error: how much E_φ appears where we expect E_θ and vice versa
    # Define as the area-weighted correlation between true and predicted polarization patterns
    correlations = []
    for i in range(len(p_true)):
        # Area-weighted correlation coefficient
        theta_t = theta_frac_true[i]
        theta_p = theta_frac_pred[i]
        
        # Weighted means
        mean_t = (theta_t * w_area).sum()
        mean_p = (theta_p * w_area).sum()
        
        # Weighted covariance and variances
        cov = ((theta_t - mean_t) * (theta_p - mean_p) * w_area).sum()
        var_t = ((theta_t - mean_t) ** 2 * w_area).sum()
        var_p = ((theta_p - mean_p) ** 2 * w_area).sum()
        
        # Correlation coefficient
        if var_t > eps and var_p > eps:
            corr = cov / np.sqrt(var_t * var_p)
            correlations.append(corr)
    
    pol_correlation = float(np.mean(correlations)) if correlations else 0.0
    
    return {
        'polarization_fraction_mse': float(frac_mse),
        'polarization_correlation': pol_correlation,
        'polarization_decorrelation': float(1.0 - pol_correlation),  # Error-like metric (0 is perfect)
    }


def directional_polarization_error(
    e_theta_true: np.ndarray,
    e_phi_true: np.ndarray,
    e_theta_pred: np.ndarray, 
    e_phi_pred: np.ndarray,
    sin_theta: np.ndarray,
    beam_threshold_db: float = -3.0,
) -> dict[str, float]:
    """
    Analyze polarization errors specifically in the main beam direction.
    
    This focuses on the regions where polarization purity matters most:
    near the peak of the radiation pattern.
    
    Parameters
    ----------
    e_theta_true, e_phi_true : (N, n_points) float arrays
        True |E_θ|² and |E_φ|² power components  
    e_theta_pred, e_phi_pred : (N, n_points) float arrays
        Predicted |E_θ|² and |E_φ|² power components
    sin_theta : (n_points,) float array
        Area weighting factors
    beam_threshold_db : float
        dB threshold below peak to define "main beam" region
        
    Returns
    -------
    dict  
        Main beam polarization metrics
    """
    results = {}
    w_area = sin_theta.astype(np.float64) / sin_theta.sum()
    
    beam_errors_theta = []
    beam_errors_phi = []
    beam_fractions_true = []
    beam_fractions_pred = []
    
    for i in range(len(e_theta_true)):
        p_true = e_theta_true[i] + e_phi_true[i]
        p_pred = e_theta_pred[i] + e_phi_pred[i] 
        
        # Define main beam region based on true pattern
        p_peak = p_true.max()
        threshold = p_peak * (10.0 ** (beam_threshold_db / 10.0))
        beam_mask = p_true >= threshold
        
        if not beam_mask.any():
            continue  # Skip samples with no beam region
            
        # Extract beam region data
        et_t_beam = e_theta_true[i][beam_mask]
        ep_t_beam = e_phi_true[i][beam_mask]
        et_p_beam = e_theta_pred[i][beam_mask] 
        ep_p_beam = e_phi_pred[i][beam_mask]
        w_beam = w_area[beam_mask]
        w_beam = w_beam / w_beam.sum()  # Renormalize for beam region
        
        # Component errors in beam
        beam_errors_theta.append(((et_p_beam - et_t_beam) ** 2 * w_beam).sum())
        beam_errors_phi.append(((ep_p_beam - ep_t_beam) ** 2 * w_beam).sum())
        
        # Polarization fractions in beam
        p_t_beam = et_t_beam + ep_t_beam + 1e-12
        p_p_beam = et_p_beam + ep_p_beam + 1e-12
        beam_fractions_true.append((et_t_beam / p_t_beam * w_beam).sum())
        beam_fractions_pred.append((et_p_beam / p_p_beam * w_beam).sum())
    
    if beam_errors_theta:
        results.update({
            'beam_theta_mse': float(np.mean(beam_errors_theta)),
            'beam_phi_mse': float(np.mean(beam_errors_phi)),
            'beam_total_mse': float(np.mean(beam_errors_theta) + np.mean(beam_errors_phi)),
            'beam_theta_fraction_error': float(np.mean(beam_fractions_pred) - np.mean(beam_fractions_true)),
            'beam_theta_fraction_mae': float(np.mean(np.abs(np.array(beam_fractions_pred) - np.array(beam_fractions_true)))),
        })
    else:
        # No valid beam regions found
        results.update({
            'beam_theta_mse': 0.0,
            'beam_phi_mse': 0.0,
            'beam_total_mse': 0.0,
            'beam_theta_fraction_error': 0.0,
            'beam_theta_fraction_mae': 0.0,
        })
    
    return results


def compute_polarization_metrics(
    e_theta_true: np.ndarray,
    e_phi_true: np.ndarray,
    e_theta_pred: np.ndarray,
    e_phi_pred: np.ndarray,
    sin_theta: np.ndarray,
    log_power_weight: float = 1.0,
    min_power_db: float = -40.0,
    beam_threshold_db: float = -3.0,
) -> dict[str, float]:
    """
    Compute all polarization-aware metrics.
    
    This is the main entry point for polarization analysis, combining
    power-weighted errors and directional (main beam) analysis.
    
    Parameters
    ----------
    e_theta_true, e_phi_true : (N, n_points) float arrays
        True |E_θ|² and |E_φ|² power components
    e_theta_pred, e_phi_pred : (N, n_points) float arrays
        Predicted |E_θ|² and |E_φ|² power components 
    sin_theta : (n_points,) float array
        Area weighting factors (sin of polar angle)
    log_power_weight : float
        Scaling for logarithmic power weighting (advisor's requirement)
    min_power_db : float
        Floor for logarithmic power scaling
    beam_threshold_db : float
        Threshold for main beam analysis
        
    Returns
    -------
    dict
        Complete set of polarization metrics
    """
    metrics = {}
    
    # Power-weighted polarization errors
    power_weighted = power_weighted_polarization_error(
        e_theta_true, e_phi_true, e_theta_pred, e_phi_pred, 
        sin_theta, log_power_weight, min_power_db
    )
    metrics.update(power_weighted)
    
    # Directional (main beam) polarization errors
    beam_metrics = directional_polarization_error(
        e_theta_true, e_phi_true, e_theta_pred, e_phi_pred,
        sin_theta, beam_threshold_db
    )
    metrics.update(beam_metrics)
    
    return metrics