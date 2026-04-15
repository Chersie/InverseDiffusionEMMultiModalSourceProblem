"""
Field visualization utilities for the multipole ML pipeline.

All functions accept numpy arrays and return matplotlib Figure objects or save
to disk; they never display interactively (no plt.show()) so they are safe to
call in background training scripts.

Grid convention used everywhere:
    - n_points = 360 * 179 = 64 440
    - Flat index order: phi runs in the outer loop (0..359°), theta in the
      inner loop (1..179°), matching build_dataset row order.
    - 2-D shape after reshape: (n_phi=360, n_theta=179)
    - imshow: theta on x-axis (horizontal), phi on y-axis (vertical)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in training scripts
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    _MPL_AVAILABLE = False

_N_PHI = 360
_N_THETA = 179
_THETA_TICKS = [1, 45, 90, 135, 179]
_PHI_TICKS = [0, 90, 180, 270, 359]


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for field visualization.")


def _to_2d(flat: np.ndarray) -> np.ndarray:
    """Reshape (n_points,) → (n_phi, n_theta) = (360, 179)."""
    return flat.reshape(_N_PHI, _N_THETA)


def _imshow_field(
    ax: "Axes",
    data_2d: np.ndarray,
    *,
    title: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Show a 2-D field (n_phi, n_theta) as a heatmap.

    The image is an "unrolled" sphere:
      x-axis = polar angle θ   (1..179°, tip-to-tip)
      y-axis = azimuth angle φ (0..359°, around the equator)
    Note: θ and φ here are ANGULAR COORDINATES, not field components.
    Field components E_θ and E_φ are named in the subplot title.

    If vmin/vmax are omitted, each panel autoscales independently (can hide
    large amplitude errors between true vs predicted when shapes are similar).
    """
    im = ax.imshow(
        data_2d,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        extent=[1, 179, 359, 0],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("polar angle θ (deg)")
    ax.set_ylabel("azimuth angle φ (deg)")
    ax.set_xticks(_THETA_TICKS)
    ax.set_yticks(_PHI_TICKS)
    if title:
        ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)


def plot_power_map(
    power_flat: np.ndarray,
    *,
    title: str = "P_UT",
    ax: "Axes | None" = None,
    save_path: "Path | None" = None,
) -> "Figure":
    """
    Plot a flattened power pattern (n_points,) as a 2-D heatmap.

    Parameters
    ----------
    power_flat : (n_points,) float array — P_UT = |E_θ|² + |E_φ|².
    title      : subplot title.
    ax         : existing Axes to draw into; creates a new Figure if None.
    save_path  : if given, save the Figure to this path.

    Returns
    -------
    Figure
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    _imshow_field(ax, _to_2d(power_flat.astype(np.float32)), title=title)
    if standalone:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
    return fig


def _reconstruct_field(
    a_e: np.ndarray,
    a_m: np.ndarray,
    basis: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct E_theta and E_phi from packed coefficients and basis.

    Parameters
    ----------
    a_e, a_m : (n_modes,) complex arrays.
    basis    : dict with keys e_theta, e_phi, m_theta, m_phi each (n_modes, n_points).

    Returns
    -------
    e_theta_flat, e_phi_flat : (n_points,) complex64.
    """
    a_e = a_e.astype(np.complex64)
    a_m = a_m.astype(np.complex64)
    e_theta = (a_e @ basis["e_theta"] + a_m @ basis["m_theta"]).astype(np.complex64)
    e_phi = (a_e @ basis["e_phi"] + a_m @ basis["m_phi"]).astype(np.complex64)
    return e_theta, e_phi


def plot_field_components(
    a_e: np.ndarray,
    a_m: np.ndarray,
    basis: dict[str, np.ndarray],
    *,
    title: str = "",
    axes: "tuple[Axes, Axes] | None" = None,
    save_path: "Path | None" = None,
) -> "Figure":
    """
    Reconstruct a field from coefficients and plot |E_θ| and |E_φ| heatmaps.

    Parameters
    ----------
    a_e, a_m : (n_modes,) complex — electric and magnetic coefficients.
    basis    : loaded basis dict from load_or_build_basis.
    title    : prefix appended to subplot titles.
    axes     : pair of existing Axes; creates a new Figure if None.
    save_path: if given, save the Figure.
    """
    _require_mpl()
    standalone = axes is None
    if standalone:
        fig, (ax_theta, ax_phi) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        ax_theta, ax_phi = axes
        fig = ax_theta.get_figure()

    e_theta_flat, e_phi_flat = _reconstruct_field(a_e, a_m, basis)
    prefix = f"{title} — " if title else ""
    _imshow_field(ax_theta, _to_2d(np.abs(e_theta_flat)), title=f"{prefix}|E_θ|")
    _imshow_field(ax_phi, _to_2d(np.abs(e_phi_flat)), title=f"{prefix}|E_φ|")

    if standalone:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
    return fig


def plot_comparison(
    p_true: np.ndarray,
    p_hat: np.ndarray,
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_pred: np.ndarray,
    a_m_pred: np.ndarray,
    basis: dict[str, np.ndarray],
    *,
    suptitle: str = "",
    save_path: "Path | None" = None,
) -> "Figure":
    """
    2×3 comparison figure.

    Top row    (true):      [ P_UT ]  [ |E_θ| from true coeffs ]  [ |E_φ| from true coeffs ]
    Bottom row (predicted): [ P^   ]  [ |E_θ^|                 ]  [ |E_φ^|                 ]

    Parameters
    ----------
    p_true, p_hat         : (n_points,) float — true and predicted power patterns.
    a_e_true, a_m_true    : (n_modes,) complex — true generating coefficients.
    a_e_pred, a_m_pred    : (n_modes,) complex — predicted coefficients.
    basis                 : loaded basis dict.
    suptitle              : figure-level title.
    save_path             : if given, save the Figure.
    """
    _require_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    e_theta_true, e_phi_true = _reconstruct_field(a_e_true, a_m_true, basis)
    e_theta_pred, e_phi_pred = _reconstruct_field(a_e_pred, a_m_pred, basis)
    abs_et_t = np.abs(e_theta_true).astype(np.float32)
    abs_ep_t = np.abs(e_phi_true).astype(np.float32)
    abs_et_p = np.abs(e_theta_pred).astype(np.float32)
    abs_ep_p = np.abs(e_phi_pred).astype(np.float32)

    # Robust color scaling to prevent one pattern from dominating the color range  
    p_t = p_true.astype(np.float32)
    p_p = p_hat.astype(np.float32)
    
    # Use percentile-based scaling to avoid outliers crushing dynamic range
    all_power = np.concatenate([p_t.flatten(), p_p.flatten()])
    p_lo = float(np.percentile(all_power, 2))   # 2nd percentile
    p_hi = float(np.percentile(all_power, 98))  # 98th percentile
    
    # Ensure reasonable minimum dynamic range for true pattern visibility
    true_range = p_t.max() - p_t.min()
    if (p_hi - p_lo) > 5 * true_range:  # If combined range is much larger than true range
        # Use separate scaling for power column to preserve true pattern structure
        p_lo, p_hi = None, None  # Will auto-scale each panel independently

    # Top row: true
    _imshow_field(
        axes[0, 0], _to_2d(p_t), title="P_UT (true)", cmap="viridis", vmin=p_lo, vmax=p_hi
    )
    _imshow_field(
        axes[0, 1],
        _to_2d(abs_et_t),
        title="|E_θ| (true)",
        cmap="plasma",
        vmin=float(min(abs_et_t.min(), abs_et_p.min())),
        vmax=float(max(abs_et_t.max(), abs_et_p.max())),
    )
    _imshow_field(
        axes[0, 2],
        _to_2d(abs_ep_t),
        title="|E_φ| (true)",
        cmap="plasma",
        vmin=float(min(abs_ep_t.min(), abs_ep_p.min())),
        vmax=float(max(abs_ep_t.max(), abs_ep_p.max())),
    )

    # Bottom row: predicted (same vmin/vmax as top in each column)
    _imshow_field(
        axes[1, 0], _to_2d(p_p), title="P^ (predicted)", cmap="viridis", vmin=p_lo, vmax=p_hi
    )
    _imshow_field(
        axes[1, 1],
        _to_2d(abs_et_p),
        title="|E_θ^| (predicted)",
        cmap="plasma",
        vmin=float(min(abs_et_t.min(), abs_et_p.min())),
        vmax=float(max(abs_et_t.max(), abs_et_p.max())),
    )
    _imshow_field(
        axes[1, 2],
        _to_2d(abs_ep_p),
        title="|E_φ^| (predicted)",
        cmap="plasma",
        vmin=float(min(abs_ep_t.min(), abs_ep_p.min())),
        vmax=float(max(abs_ep_t.max(), abs_ep_p.max())),
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    return fig


def plot_normalized_difference_map(
    p_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    title: str = "Normalized Difference",
    ax: "Axes | None" = None,
    save_path: "Path | None" = None,
    epsilon: float = 1e-6,
) -> "Figure":
    """
    Plot normalized difference map: (p_pred - p_true) / |p_true|.
    
    Parameters
    ----------
    p_true, p_pred : (n_points,) float arrays
        True and predicted power patterns
    title : str
        Plot title
    ax : Axes, optional
        Existing axes to draw into; creates new figure if None
    save_path : Path, optional
        If given, save figure to this path
    epsilon : float
        Small value to prevent division by zero
        
    Returns
    -------
    Figure
    """
    _require_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    
    # Compute normalized difference
    p_t = p_true.astype(np.float64)
    p_p = p_pred.astype(np.float64)
    
    # Avoid division by very small values
    denominator = np.abs(p_t) + epsilon
    norm_diff = (p_p - p_t) / denominator
    
    # Use symmetric color limits
    vmax = np.percentile(np.abs(norm_diff), 95)  # Robust to outliers
    
    _imshow_field(
        ax, 
        _to_2d(norm_diff.astype(np.float32)), 
        title=title, 
        cmap="RdBu_r",  # Diverging colormap: red=positive, blue=negative
        vmin=-vmax, 
        vmax=vmax
    )
    
    if standalone:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100)
            plt.close(fig)
    
    return fig


def plot_comparison_with_difference(
    p_true: np.ndarray,
    p_hat: np.ndarray,
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_pred: np.ndarray,
    a_m_pred: np.ndarray,
    basis: dict[str, np.ndarray],
    *,
    suptitle: str = "",
    save_path: "Path | None" = None,
) -> "Figure":
    """
    3×3 comparison figure with normalized difference maps.
    
    Top row    (true):      [ P_UT ]  [ |E_θ| from true coeffs ]  [ |E_φ| from true coeffs ]
    Middle row (predicted): [ P^   ]  [ |E_θ^|                 ]  [ |E_φ^|                 ]
    Bottom row (difference):[ ΔP/P  ]  [ Δ|E_θ|/|E_θ|          ]  [ Δ|E_φ|/|E_φ|           ]
    
    Parameters
    ----------
    p_true, p_hat         : (n_points,) float — true and predicted power patterns.
    a_e_true, a_m_true    : (n_modes,) complex — true generating coefficients.
    a_e_pred, a_m_pred    : (n_modes,) complex — predicted coefficients.
    basis                 : loaded basis dict.
    suptitle              : figure-level title.
    save_path             : if given, save the Figure.
    """
    _require_mpl()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    e_theta_true, e_phi_true = _reconstruct_field(a_e_true, a_m_true, basis)
    e_theta_pred, e_phi_pred = _reconstruct_field(a_e_pred, a_m_pred, basis)
    abs_et_t = np.abs(e_theta_true).astype(np.float32)
    abs_ep_t = np.abs(e_phi_true).astype(np.float32)
    abs_et_p = np.abs(e_theta_pred).astype(np.float32)
    abs_ep_p = np.abs(e_phi_pred).astype(np.float32)

    # Robust color scaling to prevent one pattern from dominating the color range
    p_t = p_true.astype(np.float32)
    p_p = p_hat.astype(np.float32)
    
    # Use percentile-based scaling to avoid outliers crushing dynamic range
    all_power = np.concatenate([p_t.flatten(), p_p.flatten()])
    p_lo = float(np.percentile(all_power, 2))   # 2nd percentile
    p_hi = float(np.percentile(all_power, 98))  # 98th percentile
    
    # Ensure reasonable minimum dynamic range for true pattern visibility
    true_range = p_t.max() - p_t.min()
    if (p_hi - p_lo) > 5 * true_range:  # If combined range is much larger than true range
        # Use separate scaling for power column to preserve true pattern structure
        p_lo, p_hi = None, None  # Will auto-scale each panel independently

    # Top row: true
    _imshow_field(
        axes[0, 0], _to_2d(p_t), title="P_UT (true)", cmap="viridis", vmin=p_lo, vmax=p_hi
    )
    _imshow_field(
        axes[0, 1],
        _to_2d(abs_et_t),
        title="|E_θ| (true)",
        cmap="plasma",
        vmin=float(min(abs_et_t.min(), abs_et_p.min())),
        vmax=float(max(abs_et_t.max(), abs_et_p.max())),
    )
    _imshow_field(
        axes[0, 2],
        _to_2d(abs_ep_t),
        title="|E_φ| (true)",
        cmap="plasma",
        vmin=float(min(abs_ep_t.min(), abs_ep_p.min())),
        vmax=float(max(abs_ep_t.max(), abs_ep_p.max())),
    )

    # Middle row: predicted (same vmin/vmax as top in each column)
    _imshow_field(
        axes[1, 0], _to_2d(p_p), title="P^ (predicted)", cmap="viridis", vmin=p_lo, vmax=p_hi
    )
    _imshow_field(
        axes[1, 1],
        _to_2d(abs_et_p),
        title="|E_θ^| (predicted)",
        cmap="plasma",
        vmin=float(min(abs_et_t.min(), abs_et_p.min())),
        vmax=float(max(abs_et_t.max(), abs_et_p.max())),
    )
    _imshow_field(
        axes[1, 2],
        _to_2d(abs_ep_p),
        title="|E_φ^| (predicted)",
        cmap="plasma",
        vmin=float(min(abs_ep_t.min(), abs_ep_p.min())),
        vmax=float(max(abs_ep_t.max(), abs_ep_p.max())),
    )
    
    # Bottom row: normalized differences
    plot_normalized_difference_map(
        p_t, p_p, 
        title="ΔP/|P| (normalized difference)", 
        ax=axes[2, 0]
    )
    plot_normalized_difference_map(
        abs_et_t, abs_et_p,
        title="Δ|E_θ|/|E_θ| (normalized difference)", 
        ax=axes[2, 1]
    )
    plot_normalized_difference_map(
        abs_ep_t, abs_ep_p,
        title="Δ|E_φ|/|E_φ| (normalized difference)", 
        ax=axes[2, 2]
    )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    return fig


def plot_coeff_bars(
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_pred: np.ndarray,
    a_m_pred: np.ndarray,
    *,
    suptitle: str = "",
    save_path: "Path | None" = None,
) -> "Figure":
    """
    Bar chart comparing true vs predicted coefficient magnitudes.

    Top panel   : |a_E| per mode index.
    Bottom panel: |a_M| per mode index.

    Parameters
    ----------
    a_e_true, a_m_true : (n_modes,) complex — true coefficients.
    a_e_pred, a_m_pred : (n_modes,) complex — predicted coefficients.
    suptitle           : figure-level title.
    save_path          : if given, save the Figure.
    """
    _require_mpl()
    n_modes = len(a_e_true)
    x = np.arange(n_modes)
    width = 0.4

    fig, (ax_e, ax_m) = plt.subplots(2, 1, figsize=(max(8, n_modes // 4), 6))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    ax_e.bar(x - width / 2, np.abs(a_e_true), width, label="true", alpha=0.8)
    ax_e.bar(x + width / 2, np.abs(a_e_pred), width, label="predicted", alpha=0.8)
    ax_e.set_xlabel("mode index k")
    ax_e.set_ylabel("|a_E[k]|")
    ax_e.set_title("Electric-type coefficients")
    ax_e.legend(fontsize=8)

    ax_m.bar(x - width / 2, np.abs(a_m_true), width, label="true", alpha=0.8)
    ax_m.bar(x + width / 2, np.abs(a_m_pred), width, label="predicted", alpha=0.8)
    ax_m.set_xlabel("mode index k")
    ax_m.set_ylabel("|a_M[k]|")
    ax_m.set_title("Magnetic-type coefficients")
    ax_m.legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    return fig


def save_sample_preview(
    sample_idx: int,
    p_true: np.ndarray,
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_ref: np.ndarray,
    a_m_ref: np.ndarray,
    basis: dict[str, np.ndarray],
    out_dir: Path,
    *,
    ref_label: str = "proj",
    split_label: str = "",
    include_difference_maps: bool = True,
) -> list[Path]:
    """
    Save a comparison figure and a coeff-bar figure for one sample.

    Parameters
    ----------
    ref_label   : describes what the bottom row is — "proj" for dataset previews,
                  "model" for training validation.
    split_label : which data split this sample belongs to, e.g. "val" or "test".
                  Shown in the title so it is always clear which samples are plotted.

    Returns list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p_ref = (np.abs(a_e_ref @ basis["e_theta"] + a_m_ref @ basis["m_theta"]) ** 2
             + np.abs(a_e_ref @ basis["e_phi"] + a_m_ref @ basis["m_phi"]) ** 2).astype(np.float32)

    split_suffix = f" [{split_label}]" if split_label else ""
    comp_path = out_dir / f"sample_{sample_idx:04d}_fields.png"
    bars_path = out_dir / f"sample_{sample_idx:04d}_coeffs.png"

    if include_difference_maps:
        plot_comparison_with_difference(
            p_true, p_ref,
            a_e_true, a_m_true,
            a_e_ref, a_m_ref,
            basis,
            suptitle=f"Sample {sample_idx}{split_suffix} — true vs {ref_label} with differences",
            save_path=comp_path,
        )
    else:
        plot_comparison(
            p_true, p_ref,
            a_e_true, a_m_true,
            a_e_ref, a_m_ref,
            basis,
            suptitle=f"Sample {sample_idx}{split_suffix} — true vs {ref_label}",
            save_path=comp_path,
        )
    plot_coeff_bars(
        a_e_true, a_m_true,
        a_e_ref, a_m_ref,
        suptitle=f"Sample {sample_idx}{split_suffix} — coefficient magnitudes",
        save_path=bars_path,
    )
    return [comp_path, bars_path]
