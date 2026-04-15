"""
Detailed coefficient comparison and validation for multipole analysis.

This module provides tools to analyze the difference between true generating
coefficients and model-predicted coefficients, with detailed per-mode breakdowns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    _MPL_AVAILABLE = False


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for coefficient visualization.")


def create_coefficient_table(
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_pred: np.ndarray,
    a_m_pred: np.ndarray,
    mode_list: list[tuple[int, int]],
    sample_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Create detailed coefficient comparison table.
    
    Parameters
    ----------
    a_e_true, a_m_true : (N, n_modes) complex arrays
        True electric and magnetic coefficients
    a_e_pred, a_m_pred : (N, n_modes) complex arrays  
        Predicted electric and magnetic coefficients
    mode_list : list of (l, m) tuples
        Mode indices corresponding to coefficient arrays
    sample_indices : array of int, optional
        Sample indices to include. If None, uses all samples.
        
    Returns
    -------
    pd.DataFrame
        Table with columns: sample, l, m, type, true_real, true_imag, pred_real, pred_imag,
        true_mag, pred_mag, mag_error, true_phase, pred_phase, phase_error_deg
    """
    if sample_indices is None:
        sample_indices = np.arange(len(a_e_true))
    
    n_samples = len(sample_indices)
    n_modes = len(mode_list)
    
    # Pre-allocate lists for efficiency
    rows = []
    
    for i, sample_idx in enumerate(sample_indices):
        # Electric coefficients
        for k, (l, m) in enumerate(mode_list):
            true_coeff = a_e_true[sample_idx, k]
            pred_coeff = a_e_pred[sample_idx, k]
            
            true_mag = np.abs(true_coeff)
            pred_mag = np.abs(pred_coeff)
            true_phase = np.angle(true_coeff)
            pred_phase = np.angle(pred_coeff)
            
            # Phase error in degrees, wrapped to [-180, 180]
            phase_error = np.angle(pred_coeff / true_coeff) if true_mag > 1e-12 else 0.0
            phase_error_deg = np.degrees(phase_error)
            if phase_error_deg > 180:
                phase_error_deg -= 360
            elif phase_error_deg < -180:
                phase_error_deg += 360
                
            rows.append({
                'sample': int(sample_idx),
                'l': l,
                'm': m,
                'type': 'E',
                'mode_idx': k,
                'true_real': float(true_coeff.real),
                'true_imag': float(true_coeff.imag),
                'pred_real': float(pred_coeff.real),
                'pred_imag': float(pred_coeff.imag),
                'true_mag': float(true_mag),
                'pred_mag': float(pred_mag),
                'mag_error': float(pred_mag - true_mag),
                'mag_rel_error': float((pred_mag - true_mag) / (true_mag + 1e-12)),
                'true_phase_deg': float(np.degrees(true_phase)),
                'pred_phase_deg': float(np.degrees(pred_phase)),
                'phase_error_deg': float(phase_error_deg),
                'complex_error': float(np.abs(pred_coeff - true_coeff)),
            })
        
        # Magnetic coefficients
        for k, (l, m) in enumerate(mode_list):
            true_coeff = a_m_true[sample_idx, k]
            pred_coeff = a_m_pred[sample_idx, k]
            
            true_mag = np.abs(true_coeff)
            pred_mag = np.abs(pred_coeff)
            true_phase = np.angle(true_coeff)
            pred_phase = np.angle(pred_coeff)
            
            # Phase error in degrees, wrapped to [-180, 180]
            phase_error = np.angle(pred_coeff / true_coeff) if true_mag > 1e-12 else 0.0
            phase_error_deg = np.degrees(phase_error)
            if phase_error_deg > 180:
                phase_error_deg -= 360
            elif phase_error_deg < -180:
                phase_error_deg += 360
                
            rows.append({
                'sample': int(sample_idx),
                'l': l,
                'm': m,
                'type': 'M',
                'mode_idx': k,
                'true_real': float(true_coeff.real),
                'true_imag': float(true_coeff.imag),
                'pred_real': float(pred_coeff.real),
                'pred_imag': float(pred_coeff.imag),
                'true_mag': float(true_mag),
                'pred_mag': float(pred_mag),
                'mag_error': float(pred_mag - true_mag),
                'mag_rel_error': float((pred_mag - true_mag) / (true_mag + 1e-12)),
                'true_phase_deg': float(np.degrees(true_phase)),
                'pred_phase_deg': float(np.degrees(pred_phase)),
                'phase_error_deg': float(phase_error_deg),
                'complex_error': float(np.abs(pred_coeff - true_coeff)),
            })
    
    return pd.DataFrame(rows)


def export_coefficient_csv(
    table_data: pd.DataFrame,
    output_path: Path,
    include_summary: bool = True
) -> None:
    """
    Export coefficient comparison table to CSV.
    
    Parameters
    ----------
    table_data : pd.DataFrame
        Table from create_coefficient_table
    output_path : Path
        Output CSV file path
    include_summary : bool
        If True, also create a summary statistics file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_data.to_csv(output_path, index=False)
    
    if include_summary:
        summary_path = output_path.with_suffix('.summary.json')
        
        summary = {
            'total_coefficients': len(table_data),
            'n_samples': len(table_data['sample'].unique()),
            'n_modes': len(table_data['mode_idx'].unique()),
            'max_l': int(table_data['l'].max()),
            'statistics': {
                'mag_error': {
                    'mean': float(table_data['mag_error'].mean()),
                    'std': float(table_data['mag_error'].std()),
                    'rms': float(np.sqrt((table_data['mag_error'] ** 2).mean())),
                },
                'mag_rel_error': {
                    'mean': float(table_data['mag_rel_error'].mean()),
                    'std': float(table_data['mag_rel_error'].std()),
                    'rms': float(np.sqrt((table_data['mag_rel_error'] ** 2).mean())),
                },
                'phase_error_deg': {
                    'mean': float(table_data['phase_error_deg'].mean()),
                    'std': float(table_data['phase_error_deg'].std()),
                    'rms': float(np.sqrt((table_data['phase_error_deg'] ** 2).mean())),
                },
                'complex_error': {
                    'mean': float(table_data['complex_error'].mean()),
                    'std': float(table_data['complex_error'].std()),
                    'rms': float(np.sqrt((table_data['complex_error'] ** 2).mean())),
                },
            },
            'by_type': {}
        }
        
        # Statistics by coefficient type
        for coeff_type in ['E', 'M']:
            type_data = table_data[table_data['type'] == coeff_type]
            summary['by_type'][coeff_type] = {
                'mag_error_rms': float(np.sqrt((type_data['mag_error'] ** 2).mean())),
                'phase_error_rms_deg': float(np.sqrt((type_data['phase_error_deg'] ** 2).mean())),
                'complex_error_rms': float(np.sqrt((type_data['complex_error'] ** 2).mean())),
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def plot_coefficient_error_heatmap(
    table_data: pd.DataFrame,
    error_type: str = 'complex_error',
    save_path: Path | None = None,
) -> Figure:
    """
    Create heatmap of coefficient errors by (l,m) mode.
    
    Parameters
    ----------
    table_data : pd.DataFrame
        Table from create_coefficient_table
    error_type : str
        Column name for error to plot ('complex_error', 'mag_error', 'phase_error_deg')
    save_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    Figure
    """
    _require_mpl()
    
    # Aggregate error by (l,m) and type
    grouped = table_data.groupby(['l', 'm', 'type'])[error_type].mean().reset_index()
    
    # Get unique l and m values
    l_values = sorted(grouped['l'].unique())
    m_max = grouped['m'].abs().max()
    m_values = list(range(-m_max, m_max + 1))
    
    fig, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(12, 5))
    
    for coeff_type, ax in [('E', ax_e), ('M', ax_m)]:
        type_data = grouped[grouped['type'] == coeff_type]
        
        # Create matrix for heatmap
        error_matrix = np.full((len(l_values), len(m_values)), np.nan)
        
        for _, row in type_data.iterrows():
            l_idx = l_values.index(row['l'])
            m_idx = m_values.index(row['m'])
            if abs(row['m']) <= row['l']:  # Valid (l,m) pair
                error_matrix[l_idx, m_idx] = row[error_type]
        
        im = ax.imshow(
            error_matrix,
            aspect='auto',
            cmap='viridis',
            origin='lower'
        )
        ax.set_xlabel('m')
        ax.set_ylabel('l')
        ax.set_title(f'{coeff_type}-type coefficients: {error_type}')
        
        # Set ticks
        ax.set_xticks(range(len(m_values)))
        ax.set_xticklabels(m_values)
        ax.set_yticks(range(len(l_values)))
        ax.set_yticklabels(l_values)
        
        plt.colorbar(im, ax=ax)
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_coefficient_comparison_bars(
    table_data: pd.DataFrame,
    sample_idx: int,
    save_path: Path | None = None,
) -> Figure:
    """
    Create bar chart comparing true vs predicted coefficients for a specific sample.
    
    Parameters
    ----------
    table_data : pd.DataFrame
        Table from create_coefficient_table
    sample_idx : int
        Sample index to plot
    save_path : Path, optional
        If provided, save figure to this path
        
    Returns
    -------
    Figure
    """
    _require_mpl()
    
    sample_data = table_data[table_data['sample'] == sample_idx]
    if len(sample_data) == 0:
        raise ValueError(f"No data found for sample {sample_idx}")
    
    fig, (ax_e, ax_m) = plt.subplots(2, 1, figsize=(12, 8))
    
    for coeff_type, ax in [('E', ax_e), ('M', ax_m)]:
        type_data = sample_data[sample_data['type'] == coeff_type].copy()
        type_data = type_data.sort_values(['l', 'm'])
        
        n_modes = len(type_data)
        x = np.arange(n_modes)
        width = 0.35
        
        ax.bar(x - width/2, type_data['true_mag'], width, 
               label='True', alpha=0.8, color='blue')
        ax.bar(x + width/2, type_data['pred_mag'], width, 
               label='Predicted', alpha=0.8, color='red')
        
        # Mode labels
        mode_labels = [f"({row['l']},{row['m']})" for _, row in type_data.iterrows()]
        ax.set_xlabel('Mode (l,m)')
        ax.set_ylabel('|coefficient|')
        ax.set_title(f'Sample {sample_idx}: {coeff_type}-type coefficient magnitudes')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def coefficient_validation_summary(
    a_e_true: np.ndarray,
    a_m_true: np.ndarray,
    a_e_pred: np.ndarray,
    a_m_pred: np.ndarray,
    mode_list: list[tuple[int, int]],
    n_sample_previews: int = 5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Complete coefficient validation with tables, plots, and summary statistics.
    
    Parameters
    ----------
    a_e_true, a_m_true : (N, n_modes) complex arrays
        True coefficients
    a_e_pred, a_m_pred : (N, n_modes) complex arrays  
        Predicted coefficients
    mode_list : list of (l, m) tuples
        Mode indices
    n_sample_previews : int
        Number of individual sample plots to create
    output_dir : Path, optional
        Directory to save outputs. If None, only return statistics.
        
    Returns
    -------
    dict
        Summary statistics and file paths
    """
    n_samples = len(a_e_true)
    
    # Select sample indices for detailed analysis
    if n_samples <= n_sample_previews:
        preview_indices = np.arange(n_samples)
    else:
        preview_indices = np.linspace(0, n_samples - 1, n_sample_previews, dtype=int)
    
    # Create full coefficient table
    table_data = create_coefficient_table(
        a_e_true, a_m_true, a_e_pred, a_m_pred, mode_list, preview_indices
    )
    
    results = {
        'n_samples_analyzed': len(preview_indices),
        'n_modes': len(mode_list),
        'table_stats': {
            'mag_error_rms': float(np.sqrt((table_data['mag_error'] ** 2).mean())),
            'phase_error_rms_deg': float(np.sqrt((table_data['phase_error_deg'] ** 2).mean())),
            'complex_error_rms': float(np.sqrt((table_data['complex_error'] ** 2).mean())),
        },
        'files': {}
    }
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export CSV table
        csv_path = output_dir / 'coefficient_comparison.csv'
        export_coefficient_csv(table_data, csv_path, include_summary=True)
        results['files']['table_csv'] = str(csv_path)
        results['files']['summary_json'] = str(csv_path.with_suffix('.summary.json'))
        
        # Create error heatmaps
        for error_type in ['complex_error', 'mag_error', 'phase_error_deg']:
            heatmap_path = output_dir / f'coefficient_heatmap_{error_type}.png'
            plot_coefficient_error_heatmap(table_data, error_type, heatmap_path)
            results['files'][f'heatmap_{error_type}'] = str(heatmap_path)
        
        # Create individual sample comparisons
        sample_plots = []
        for sample_idx in preview_indices:
            plot_path = output_dir / f'coefficient_bars_sample_{sample_idx:04d}.png'
            plot_coefficient_comparison_bars(table_data, sample_idx, plot_path)
            sample_plots.append(str(plot_path))
        results['files']['sample_plots'] = sample_plots
    
    return results