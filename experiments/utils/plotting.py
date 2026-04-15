#!/usr/bin/env python3
"""
Training and experiment visualization utilities.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.data_generator import unpack_coefficients, DataGenerator

def plot_training_curves(
    training_history: Dict[str, List[float]], 
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        training_history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    ax.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in training_history:
        ax.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to log scale if values span multiple orders of magnitude
    train_range = max(training_history['train_loss']) / min(training_history['train_loss'])
    if train_range > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_prediction_scatter(
    y_true: Union[np.ndarray, str],
    y_pred: Union[np.ndarray, str],
    title: str = "Predictions vs True Values",
    save_path: Optional[Path] = None,
    max_points: int = 10000,
    streaming_batch_size: int = 5000
) -> Figure:
    """
    Plot scatter plot of predictions vs true values with memory-mapped data support.
    
    Args:
        y_true: True target values (array or path to memory-mapped .npy file)
        y_pred: Predicted values (array or path to memory-mapped .npy file)
        title: Plot title
        save_path: Optional path to save the plot
        max_points: Maximum points to plot (for performance)
        streaming_batch_size: Batch size for loading streaming data
        
    Returns:
        matplotlib Figure object
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Handle memory-mapped data
    if isinstance(y_true, str) and isinstance(y_pred, str):
        logger.info(f"Loading streaming data for plotting from {y_true} and {y_pred}")
        y_true_flat, y_pred_flat = _load_data_for_plotting(
            y_true, y_pred, max_points, streaming_batch_size
        )
    else:
        # Handle in-memory arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Sample points if too many
        if len(y_true_flat) > max_points:
            logger.info(f"Sampling {max_points} points from {len(y_true_flat)} for performance")
            indices = np.random.choice(len(y_true_flat), max_points, replace=False)
            y_true_flat = y_true_flat[indices]
            y_pred_flat = y_pred_flat[indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create scatter plot with reduced alpha for better visualization
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=1)
    
    # Add perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R² using the standard SS-based definition (consistent with training metrics)
    try:
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    except Exception:
        r_squared = 0.0
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values') 
    ax.set_title(f'{title}\nR² = {r_squared:.4f} ({len(y_true_flat)} points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def _load_data_for_plotting(
    y_true_path: str, 
    y_pred_path: str, 
    max_points: int,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data for plotting from memory-mapped files efficiently.
    
    Args:
        y_true_path: Path to true values memory-mapped file
        y_pred_path: Path to predicted values memory-mapped file
        max_points: Maximum points to load
        batch_size: Batch size for loading
        
    Returns:
        Tuple of (y_true_flat, y_pred_flat) flattened arrays
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading streaming data for plotting (max {max_points} points)")
    
    # Load memory-mapped arrays
    y_true_mm = np.load(y_true_path, mmap_mode='r')
    y_pred_mm = np.load(y_pred_path, mmap_mode='r')
    
    if y_true_mm.shape != y_pred_mm.shape:
        logger.warning(f"Shape mismatch: y_true={y_true_mm.shape}, y_pred={y_pred_mm.shape}")
        # Try to handle by taking minimum dimensions
        min_samples = min(y_true_mm.shape[0], y_pred_mm.shape[0])
        y_true_mm = y_true_mm[:min_samples]
        y_pred_mm = y_pred_mm[:min_samples]
    
    n_samples = y_true_mm.shape[0]
    total_elements = np.prod(y_true_mm.shape)
    
    # If total elements <= max_points, load everything
    if total_elements <= max_points:
        logger.info(f"Loading all {total_elements} elements")
        y_true_flat = y_true_mm.flatten()
        y_pred_flat = y_pred_mm.flatten()
        return y_true_flat, y_pred_flat
    
    # Otherwise, sample uniformly across the dataset
    logger.info(f"Sampling {max_points} points from {total_elements} total elements")
    
    # Use systematic sampling for better coverage
    step = total_elements // max_points
    indices = np.arange(0, total_elements, step)[:max_points]
    
    # Convert to multi-dimensional indices
    sample_indices = np.unravel_index(indices, y_true_mm.shape)
    
    # Extract samples
    y_true_flat = y_true_mm[sample_indices]
    y_pred_flat = y_pred_mm[sample_indices]
    
    logger.info(f"Loaded {len(y_true_flat)} points for plotting")
    return y_true_flat, y_pred_flat

def plot_coefficient_comparison(
    coeffs_true: np.ndarray,
    coeffs_pred: np.ndarray,
    maxorder: int,
    sample_idx: int = 0,
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot comparison of true vs predicted coefficients for a single sample.
    
    Args:
        coeffs_true: True coefficients (n_samples, n_coeffs) or (n_coeffs,)
        coeffs_pred: Predicted coefficients (n_samples, n_coeffs) or (n_coeffs,)
        maxorder: Maximum multipole order
        sample_idx: Which sample to plot (if multi-sample arrays)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    # Extract single sample if needed
    if coeffs_true.ndim > 1:
        coeffs_true = coeffs_true[sample_idx]
    if coeffs_pred.ndim > 1:
        coeffs_pred = coeffs_pred[sample_idx]
    
    # Split into E and M coefficients (assuming packed format)
    n_modes = maxorder * (maxorder + 2)
    n_coeffs_per_type = n_modes * 2  # real + imaginary
    
    e_coeffs_true = coeffs_true[:n_coeffs_per_type]
    m_coeffs_true = coeffs_true[n_coeffs_per_type:2*n_coeffs_per_type]
    e_coeffs_pred = coeffs_pred[:n_coeffs_per_type]
    m_coeffs_pred = coeffs_pred[n_coeffs_per_type:2*n_coeffs_per_type]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # E coefficients
    x = np.arange(len(e_coeffs_true))
    width = 0.35
    
    ax1.bar(x - width/2, e_coeffs_true, width, label='True', alpha=0.7)
    ax1.bar(x + width/2, e_coeffs_pred, width, label='Predicted', alpha=0.7)
    ax1.set_title(f'Electric Coefficients (Sample {sample_idx})')
    ax1.set_ylabel('Coefficient Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # M coefficients
    ax2.bar(x - width/2, m_coeffs_true, width, label='True', alpha=0.7)
    ax2.bar(x + width/2, m_coeffs_pred, width, label='Predicted', alpha=0.7)
    ax2.set_title(f'Magnetic Coefficients (Sample {sample_idx})')
    ax2.set_xlabel('Coefficient Index')
    ax2.set_ylabel('Coefficient Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def plot_loss_distribution(
    losses: np.ndarray,
    title: str = "Loss Distribution",
    save_path: Optional[Path] = None
) -> Figure:
    """
    Plot histogram of loss values.
    
    Args:
        losses: Array of loss values per sample
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(losses, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(losses.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {losses.mean():.4f}')
    ax.axvline(np.median(losses), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(losses):.4f}')
    
    ax.set_xlabel('Loss Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def create_experiment_summary_plot(
    config: Dict[str, Any],
    training_history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> Figure:
    """
    Create a comprehensive experiment summary plot.
    
    Args:
        config: Experiment configuration
        training_history: Training history with losses
        test_metrics: Final test metrics
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.3)
    
    # Training curves
    ax1 = fig.add_subplot(gs[0, :])
    epochs = range(1, len(training_history['train_loss']) + 1)
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in training_history:
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configuration summary
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    config_text = "Configuration:\n"
    if 'model' in config:
        model_config = config['model']
        config_text += f"• Model: {model_config.get('type', 'N/A')}\n"
        config_text += f"• Max Order: {model_config.get('maxorder', 'N/A')}\n"
        config_text += f"• Hidden Size: {model_config.get('hidden_size', 'N/A')}\n"
        config_text += f"• Layers: {model_config.get('n_hidden_layers', 'N/A')}\n"
    if 'training' in config:
        train_config = config['training']
        config_text += f"• Samples: {train_config.get('n_samples', 'N/A')}\n"
        config_text += f"• Epochs: {train_config.get('epochs', 'N/A')}\n"
        config_text += f"• Batch Size: {train_config.get('batch_size', 'N/A')}\n"
        config_text += f"• Learning Rate: {train_config.get('learning_rate', 'N/A')}\n"
    
    ax2.text(0.05, 0.95, config_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Test metrics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    metrics_text = "Final Results:\n"
    for key, value in test_metrics.items():
        if isinstance(value, float):
            metrics_text += f"• {key}: {value:.6f}\n"
        else:
            metrics_text += f"• {key}: {value}\n"
    
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Experiment info
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    experiment_name = config.get('experiment', {}).get('name', 'Unknown')
    experiment_desc = config.get('experiment', {}).get('description', '')
    ax4.text(0.5, 0.5, f"Experiment: {experiment_name}\n{experiment_desc}", 
             transform=ax4.transAxes, fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_field_comparison(
    y_true: Union[np.ndarray, str],
    y_pred: Union[np.ndarray, str],
    sample_idx: int = 0,
    maxorder: int = 5,
    save_path: Optional[Path] = None,
    title_prefix: str = ""
) -> Figure:
    """
    Plot comprehensive field comparison showing true vs predicted electromagnetic fields.
    
    Creates a 3x3 visualization:
    - Top row: True fields (P_LT, |E_θ|, |E_φ|)
    - Middle row: Predicted fields (P^∞, |E_θ^∞|, |E_φ^∞|) 
    - Bottom row: Normalized differences
    
    Args:
        y_true: True coefficient values (array or path to memory-mapped .npy file)
        y_pred: Predicted coefficient values (array or path to memory-mapped .npy file)
        sample_idx: Which sample to visualize
        maxorder: Maximum multipole order for reconstruction
        save_path: Optional path to save the plot
        title_prefix: Prefix for plot titles (e.g., "Sample 1576 [test]")
        
    Returns:
        matplotlib Figure object
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Handle memory-mapped data loading
    if isinstance(y_true, str) and isinstance(y_pred, str):
        logger.info(f"Loading sample {sample_idx} from memory-mapped files for field comparison")
        
        # Load only the specific sample we need
        y_true_mm = np.load(y_true, mmap_mode='r')
        y_pred_mm = np.load(y_pred, mmap_mode='r')
        
        if sample_idx >= y_true_mm.shape[0] or sample_idx >= y_pred_mm.shape[0]:
            logger.warning(f"Sample index {sample_idx} out of range, using last sample")
            sample_idx = min(y_true_mm.shape[0] - 1, y_pred_mm.shape[0] - 1)
        
        y_true_sample = y_true_mm[sample_idx:sample_idx+1].copy()  # Copy to avoid memory mapping issues
        y_pred_sample = y_pred_mm[sample_idx:sample_idx+1].copy()
        
    else:
        # Handle in-memory arrays
        y_true_sample = y_true[sample_idx:sample_idx+1] if len(y_true.shape) > 1 else y_true.reshape(1, -1)
        y_pred_sample = y_pred[sample_idx:sample_idx+1] if len(y_pred.shape) > 1 else y_pred.reshape(1, -1)
    
    # Unpack coefficients from predictions
    a_e_true, a_m_true = unpack_coefficients(y_true_sample)
    a_e_pred, a_m_pred = unpack_coefficients(y_pred_sample)
    
    # Reshape to remove batch dimension
    a_e_true = a_e_true[0]  # Shape: (n_modes,)
    a_m_true = a_m_true[0] 
    a_e_pred = a_e_pred[0]
    a_m_pred = a_m_pred[0]
    
    # Reconstruct electromagnetic fields using DataGenerator
    data_generator = DataGenerator.for_pipeline()
    field_generator = data_generator.field_generator
    
    # Build angular grid
    theta, phi = field_generator.build_grid()  # Shape: (n_phi, n_theta)
    
    # Compute true fields
    amplitude_true = field_generator.compute_field_from_array(a_e_true, a_m_true, maxorder)
    E_theta_true = amplitude_true[..., 0]  # Shape: (n_phi, n_theta)
    E_phi_true = amplitude_true[..., 1]    # Shape: (n_phi, n_theta)
    P_true = field_generator.compute_power(amplitude_true)  # Shape: (n_phi, n_theta)
    
    # Debug: Check coefficient values before field computation
    logger.info(f"DEBUG - Field computation setup:")
    logger.info(f"  maxorder used: {maxorder}")
    logger.info(f"  coefficient shapes: a_e_true={a_e_true.shape}, a_m_true={a_m_true.shape}, a_e_pred={a_e_pred.shape}, a_m_pred={a_m_pred.shape}")
    logger.info(f"DEBUG - Input coefficient values:")
    logger.info(f"  a_e_true: mean={a_e_true.mean():.6f}, std={a_e_true.std():.6f}, range=[{a_e_true.min():.6f}, {a_e_true.max():.6f}]")
    logger.info(f"  a_m_true: mean={a_m_true.mean():.6f}, std={a_m_true.std():.6f}, range=[{a_m_true.min():.6f}, {a_m_true.max():.6f}]")
    logger.info(f"  a_e_pred: mean={a_e_pred.mean():.6f}, std={a_e_pred.std():.6f}, range=[{a_e_pred.min():.6f}, {a_e_pred.max():.6f}]")
    logger.info(f"  a_m_pred: mean={a_m_pred.mean():.6f}, std={a_m_pred.std():.6f}, range=[{a_m_pred.min():.6f}, {a_m_pred.max():.6f}]")
    
    # Compute predicted fields  
    amplitude_pred = field_generator.compute_field_from_array(a_e_pred, a_m_pred, maxorder)
    E_theta_pred = amplitude_pred[..., 0]
    E_phi_pred = amplitude_pred[..., 1]
    P_pred = field_generator.compute_power(amplitude_pred)
    
    # Debug: Check computed field values
    logger.info(f"DEBUG - Computed field values:")
    logger.info(f"  P_true: mean={P_true.mean():.6f}, std={P_true.std():.6f}, range=[{P_true.min():.6f}, {P_true.max():.6f}]")
    logger.info(f"  P_pred: mean={P_pred.mean():.6f}, std={P_pred.std():.6f}, range=[{P_pred.min():.6f}, {P_pred.max():.6f}]")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    
    # Define field data and labels
    fields_true = [P_true, np.abs(E_theta_true), np.abs(E_phi_true)]
    fields_pred = [P_pred, np.abs(E_theta_pred), np.abs(E_phi_pred)]
    field_labels = ['P_LT (true)', '|E_θ| (true)', '|E_φ| (true)']
    pred_labels = ['P^∞ (predicted)', '|E_θ^∞| (predicted)', '|E_φ^∞| (predicted)']
    diff_labels = ['ΔP/|P| (% difference)', 'Δ|E_θ|/|E_θ| (% difference)', 'Δ|E_φ|/|E_φ| (% difference)']
    
    # Use separate scales for true vs predicted fields (better visualization) 
    # The synthetic "true" coefficients often have very different magnitudes than model predictions
    true_vmins = []
    true_vmaxs = []
    pred_vmins = []
    pred_vmaxs = []
    
    for i in range(3):
        true_field = fields_true[i]
        pred_field = fields_pred[i]
        
        # Separate scales for better dynamic range
        true_vmins.append(np.min(true_field))
        true_vmaxs.append(np.max(true_field))
        pred_vmins.append(np.min(pred_field))
        pred_vmaxs.append(np.max(pred_field))
        
        logger.info(f"Field {i}: True range=[{true_vmins[i]:.6f}, {true_vmaxs[i]:.6f}], Pred range=[{pred_vmins[i]:.6f}, {pred_vmaxs[i]:.6f}]")
    
    # Plot true fields (top row) - use separate scale for true fields
    for i in range(3):
        ax = axes[0, i]
        field = fields_true[i]
        im = ax.imshow(field, aspect='auto', origin='lower', cmap='viridis', 
                      vmin=true_vmins[i], vmax=true_vmaxs[i],
                      extent=[1, theta.shape[1], 1, theta.shape[0]])
        ax.set_title(f'{field_labels[i]} (max={true_vmaxs[i]:.3f})')
        ax.set_xlabel('polar angle θ (deg)')
        ax.set_ylabel('azimuth angle φ (deg)')
        
        # Set proper tick labels
        n_theta = theta.shape[1]
        n_phi = theta.shape[0] 
        ax.set_xticks([1, n_theta//4, n_theta//2, 3*n_theta//4, n_theta])
        ax.set_xticklabels(['1', '45', '90', '135', '179'])
        ax.set_yticks([1, n_phi//4, n_phi//2, 3*n_phi//4, n_phi])
        ax.set_yticklabels(['1', '90', '180', '270', '359'])
        
        plt.colorbar(im, ax=ax)
    
    # Plot predicted fields (middle row) - use separate scale for predicted fields
    for i in range(3):
        ax = axes[1, i]
        field = fields_pred[i]
        im = ax.imshow(field, aspect='auto', origin='lower', cmap='viridis',  # Same colormap
                      vmin=pred_vmins[i], vmax=pred_vmaxs[i],  # Separate scale for predicted
                      extent=[1, theta.shape[1], 1, theta.shape[0]])
        ax.set_title(f'{pred_labels[i]} (max={pred_vmaxs[i]:.3f})')
        ax.set_xlabel('polar angle θ (deg)')
        ax.set_ylabel('azimuth angle φ (deg)')
        
        # Set proper tick labels
        n_theta = theta.shape[1]
        n_phi = theta.shape[0]
        ax.set_xticks([1, n_theta//4, n_theta//2, 3*n_theta//4, n_theta])
        ax.set_xticklabels(['1', '45', '90', '135', '179'])
        ax.set_yticks([1, n_phi//4, n_phi//2, 3*n_phi//4, n_phi])
        ax.set_yticklabels(['1', '90', '180', '270', '359'])
        
        plt.colorbar(im, ax=ax)
    
    # Plot normalized differences (bottom row) - FIXED SCALE [-1, 1]
    for i in range(3):
        ax = axes[2, i]
        true_field = fields_true[i]
        pred_field = fields_pred[i]
        
        # Compute normalized percentage difference
        # Avoid division by zero
        epsilon = 1e-10
        true_field_safe = np.where(np.abs(true_field) < epsilon, epsilon, true_field)
        normalized_diff = (pred_field - true_field) / np.abs(true_field_safe)
        
        # Clamp to [-1, 1] range for percentage interpretation
        normalized_diff = np.clip(normalized_diff, -1.0, 1.0)
        
        # Use diverging colormap with FIXED scale [-1, 1] for percentage comparison
        im = ax.imshow(normalized_diff, aspect='auto', origin='lower', 
                      cmap='RdBu_r', vmin=-1.0, vmax=1.0,  # Fixed scale for percentage
                      extent=[1, theta.shape[1], 1, theta.shape[0]])
        ax.set_title(diff_labels[i])
        ax.set_xlabel('polar angle θ (deg)')
        ax.set_ylabel('azimuth angle φ (deg)')
        
        # Set proper tick labels
        n_theta = theta.shape[1]
        n_phi = theta.shape[0]
        ax.set_xticks([1, n_theta//4, n_theta//2, 3*n_theta//4, n_theta])
        ax.set_xticklabels(['1', '45', '90', '135', '179'])
        ax.set_yticks([1, n_phi//4, n_phi//2, 3*n_phi//4, n_phi])
        ax.set_yticklabels(['1', '90', '180', '270', '359'])
        
        # Add percentage ticks for difference plots
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['-100%', '-50%', '0%', '+50%', '+100%'])
    
    # Set overall title
    title_text = f"{title_prefix} — true vs model with differences" if title_prefix else "Field comparison — true vs model with differences"
    fig.suptitle(title_text, fontsize=16, y=0.98)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_p_field_comparison(
    P_true: np.ndarray,
    P_pred: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[Path] = None,
    title_prefix: str = ""
) -> Figure:
    """
    Plot side-by-side 2D heatmaps comparing true and predicted P-fields.

    Creates a 1x3 figure:
    - Panel 1: True P field (imshow)
    - Panel 2: Predicted P field (imshow, same colour range)
    - Panel 3: Relative difference (pred-true)/|true|, clipped to [-1, 1]

    Args:
        P_true: True power-field array, shape (N, n_phi, n_theta) or (n_phi, n_theta).
        P_pred: Predicted power-field array, same shape as P_true.
        sample_idx: Which sample to visualise when arrays are 3-D.
        save_path: Optional path to save the figure.
        title_prefix: String prepended to the figure suptitle.

    Returns:
        matplotlib Figure object.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Extract a single sample
    if P_true.ndim == 3:
        if sample_idx >= P_true.shape[0]:
            logger.warning(f"sample_idx {sample_idx} out of range, clamping to last sample")
            sample_idx = P_true.shape[0] - 1
        P_true_s = P_true[sample_idx]
        P_pred_s = P_pred[sample_idx]
    else:
        P_true_s = P_true
        P_pred_s = P_pred

    # Relative difference, clipped to [-1, 1]
    eps = 1e-10
    rel_diff = (P_pred_s - P_true_s) / (np.abs(P_true_s) + eps)
    rel_diff = np.clip(rel_diff, -1.0, 1.0)

    # Colour range set by true values only so the predicted panel uses the same
    # physical scale as the ground truth, regardless of prediction extremes.
    vmin = float(P_true_s.min())
    vmax = float(P_true_s.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # After the grid-alignment fix, P arrays are stored as (n_theta, n_phi) = (179, 360).
    # imshow: rows → y-axis (theta), columns → x-axis (phi).
    n_theta, n_phi = P_true_s.shape
    extent = [0, n_phi - 1, 0, n_theta - 1]

    # Panel 1 – True P
    im0 = axes[0].imshow(P_true_s, aspect='auto', origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("True P field")
    axes[0].set_xlabel("phi index (0..359)")
    axes[0].set_ylabel("theta index (0..178)")
    plt.colorbar(im0, ax=axes[0])

    # Panel 2 – Predicted P
    im1 = axes[1].imshow(P_pred_s, aspect='auto', origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title("Predicted P field")
    axes[1].set_xlabel("phi index (0..359)")
    axes[1].set_ylabel("theta index (0..178)")
    plt.colorbar(im1, ax=axes[1])

    # Panel 3 – Relative difference with symmetric-log scale.
    # SymLogNorm: linear between ±linthresh, logarithmic outside.
    # linthresh=0.01 means differences below 1% are treated linearly (avoids log(0)).
    from matplotlib.colors import SymLogNorm
    sym_norm = SymLogNorm(linthresh=0.01, vmin=-1.0, vmax=1.0, base=10)
    im2 = axes[2].imshow(rel_diff, aspect='auto', origin='lower',
                         cmap='RdBu_r', norm=sym_norm, extent=extent)
    axes[2].set_title("Relative diff (pred-true)/|true|  [symlog]")
    axes[2].set_xlabel("phi index (0..359)")
    axes[2].set_ylabel("theta index (0..178)")
    cbar2 = plt.colorbar(im2, ax=axes[2])
    cbar2.set_ticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
    cbar2.set_ticklabels(['-100%', '-10%', '-1%', '0%', '+1%', '+10%', '+100%'])

    # Compute and display R² in panel 2 title
    ss_res = np.sum((P_true_s - P_pred_s) ** 2)
    ss_tot = np.sum((P_true_s - np.mean(P_true_s)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    axes[1].set_title(f"Predicted P field  (R²={r2:.3f})")

    title_text = (f"{title_prefix} — P field comparison" if title_prefix
                  else "P field comparison: true vs predicted")
    fig.suptitle(title_text, fontsize=14, y=1.01)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# =============================================================================
# Streaming and Memory-Mapped Plotting Utilities
# =============================================================================

def plot_streaming_results(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_samples: int = 5,
    max_points_per_plot: int = 10000
) -> Dict[str, Path]:
    """
    Create comprehensive plots from streaming evaluation results.
    
    Args:
        results_dir: Directory containing streaming evaluation results
        output_dir: Directory to save plots
        max_samples: Maximum number of samples to plot for field comparisons
        max_points_per_plot: Maximum points per scatter plot
        
    Returns:
        Dictionary mapping plot types to saved paths
    """
    import json
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating comprehensive plots from {results_dir}")
    
    plot_paths = {}
    
    # 1. Load evaluation results
    results_file = results_dir / "evaluation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        output_paths = results.get('output_paths', {})
        
        # 2. Create scatter plots if we have predictions
        if 'P_predicted' in output_paths and 'P_field' in output_paths:
            scatter_path = output_dir / "P_field_scatter.png"
            logger.info("Creating P field scatter plot...")
            
            plot_prediction_scatter(
                output_paths['P_field'],  # True P field (from components)
                output_paths['P_predicted'],  # Predicted P field
                title="P Field Predictions (Streaming Evaluation)",
                save_path=scatter_path,
                max_points=max_points_per_plot
            )
            plot_paths['P_field_scatter'] = scatter_path
        
        # 3. Create field component scatter plots
        if 'E_theta' in output_paths and 'E_phi' in output_paths:
            E_theta_path = output_dir / "E_theta_scatter.png" 
            E_phi_path = output_dir / "E_phi_scatter.png"
            
            # Note: We don't have separate true E field files, so we skip these for now
            # In a full implementation, we'd need to generate or load true field components
            logger.info("Field component plots require additional true field data")
    
    # 4. Create memory usage plots if we have metrics
    metrics_file = results_dir / "physics_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        # Create simple metrics summary plot
        metrics_path = output_dir / "evaluation_metrics.png"
        _create_metrics_summary_plot(metrics, metrics_path)
        plot_paths['metrics_summary'] = metrics_path
    
    logger.info(f"Created {len(plot_paths)} plots in {output_dir}")
    return plot_paths


def _create_metrics_summary_plot(metrics: Dict[str, float], save_path: Path) -> None:
    """Create a simple metrics summary visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract numeric metrics for plotting
    metric_names = []
    metric_values = []
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.startswith('n_'):
            metric_names.append(key.replace('_', ' ').title())
            metric_values.append(value)
    
    if metric_names:
        # Create bar plot
        bars = ax.bar(metric_names, metric_values)
        ax.set_ylabel('Metric Value')
        ax.set_title('Physics Evaluation Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom')
        
        # Add processing info as text
        info_text = f"Samples: {metrics.get('n_samples', 'N/A')}\n"
        info_text += f"Batches: {metrics.get('n_batches', 'N/A')}\n" 
        info_text += f"Batch Size: {metrics.get('batch_size', 'N/A')}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)