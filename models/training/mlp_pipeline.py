"""
MLP Physics Trainer — v0 neural model.

Architecture:
    [X_theta || X_phi]  (N, 2*n_points = 128880)
        → PCA  (128880 → pca_components)
        → _PhysicsMLP  (pca_components → hidden × n_hidden_layers → 4·K)
        → [frozen physics decoder]  (coefficients → power)

Training objective (same as the 'physics' baseline trainer):
    L = L_shape + λ_amp · mean_batch (log P̂ − log P)²
    L_shape uses normalised p and area weights w ∝ sin θ (pole-corrected).
    P, P̂ are the area-weighted totals of observed and predicted power.

Use --amplitude-loss-weight 0 for shape-only training.

UPDATED: Real-time comprehensive physics metrics logging during training.
- Computes full physics metrics every `detailed_metrics_frequency` epochs
- Logs comprehensive metrics to MLflow during training (not just at end)  
- Creates enhanced training plots with comprehensive physics metrics
- Includes validation metrics: weighted_mse, beam_pointing_error_deg, 
  polarization_correlation, fss, and other physics-aware metrics

No coefficient supervision is used.  The model discovers coefficients
that reproduce observed power through the frozen decoder alone.

Compare against:
    python -m src.cli.run_train_baseline --trainer ridge    (PCA + Ridge, coeff MSE)
    python -m src.cli.run_train_baseline --trainer physics  (PCA + Linear, shape+amplitude)
    python -m src.cli.run_train_mlp                        (PCA + MLP,    shape+amplitude)  ← this
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

# Plotting for MLflow visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reuse all shared helpers from the baseline pipeline — no duplication.
# ---------------------------------------------------------------------------
from models.evaluation.metrics import compute_all as _compute_all
from models.tracking.mlflow_utils import log_basic_metrics, log_images, set_tag, start_run
from models.training.baseline_pipeline import (
    _concat_inputs,
    _decode_power_torch,
    _fit_normalization,
    _iter_batches,
    _log_validation_images,
    _mode_list,
    _pca_transform,
    _pca_transform_split,
    _physics_power_loss_batch,
    _randomized_pca_fit,
    _randomized_pca_fit_split,
    _reconstruct_polarization_batch,
    _reconstruct_power_batch,
    _save_splits,
    build_dataset,
    load_or_build_basis,
    log_dataset_to_mlflow,
    unpack_coeffs,
)
from models.analysis.coefficient_comparison import coefficient_validation_summary
from src.common.paths import DATA_ML_DATASETS_DIR, MODELS_ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MlpConfig:
    # Dataset / splits — must match the BaselineConfig used to build the dataset.
    n_samples: int = 10_000
    maxorder: int = 15
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # PCA preprocessing
    pca_components: int = 256
    pca_oversample: int = 16
    pca_iterations: int = 0
    # MLP architecture
    hidden_size: int = 512
    n_hidden_layers: int = 2
    dropout: float = 0.1
    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    # MLflow logging during training
    val_log_frequency: int = 10  # Log validation metrics every N epochs
    detailed_metrics_frequency: int = 20  # Log detailed physics metrics every N epochs
    # Memory optimization
    use_memory_efficient_pca: bool = True  # Use split arrays for PCA to avoid memory spikes
    transform_batch_size: int = 0  # Batch size for PCA transform (0 = auto: 4-16 based on dataset size)
    ultra_conservative: bool = False  # Force transform_batch_size = 2 for maximum memory safety
    device: str = "cpu"
    rebuild_dataset: bool = False
    # Same as BaselineConfig: log-total-power amplitude term in physics loss.
    amplitude_loss_weight: float = 1.0

    @property
    def n_modes(self) -> int:
        return self.maxorder * (self.maxorder + 2)

    @property
    def n_targets(self) -> int:
        return 4 * self.n_modes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _PhysicsMLP(nn.Module):
    """
    MLP that maps PCA-compressed input features to multipole coefficients.

    Architecture:
        Linear(in_dim, hidden_size) → GELU → Dropout
        [Linear(hidden_size, hidden_size) → GELU → Dropout] × (n_hidden_layers − 1)
        Linear(hidden_size, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        n_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _metrics(
    p_true: np.ndarray, 
    p_pred: np.ndarray, 
    sin_theta: np.ndarray,
    e_theta_true: np.ndarray | None = None,
    e_phi_true: np.ndarray | None = None,
    e_theta_pred: np.ndarray | None = None,
    e_phi_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute physics-aware metrics, optionally including polarization analysis.
    """
    # Standard power metrics
    metrics = _compute_all(p_true, p_pred, sin_theta)
    
    # Add polarization metrics if components are provided
    if all(x is not None for x in [e_theta_true, e_phi_true, e_theta_pred, e_phi_pred]):
        from models.evaluation.polarization_metrics import compute_polarization_metrics
        pol_metrics = compute_polarization_metrics(
            e_theta_true, e_phi_true, e_theta_pred, e_phi_pred, sin_theta
        )
        metrics.update(pol_metrics)
    
    return metrics


def _artifact_dir(config: MlpConfig) -> Path:
    return (
        MODELS_ARTIFACTS_DIR
        / f"mlp_L{config.maxorder}_N{config.n_samples}_seed{config.seed}"
        f"_h{config.hidden_size}_d{config.n_hidden_layers}"
    )


def train_and_evaluate(config: MlpConfig) -> dict[str, float]:
    """Build dataset, train the MLP, evaluate, log everything to MLflow."""
    if torch is None or nn is None:
        raise RuntimeError(
            "MLP pipeline requires PyTorch.  "
            "Install it with:  pip install torch"
        )

    DATA_ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset (reuse or build)
    # ------------------------------------------------------------------
    # MlpConfig shares dataset fields with BaselineConfig; build_dataset
    # only depends on n_samples / maxorder / seed / rebuild_dataset.
    x_theta_path, x_phi_path, y_true_path, y_proj_path, meta_path = build_dataset(
        config  # type: ignore[arg-type]  # duck-typed: same relevant fields
    )

    X_theta = np.load(x_theta_path, mmap_mode="r")
    X_phi = np.load(x_phi_path, mmap_mode="r")
    Y_true = np.load(y_true_path, mmap_mode="r")

    train_idx, val_idx, test_idx = _save_splits(
        config, X_theta.shape[0]  # type: ignore[arg-type]
    )

    log_dataset_to_mlflow(
        config,  # type: ignore[arg-type]
        x_theta_path, x_phi_path,
        y_true_path, y_proj_path,
        meta_path,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # ------------------------------------------------------------------
    # MLflow run for this training experiment
    # ------------------------------------------------------------------
    with start_run(
        "mlp_power_to_multipoles",
        params={
            "n_samples": config.n_samples,
            "maxorder": config.maxorder,
            "seed": config.seed,
            "pca_components": config.pca_components,
            "hidden_size": config.hidden_size,
            "n_hidden_layers": config.n_hidden_layers,
            "dropout": config.dropout,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "model_input": "[X_theta || X_phi], shape (N, 128880)",
            "training_loss": "area-weighted shape + log-total-power amplitude",
            "amplitude_loss_weight": config.amplitude_loss_weight,
        },
    ):
        set_tag("model", "MLP")
        metrics = _run_mlp(
            config=config,
            X_theta=X_theta,
            X_phi=X_phi,
            Y=Y_true,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            meta_path=meta_path,
            x_theta_path=x_theta_path,
            x_phi_path=x_phi_path,
            y_true_path=y_true_path,
            y_proj_path=y_proj_path,
        )

    return metrics


def _pca_transform_split_with_progress(
    X_theta: np.ndarray,
    X_phi: np.ndarray, 
    idx: np.ndarray, 
    pca: dict[str, np.ndarray], 
    batch_size: int,
    split_name: str = "Data"
) -> np.ndarray:
    """
    PCA transform with progress reporting for large datasets.
    """
    out = np.zeros((len(idx), pca["components"].shape[0]), dtype=np.float32)
    
    # Split mean and components
    theta_mean = pca["mean"][:X_theta.shape[1]]
    phi_mean = pca["mean"][X_theta.shape[1]:]
    theta_components = pca["components"][:, :X_theta.shape[1]]
    phi_components = pca["components"][:, X_theta.shape[1]:]
    
    n_batches = (len(idx) + batch_size - 1) // batch_size
    
    row = 0
    for i, batch in enumerate(_iter_batches(idx, batch_size)):
        # Progress reporting every 10% or every 100 batches
        if (i + 1) % max(1, n_batches // 10) == 0 or (i + 1) % 100 == 0:
            print(f"  {split_name} transform batch {i+1}/{n_batches} ({100*(i+1)/n_batches:.0f}%)")
        
        theta_centered = X_theta[batch].astype(np.float32) - theta_mean
        phi_centered = X_phi[batch].astype(np.float32) - phi_mean
        
        # Transform each part and combine
        theta_proj = theta_centered @ theta_components.T
        phi_proj = phi_centered @ phi_components.T
        
        out[row : row + len(batch)] = theta_proj + phi_proj
        row += len(batch)
    
    print(f"  {split_name} transform complete!")
    return out


def _compute_quick_val_metrics(
    y_pred_denorm: np.ndarray, 
    y_true: np.ndarray, 
    x_theta: np.ndarray, 
    x_phi: np.ndarray,
    basis: dict,
    n_modes: int,
    max_samples: int = 100
) -> dict[str, float]:
    """Compute quick validation metrics during training (subset of samples for speed)."""
    n_samples = min(len(y_pred_denorm), max_samples)
    indices = np.random.choice(len(y_pred_denorm), n_samples, replace=False)
    
    y_pred_subset = y_pred_denorm[indices]
    y_true_subset = y_true[indices] 
    x_theta_subset = x_theta[indices]
    x_phi_subset = x_phi[indices]
    
    # Reconstruct power patterns
    a_e_pred, a_m_pred = unpack_coeffs(y_pred_subset, n_modes=n_modes)
    p_pred = _reconstruct_power_batch(a_e_pred, a_m_pred, basis)
    p_true = x_theta_subset + x_phi_subset
    
    # Quick metrics
    p_mse = float(np.mean((p_pred - p_true) ** 2))
    p_mae = float(np.mean(np.abs(p_pred - p_true)))
    
    # Relative power error
    p_rel_err = np.mean(np.abs(p_pred - p_true) / (p_true + 1e-8))
    
    return {
        "quick_power_mse": p_mse,
        "quick_power_mae": p_mae, 
        "quick_power_rel_err": float(p_rel_err),
    }


def _create_training_plots(training_history: dict, epoch: int, artifact_dir: Path) -> list[str]:
    """
    Create a single comprehensive training plot with all valuable metrics.
    
    Args:
        training_history: Dict with lists of metrics over epochs
        epoch: Current epoch number
        artifact_dir: Directory to save plots
        
    Returns:
        List of plot file paths that were created (single comprehensive plot)
    """
    plt.style.use('default')
    plot_files = []
    
    # Create a single comprehensive 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    
    epochs = training_history['epochs']
    if not epochs:
        return plot_files
    
    # 1. TOP LEFT: Training vs Validation Loss (most important for overfitting detection)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=3)
    ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('[LOSS] Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(epochs))
    
    # Add overfitting indicator text
    if len(epochs) >= 3:
        train_trend = training_history['train_loss'][-1] - training_history['train_loss'][-min(3, len(epochs))]
        val_trend = training_history['val_loss'][-1] - training_history['val_loss'][-min(3, len(epochs))]
        if train_trend < -0.01 and val_trend > 0.01:  # Training decreasing, validation increasing
            ax1.text(0.02, 0.98, 'WARNING: Potential Overfitting', transform=ax1.transAxes, 
                    fontsize=10, color='red', weight='bold', va='top')
        elif train_trend < -0.01 and val_trend < -0.01:  # Both decreasing
            ax1.text(0.02, 0.98, 'OK: Good Training', transform=ax1.transAxes, 
                    fontsize=10, color='green', weight='bold', va='top')
    
    # 2. TOP RIGHT: Physics Metrics (Power prediction quality)
    ax2 = plt.subplot(2, 2, 2)
    physics_epochs = [e for i, e in enumerate(epochs) 
                     if i < len(training_history['quick_power_mse']) and training_history['quick_power_mse'][i] is not None]
    quick_power_mse = [val for val in training_history['quick_power_mse'] if val is not None]
    quick_power_mae = [val for val in training_history['quick_power_mae'] if val is not None]
    
    if physics_epochs and quick_power_mse:
        ax2_twin = ax2.twinx()  # Second y-axis for MAE
        
        line1 = ax2.plot(physics_epochs, quick_power_mse, 'purple', marker='o', linewidth=2.5, 
                        markersize=4, label='Power MSE')
        line2 = ax2_twin.plot(physics_epochs, quick_power_mae, 'orange', marker='^', linewidth=2.5, 
                             markersize=4, label='Power MAE')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Power MSE', fontsize=12, color='purple')
        ax2_twin.set_ylabel('Power MAE', fontsize=12, color='orange')
        ax2.set_title('[PHYSICS] Power Metrics (Validation)', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
        
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2.set_xlim(1, max(epochs))
    else:
        ax2.text(0.5, 0.5, 'Physics metrics\nnot yet available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12, style='italic')
        ax2.set_title('[PHYSICS] Power Metrics (Validation)', fontsize=14, fontweight='bold')
    
    # 3. BOTTOM LEFT: Loss Components (Shape vs Amplitude breakdown)
    ax3 = plt.subplot(2, 2, 3)
    val_loss_shape = [val for val in training_history['val_loss_shape'] if val is not None]
    val_loss_amplitude = [val for val in training_history['val_loss_amplitude'] if val is not None]
    
    if val_loss_shape and len(epochs) >= len(val_loss_shape):
        component_epochs = epochs[:len(val_loss_shape)]
        ax3.plot(component_epochs, val_loss_shape, 'teal', label='Shape Loss', linewidth=2.5, marker='D', markersize=3)
        ax3.plot(component_epochs, val_loss_amplitude, 'crimson', label='Amplitude Loss', linewidth=2.5, marker='v', markersize=3)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss Component', fontsize=12)
        ax3.set_title('[COMPONENTS] Loss Components (Validation)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, max(epochs))
        
        # Add balance indicator
        if len(val_loss_shape) >= 2 and len(val_loss_amplitude) >= 2:
            shape_ratio = val_loss_shape[-1] / (val_loss_shape[-1] + val_loss_amplitude[-1])
            balance_text = f'Shape: {shape_ratio:.1%}, Amp: {1-shape_ratio:.1%}'
            ax3.text(0.02, 0.02, balance_text, transform=ax3.transAxes, 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    else:
        ax3.text(0.5, 0.5, 'Loss components\nnot yet available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12, style='italic')
        ax3.set_title('[COMPONENTS] Loss Components (Validation)', fontsize=14, fontweight='bold')
    
    # 4. BOTTOM RIGHT: Comprehensive Physics Metrics + Summary
    ax4 = plt.subplot(2, 2, 4)
    
    # Check if comprehensive physics metrics are available
    comprehensive_epochs = [e for i, e in enumerate(epochs) 
                           if i < len(training_history.get('val_weighted_mse', [])) 
                           and training_history.get('val_weighted_mse', [])[i] is not None]
    
    if comprehensive_epochs and training_history.get('val_weighted_mse'):
        # Plot comprehensive metrics
        val_weighted_mse = [val for val in training_history.get('val_weighted_mse', []) if val is not None]
        val_beam_error = [val for val in training_history.get('val_beam_pointing_error_deg', []) if val is not None]
        val_fss = [val for val in training_history.get('val_fss', []) if val is not None]
        
        # Use twin axes for different scales
        ax4_twin = ax4.twinx()
        
        if val_weighted_mse:
            line1 = ax4.plot(comprehensive_epochs, val_weighted_mse, 'purple', marker='o', 
                           linewidth=2.5, markersize=4, label='Weighted MSE')
        
        if val_beam_error:
            line2 = ax4_twin.plot(comprehensive_epochs, val_beam_error, 'red', marker='^', 
                                linewidth=2.5, markersize=4, label='Beam Error (°)')
            
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Weighted MSE', fontsize=12, color='purple')
        ax4_twin.set_ylabel('Beam Error (degrees)', fontsize=12, color='red')
        ax4.set_title('[PHYSICS] Comprehensive Metrics', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4.set_xlim(1, max(epochs))
        
        # Add comprehensive metrics summary
        if val_weighted_mse and val_beam_error and val_fss:
            summary_text = f"""Physics Metrics:
Weighted MSE: {val_weighted_mse[-1]:.3f}
Beam Error: {val_beam_error[-1]:.1f}°
FSS Score: {val_fss[-1]:.3f}
Polarization: {training_history.get('val_polarization_correlation', [0])[-1]:.3f}"""
            
            ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9))
    else:
        # Fallback to basic coefficient MSE if comprehensive metrics not available
        ax4.plot(epochs, training_history['val_coeff_mse'], 'g-', label='Coefficient MSE', 
                 linewidth=2.5, marker='*', markersize=5)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Coefficient MSE', fontsize=12)
        ax4.set_title('[SUMMARY] Coefficient MSE (Basic)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, max(epochs))
        
        # Basic training summary
        current_epoch = epochs[-1]
        train_loss = training_history['train_loss'][-1]
        val_loss = training_history['val_loss'][-1]
        coeff_mse = training_history['val_coeff_mse'][-1]
        
        summary_text = f"""Training Progress:
Epoch: {current_epoch}
Train Loss: {train_loss:.4f}
Val Loss: {val_loss:.4f}
Coeff MSE: {coeff_mse:.4f}"""
        
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    # Main title for the entire figure
    fig.suptitle(f'TRAINING DASHBOARD - Epoch {epoch}', fontsize=18, fontweight='bold', y=0.96)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # Leave space for main title
    
    # Save the comprehensive plot
    comprehensive_plot_path = artifact_dir / f"training_dashboard_epoch_{epoch:04d}.png"
    plt.savefig(comprehensive_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    plot_files.append(str(comprehensive_plot_path))
    
    return plot_files


def _run_mlp(
    config: MlpConfig,
    X_theta: np.ndarray,
    X_phi: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    meta_path: Path,
    x_theta_path: Path,
    x_phi_path: Path,
    y_true_path: Path,
    y_proj_path: Path,
) -> dict[str, float]:
    """
    Core MLP training loop.

    Y is used only for:
      - fitting y_mean / y_std (denormalisation before the frozen decoder)
      - post-training coefficient MSE evaluation

    The training loss is shape + amplitude on power (see _physics_power_loss_batch).
    Y is never used as a coefficient supervision target inside the loop.
    """
    device = torch.device(config.device)
    basis = load_or_build_basis(config.maxorder)
    n_modes = config.n_modes

    # ------------------------------------------------------------------  
    # Preprocessing: PCA with optional memory optimization
    # ------------------------------------------------------------------
    
    # Memory usage estimation and optimization decision
    n_samples, n_features = len(X_theta), X_theta.shape[1] * 2
    n_train = len(train_idx)
    
    estimated_concat_gb = n_samples * n_features * 4 / (1024**3)
    estimated_train_gb = n_train * n_features * 4 / (1024**3)
    estimated_batch_gb = config.batch_size * 4 * n_features * 4 / (1024**3)  # 4x batch for stats
    
    print(f"Dataset size: {n_samples:,} samples × {n_features:,} features")
    print(f"Training set: {n_train:,} samples")
    print(f"Memory estimates:")
    print(f"  Full concatenation: {estimated_concat_gb:.1f} GB")
    print(f"  Training set only: {estimated_train_gb:.1f} GB") 
    print(f"  Batched approach: {estimated_batch_gb:.3f} GB")
    
    if config.use_memory_efficient_pca and estimated_concat_gb > 1.0:
        print("🚀 Using memory-efficient PCA (avoiding concatenation)")
        
        # 🚀 Batched normalization statistics to avoid memory spikes
        print("Computing normalization statistics (batched)...")
        
        n_train = len(train_idx)
        n_features_theta = X_theta.shape[1] 
        n_features_phi = X_phi.shape[1]
        n_features_y = Y.shape[1]
        
        # Initialize accumulators for online mean/std computation
        theta_sum = np.zeros(n_features_theta, dtype=np.float64)
        phi_sum = np.zeros(n_features_phi, dtype=np.float64)
        y_sum = np.zeros(n_features_y, dtype=np.float64)
        
        theta_sum_sq = np.zeros(n_features_theta, dtype=np.float64)
        phi_sum_sq = np.zeros(n_features_phi, dtype=np.float64)
        y_sum_sq = np.zeros(n_features_y, dtype=np.float64)
        
        # Compute statistics in batches to avoid loading full training set
        batch_size_stats = min(config.batch_size * 4, 1000)  # Larger batches for efficiency
        n_batches = (n_train + batch_size_stats - 1) // batch_size_stats
        
        print(f"Processing {n_train:,} training samples in {n_batches} batches...")
        
        for i, batch in enumerate(_iter_batches(train_idx, batch_size_stats)):
            if (i + 1) % max(1, n_batches // 10) == 0:
                print(f"  Statistics batch {i+1}/{n_batches} ({100*(i+1)/n_batches:.0f}%)")
            theta_batch = X_theta[batch].astype(np.float64)
            phi_batch = X_phi[batch].astype(np.float64)
            y_batch = Y[batch].astype(np.float64)
            
            theta_sum += theta_batch.sum(axis=0)
            phi_sum += phi_batch.sum(axis=0)
            y_sum += y_batch.sum(axis=0)
            
            theta_sum_sq += (theta_batch ** 2).sum(axis=0)
            phi_sum_sq += (phi_batch ** 2).sum(axis=0)
            y_sum_sq += (y_batch ** 2).sum(axis=0)
        
        # Compute mean and std from sums
        theta_mean = (theta_sum / n_train).astype(np.float32)
        phi_mean = (phi_sum / n_train).astype(np.float32)
        y_mean = (y_sum / n_train).astype(np.float32)
        
        theta_var = (theta_sum_sq / n_train - theta_mean.astype(np.float64) ** 2)
        phi_var = (phi_sum_sq / n_train - phi_mean.astype(np.float64) ** 2)
        y_var = (y_sum_sq / n_train - y_mean.astype(np.float64) ** 2)
        
        theta_std = np.sqrt(np.maximum(theta_var, 0)).astype(np.float32)
        phi_std = np.sqrt(np.maximum(phi_var, 0)).astype(np.float32)  
        y_std = np.sqrt(np.maximum(y_var, 0)).astype(np.float32)
        
        # Prevent division by zero
        theta_std[theta_std < 1e-6] = 1.0
        phi_std[phi_std < 1e-6] = 1.0
        y_std[y_std < 1e-6] = 1.0
        
        x_mean = np.concatenate([theta_mean, phi_mean])
        x_std = np.concatenate([theta_std, phi_std])
        
        norms = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
        
        y_train_np = ((Y[train_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
        y_val_np = ((Y[val_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
        y_test_np = ((Y[test_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)

        # Memory-efficient PCA: work directly with split theta/phi arrays
        # Use smaller batch size for PCA fitting to be extra conservative
        pca_fit_batch_size = min(max(config.batch_size // 4, 16), 64)
        print(f"Fitting PCA (memory-efficient, batch_size={pca_fit_batch_size})...")
        pca = _randomized_pca_fit_split(
            X_theta, X_phi, train_idx,
            config.pca_components, config.pca_oversample, config.pca_iterations,
            pca_fit_batch_size,
        )
        
        # 🚀 Ultra-conservative batch sizes for PCA transform to avoid memory spikes
        if config.ultra_conservative:
            transform_batch_size = 2  # Maximum memory safety
            print("🛡️  ULTRA-CONSERVATIVE mode: Using batch size = 2")
        elif config.transform_batch_size > 0:
            transform_batch_size = config.transform_batch_size
        else:
            # Auto: extremely small batch size for very large datasets
            if n_samples > 100000:  # For datasets > 100k samples, use tiny batches
                transform_batch_size = min(max(config.batch_size // 32, 4), 8)  # Extremely conservative
            else:
                transform_batch_size = min(max(config.batch_size // 8, 8), 16)  # Standard conservative
        
        print("Transforming data splits...")
        print(f"Using transform batch size: {transform_batch_size} (vs training batch size: {config.batch_size})")
        
        # Estimate memory for each transform
        samples_per_split = [len(train_idx), len(val_idx), len(test_idx)]
        transform_mb = transform_batch_size * n_features * 4 / (1024**2)  # MB per batch
        
        print(f"Transform memory per batch: {transform_mb:.1f} MB")
        print(f"Split sizes: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}")
        
        # Transform training set with progress
        n_train_batches = (len(train_idx) + transform_batch_size - 1) // transform_batch_size
        print(f"Training transform: {len(train_idx):,} samples in {n_train_batches} batches...")
        z_train = _pca_transform_split_with_progress(
            X_theta, X_phi, train_idx, pca, transform_batch_size, "Training"
        ).astype(np.float32)
        
        # Memory cleanup after training transform
        gc.collect()
        print("Training transform complete, memory cleaned.")
        
        # Transform validation set with same conservative approach
        print(f"Validation transform: {len(val_idx):,} samples...")
        z_val = _pca_transform_split_with_progress(
            X_theta, X_phi, val_idx, pca, transform_batch_size, "Validation"
        ).astype(np.float32)
        
        # Memory cleanup after validation transform
        gc.collect()
        print("Validation transform complete, memory cleaned.")
        
        # Transform test set with same conservative approach  
        print(f"Test transform: {len(test_idx):,} samples...")
        z_test = _pca_transform_split_with_progress(
            X_theta, X_phi, test_idx, pca, transform_batch_size, "Test"
        ).astype(np.float32)
        
        # Final memory cleanup
        gc.collect()
        print("All transforms complete!")
        
    else:
        print("📦 Using standard PCA (with concatenation)")
        # Standard preprocessing (backward compatibility)
        all_idx = np.arange(len(X_theta))
        X = _concat_inputs(X_theta, X_phi, all_idx, config.batch_size)
        
        norms = _fit_normalization(X, Y, train_idx)
        y_train_np = ((Y[train_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
        y_val_np = ((Y[val_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
        y_test_np = ((Y[test_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)

        pca = _randomized_pca_fit(
            X, train_idx,
            config.pca_components, config.pca_oversample, config.pca_iterations,
            config.batch_size,
        )
        # Use smaller batch sizes for transforms to avoid memory spikes
        if config.ultra_conservative:
            transform_batch_size = 2  # Maximum memory safety
            print("🛡️  ULTRA-CONSERVATIVE mode: Using batch size = 2")
        elif config.transform_batch_size > 0:
            transform_batch_size = config.transform_batch_size
        else:
            # Auto: extremely small batch size for very large datasets
            if n_samples > 100000:  # For datasets > 100k samples, use tiny batches
                transform_batch_size = min(max(config.batch_size // 32, 4), 8)  # Extremely conservative
            else:
                transform_batch_size = min(max(config.batch_size // 8, 8), 16)  # Standard conservative
        
        print(f"Using transform batch size: {transform_batch_size}")
        print(f"Training transform: {len(train_idx):,} samples...")
        z_train = _pca_transform(X, train_idx, pca, transform_batch_size).astype(np.float32)
        gc.collect()
        
        print(f"Validation transform: {len(val_idx):,} samples...")
        z_val = _pca_transform(X, val_idx, pca, transform_batch_size).astype(np.float32)
        gc.collect()
        
        print(f"Test transform: {len(test_idx):,} samples...")
        z_test = _pca_transform(X, test_idx, pca, transform_batch_size).astype(np.float32)
        gc.collect()
        print("All transforms complete!")

    # ------------------------------------------------------------------
    # Model Setup
    # ------------------------------------------------------------------
    print("🧠 Creating neural network model...")
    model = _PhysicsMLP(
        in_dim=z_train.shape[1],
        out_dim=y_train_np.shape[1],
        hidden_size=config.hidden_size,
        n_hidden_layers=config.n_hidden_layers,
        dropout=config.dropout,
    ).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("⚙️ Creating optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("📊 Setting up physics constants...")
    # Denormalization constants for the frozen decoder.
    y_mean_t = torch.from_numpy(norms["y_mean"]).to(device)
    y_std_t = torch.from_numpy(norms["y_std"]).to(device)

    # Area weights: w_i = sin(θ_i) / Σ sin(θ), summing to 1.
    sin_theta_t = torch.from_numpy(basis["sin_theta"].astype(np.float32)).to(device)
    w = sin_theta_t / sin_theta_t.sum()  # (n_points,)

    print("🚀 FULLY BATCHED TRAINING - MEMORY OPTIMIZED")
    print("=" * 60)
    print("✅ No large tensor pre-computation (X_theta + X_phi)")
    print("✅ Batched training data loading on-the-fly")
    print("✅ Batched validation evaluation")
    print("✅ Batched final inference and metrics")
    print("✅ Maximum memory usage = batch_size × n_features")
    print("=" * 60)
    
    # Store data indices and arrays for batched access (no full tensor creation)
    # We'll load data in batches during training to avoid memory issues
    print(f"Training set: {len(train_idx):,} samples")
    print(f"Validation set: {len(val_idx):,} samples") 
    print(f"Test set: {len(test_idx):,} samples")
    print(f"Batch size: {config.batch_size}")
    
    # Calculate memory usage per batch instead of total
    memory_per_batch_gb = config.batch_size * X_theta.shape[1] * 4 / (1024**3)
    print(f"Max memory per batch: {memory_per_batch_gb:.2f} GB (vs {len(train_idx) * X_theta.shape[1] * 4 / (1024**3):.1f} GB for full dataset)")
    
    # Only store PCA-transformed data (already compressed to manageable size)
    print(f"Moving PCA data to {device} (compressed: {z_train.shape[1]} features)...")
    z_train_t = torch.from_numpy(z_train).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)  # Validation for periodic evaluation
    print("🎯 PCA data ready for training!")

    # ------------------------------------------------------------------
    # 🚀 Fully Batched Training Loop (No Large Tensor Creation)
    # ------------------------------------------------------------------
    
    print("🏃 Starting fully batched training...")
    print(f"Training batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    
    # 📊 Initialize training history for plotting
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_coeff_mse': [],
        'val_loss_shape': [],
        'val_loss_amplitude': [],
        'quick_power_mse': [],
        'quick_power_mae': [],
        'quick_power_rel_err': []
    }
    
    # Create artifacts directory for plots
    artifact_dir = Path(f"artifacts/training_plots_{int(time.time())}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Training plots will be saved to: {artifact_dir}")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Create shuffled indices for this epoch
        perm = torch.randperm(len(train_idx), device=device)
        
        for s in range(0, len(train_idx), config.batch_size):
            batch_end = min(s + config.batch_size, len(train_idx))
            batch_indices = perm[s:batch_end]
            actual_batch_idx = train_idx[batch_indices.cpu().numpy()]
            
            # Load batch data on-the-fly (no pre-computed tensors)
            zb = z_train_t[batch_indices]  # PCA features for this batch
            
            # Compute power for this batch only
            theta_batch = torch.from_numpy(X_theta[actual_batch_idx].astype(np.float32)).to(device)
            phi_batch = torch.from_numpy(X_phi[actual_batch_idx].astype(np.float32)).to(device)
            pb_raw = theta_batch + phi_batch  # Raw power for this batch
            
            # Normalize power for this batch
            pb_total = (pb_raw * w).sum(dim=1, keepdim=True).clamp(min=1e-6)
            pb_norm = pb_raw / pb_total  # Normalized power for this batch

            # Forward pass
            y_pred = model(zb)
            y_pred_denorm = y_pred * y_std_t + y_mean_t
            p_pred = _decode_power_torch(y_pred_denorm, basis, n_modes=n_modes)

            # Loss computation
            loss, loss_shape, loss_amplitude = _physics_power_loss_batch(
                p_pred, pb_norm, pb_raw, w, config.amplitude_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        
        # 🔥 VALIDATION EVALUATION AT CONFIGURED FREQUENCY
        if (epoch + 1) % config.val_log_frequency == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                # 🚀 Fully batched validation evaluation (no large tensors)
                y_pred_val_parts = []  # Collect predictions in parts
                
                for s in range(0, len(val_idx), config.batch_size):
                    batch_end = min(s + config.batch_size, len(val_idx))
                    batch_val_idx = val_idx[s:batch_end]
                    
                    # Load validation batch on-the-fly
                    zb_val = z_val_t[s:batch_end]
                    theta_val_batch = torch.from_numpy(X_theta[batch_val_idx].astype(np.float32)).to(device)
                    phi_val_batch = torch.from_numpy(X_phi[batch_val_idx].astype(np.float32)).to(device)
                    pb_val_raw = theta_val_batch + phi_val_batch
                    
                    # Normalize validation power for this batch
                    pb_val_total = (pb_val_raw * w).sum(dim=1, keepdim=True).clamp(min=1e-6)
                    pb_val_norm = pb_val_raw / pb_val_total
                    
                    # Forward pass on validation batch
                    y_pred_val = model(zb_val)
                    y_pred_val_denorm = y_pred_val * y_std_t + y_mean_t
                    p_pred_val = _decode_power_torch(y_pred_val_denorm, basis, n_modes=n_modes)
                    
                    v_loss, v_loss_shape, v_loss_amplitude = _physics_power_loss_batch(
                        p_pred_val, pb_val_norm, pb_val_raw, w, config.amplitude_loss_weight
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
                    
                    # Collect predictions for coefficient MSE (in CPU memory)
                    y_pred_val_parts.append(y_pred_val.cpu().numpy())
                
                # Compute coefficient MSE from collected predictions
                y_pred_val_full = np.concatenate(y_pred_val_parts, axis=0)
                val_coeff_mse = float(np.mean((y_val_np - y_pred_val_full) ** 2))
                
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            
            # 📊 LOG TO MLFLOW DURING TRAINING
            mlflow_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss, 
                "val_coeff_mse": val_coeff_mse,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            
            # Add loss components if we have them from the last validation batch
            if val_batches > 0:
                mlflow_metrics.update({
                    "val_loss_shape": v_loss_shape.item() if 'v_loss_shape' in locals() else 0.0,
                    "val_loss_amplitude": v_loss_amplitude.item() if 'v_loss_amplitude' in locals() else 0.0,
                })
            
            # 🔬 COMPREHENSIVE PHYSICS METRICS DURING TRAINING AT CONFIGURED FREQUENCY 
            if (epoch + 1) % config.detailed_metrics_frequency == 0 or epoch == 0:
                print(f"📊 Computing comprehensive physics metrics (epoch {epoch + 1})...")
                
                y_pred_denorm = y_pred_val_full * norms["y_std"] + norms["y_mean"]
                
                # 🚀 Memory-efficient sampling for comprehensive metrics during training
                max_detailed_samples = 500  # More samples for comprehensive metrics
                n_val = len(val_idx)
                
                if n_val > max_detailed_samples:
                    # Sample representative subset for detailed metrics
                    detailed_indices = np.random.choice(n_val, max_detailed_samples, replace=False)
                    detailed_val_idx = val_idx[detailed_indices]
                    y_pred_detailed = y_pred_denorm[detailed_indices]
                    print(f"  Computing metrics on {len(detailed_indices):,} validation samples (subset)...")
                else:
                    # Small validation set - use all samples
                    detailed_val_idx = val_idx
                    y_pred_detailed = y_pred_denorm
                    print(f"  Computing metrics on full validation set ({n_val:,} samples)...")
                
                # Reconstruct physics quantities from predictions
                a_e_pred, a_m_pred = unpack_coeffs(y_pred_detailed, n_modes=n_modes)
                p_pred_detailed = _reconstruct_power_batch(a_e_pred, a_m_pred, basis)
                e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_pred, a_m_pred, basis)
                
                # Load true data for selected samples
                x_theta_detailed = X_theta[detailed_val_idx].astype(np.float32)
                x_phi_detailed = X_phi[detailed_val_idx].astype(np.float32)
                p_true_detailed = x_theta_detailed + x_phi_detailed
                
                # Compute comprehensive physics metrics
                comprehensive_metrics = _metrics(
                    p_true_detailed, p_pred_detailed, basis["sin_theta"],
                    x_theta_detailed, x_phi_detailed, e_theta_pred, e_phi_pred
                )
                
                # Add validation prefix to distinguish from final test metrics
                validation_metrics = {f"val_{k}": v for k, v in comprehensive_metrics.items()}
                mlflow_metrics.update(validation_metrics)
                
                print(f"  ✅ Comprehensive metrics computed: {len(comprehensive_metrics)} physics metrics")
                
                # Also compute basic quick metrics for consistency
                quick_metrics = _compute_quick_val_metrics(
                    Y[detailed_val_idx], y_pred_detailed, x_theta_detailed, x_phi_detailed, basis, n_modes
                )
                mlflow_metrics.update(quick_metrics)
            
            log_basic_metrics(mlflow_metrics)
            
            # 📊 UPDATE TRAINING HISTORY FOR PLOTS
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_coeff_mse'].append(val_coeff_mse)
            training_history['val_loss_shape'].append(
                v_loss_shape.item() if 'v_loss_shape' in locals() else None
            )
            training_history['val_loss_amplitude'].append(
                v_loss_amplitude.item() if 'v_loss_amplitude' in locals() else None
            )
            
            # Add physics metrics if they were computed
            if 'quick_metrics' in locals():
                training_history['quick_power_mse'].append(quick_metrics.get('quick_power_mse'))
                training_history['quick_power_mae'].append(quick_metrics.get('quick_power_mae'))
                training_history['quick_power_rel_err'].append(quick_metrics.get('quick_power_rel_err'))
            else:
                training_history['quick_power_mse'].append(None)
                training_history['quick_power_mae'].append(None)
                training_history['quick_power_rel_err'].append(None)
            
            # Add comprehensive physics metrics if they were computed
            if 'comprehensive_metrics' in locals():
                # Store key comprehensive metrics for plotting
                training_history.setdefault('val_weighted_mse', []).append(comprehensive_metrics.get('weighted_mse'))
                training_history.setdefault('val_beam_pointing_error_deg', []).append(comprehensive_metrics.get('beam_pointing_error_deg'))
                training_history.setdefault('val_polarization_correlation', []).append(comprehensive_metrics.get('polarization_correlation'))
                training_history.setdefault('val_fss', []).append(comprehensive_metrics.get('fss'))
            else:
                # Pad with None when comprehensive metrics weren't computed this epoch
                training_history.setdefault('val_weighted_mse', []).append(None)
                training_history.setdefault('val_beam_pointing_error_deg', []).append(None)
                training_history.setdefault('val_polarization_correlation', []).append(None)
                training_history.setdefault('val_fss', []).append(None)
            
            # 📊 CREATE AND LOG TRAINING PLOTS AT DETAILED METRICS FREQUENCY
            if (epoch + 1) % config.detailed_metrics_frequency == 0 or epoch == config.epochs - 1:
                print("📊 Generating training progress plots...")
                try:
                    print(f"  Creating plots for epoch {epoch + 1} in {artifact_dir}...")
                    plot_files = _create_training_plots(training_history, epoch + 1, artifact_dir)
                    print(f"  Created {len(plot_files)} plot files: {plot_files}")
                    
                    # Log plots to MLflow as artifacts
                    logged_count = 0
                    if plot_files:
                        try:
                            # Convert string paths to Path objects
                            plot_paths = [Path(p) for p in plot_files if Path(p).exists()]
                            print(f"    Logging {len(plot_paths)} plots to MLflow...")
                            
                            if plot_paths:
                                log_images(plot_paths, artifact_subdir="training_plots")
                                logged_count = len(plot_paths)
                                print(f"    ✅ Successfully logged all {logged_count} plots")
                                
                                # List the plots that were logged
                                for plot_path in plot_paths:
                                    print(f"      - {plot_path.name}")
                            else:
                                print(f"    ⚠️ No valid plot files found")
                        except Exception as plot_log_error:
                            print(f"    ❌ Error logging plots: {plot_log_error}")
                        
                    print(f"✅ Logged {logged_count}/{len(plot_files)} training plots to MLflow")
                    
                    # Clean up old plot files to save disk space (keep last 3 sets)
                    try:
                        all_plots = list(artifact_dir.glob("*_epoch_*.png"))
                        if len(all_plots) > 9:  # 3 plots × 3 epochs
                            # Sort by epoch number and remove oldest
                            epoch_numbers = []
                            for plot_path in all_plots:
                                try:
                                    epoch_num = int(plot_path.stem.split('_epoch_')[1])
                                    epoch_numbers.append((epoch_num, plot_path))
                                except (IndexError, ValueError):
                                    continue
                            
                            epoch_numbers.sort()
                            for _, old_plot_path in epoch_numbers[:-9]:  # Keep last 9 files
                                try:
                                    old_plot_path.unlink()
                                except (FileNotFoundError, OSError):
                                    pass
                    except Exception as cleanup_error:
                        # Don't let cleanup errors affect plotting
                        print(f"⚠️  Warning: Plot cleanup failed: {cleanup_error}")
                                
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create training plots: {e}")
                    # Don't let plotting errors crash training
            
            # Enhanced progress display
            if (epoch + 1) % config.detailed_metrics_frequency == 0 and 'quick_metrics' in locals():
                print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, "
                      f"Val Coeff MSE={val_coeff_mse:.4f}, "
                      f"Quick Power MSE={quick_metrics['quick_power_mse']:.4f}")
            else:
                print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, "
                      f"Val Coeff MSE={val_coeff_mse:.4f}")
            
            model.train()  # Back to training mode

    # ------------------------------------------------------------------
    # 🚀 Fully Batched Final Inference & Evaluation
    # ------------------------------------------------------------------
    print("🧠 Running final evaluation (fully batched)...")
    model.eval()

    def infer_batched(z_np: np.ndarray, split_name: str = "Data") -> np.ndarray:
        """Batched inference to avoid loading large arrays into GPU memory."""
        eval_batch_size = min(config.batch_size, 32)  # Conservative batch size for evaluation
        n_samples = len(z_np)
        n_eval_batches = (n_samples + eval_batch_size - 1) // eval_batch_size
        
        print(f"  {split_name}: {n_samples:,} samples in {n_eval_batches} batches...")
        
        parts = []
        with torch.no_grad():
            for s in range(0, n_samples, eval_batch_size):
                batch_end = min(s + eval_batch_size, n_samples)
                z_batch = z_np[s:batch_end]
                z_batch_t = torch.from_numpy(z_batch).to(device)
                pred_batch = model(z_batch_t).cpu().numpy()
                parts.append(pred_batch)
                
                # Progress for large sets
                if n_eval_batches > 100 and (s // eval_batch_size + 1) % (n_eval_batches // 10) == 0:
                    progress = 100 * (s + len(z_batch)) / n_samples
                    print(f"    {split_name} inference: {progress:.0f}%")
                    
        return np.concatenate(parts, axis=0)

    print("Running inference on all splits...")
    yhat_train = infer_batched(z_train, "Training")
    yhat_val = infer_batched(z_val, "Validation") 
    yhat_test = infer_batched(z_test, "Test")
    print("✅ All inference complete!")

    # Coefficient-space MSE (post-training diagnostic only — not the training objective).
    coeff_mse_train = float(np.mean((y_train_np - yhat_train) ** 2))
    coeff_mse_val = float(np.mean((y_val_np - yhat_val) ** 2))
    coeff_mse_test = float(np.mean((y_test_np - yhat_test) ** 2))

    # 🚀 Fully Batched Physics-space metrics computation on test set
    print("📊 Computing physics metrics (fully batched)...")
    yhat_test_denorm = yhat_test * norms["y_std"] + norms["y_mean"]
    a_e_hat, a_m_hat = unpack_coeffs(yhat_test_denorm, n_modes=n_modes)
    p_pred_np = _reconstruct_power_batch(a_e_hat, a_m_hat, basis)
    
    # 🚀 MEMORY-SAFE FINAL EVALUATION
    n_test = len(test_idx)
    
    # For very large datasets, compute metrics on a representative subset
    if n_test > 5000:
        print(f"⚠️  Large test set ({n_test:,} samples) - using representative subset for detailed metrics")
        # Use stratified sampling to get representative subset
        subset_size = 2000
        subset_indices = np.random.choice(n_test, size=min(subset_size, n_test), replace=False)
        
        # Apply subset to predictions
        p_pred_subset = p_pred_np[subset_indices]
        yhat_test_subset = yhat_test_denorm[subset_indices]
        test_idx_subset = test_idx[subset_indices]
        
        print(f"Computing metrics on {len(subset_indices):,} representative samples...")
        
        # Reconstruct polarization for subset only
        a_e_subset, a_m_subset = unpack_coeffs(yhat_test_subset, n_modes=n_modes)
        e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_subset, a_m_subset, basis)
        
        # Load true data for subset only
        p_true_subset = (X_theta[test_idx_subset] + X_phi[test_idx_subset]).astype(np.float32)
        e_theta_true = X_theta[test_idx_subset].astype(np.float32)
        e_phi_true = X_phi[test_idx_subset].astype(np.float32)
        
        print("Computing final physics metrics on subset...")
        p_metrics = _metrics(
            p_true_subset, p_pred_subset, basis["sin_theta"],
            e_theta_true, e_phi_true, e_theta_pred, e_phi_pred
        )
        
        # Add a note about subset evaluation
        p_metrics["evaluation_note"] = f"Computed on {len(subset_indices):,} representative samples (subset)"
        
    else:
        print(f"Computing full metrics on {n_test:,} test samples...")
        
        # Reconstruct polarization components for detailed analysis
        e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_hat, a_m_hat, basis)
        
        # 🚀 Batched loading of true data to avoid memory issues
        print(f"Loading true test data in batches ({len(test_idx):,} samples)...")
        eval_batch_size = min(500, len(test_idx))  # More conservative batch size
        n_eval_batches = (len(test_idx) + eval_batch_size - 1) // eval_batch_size
        
        p_true_parts = []
        e_theta_true_parts = []
        e_phi_true_parts = []
        
        for s in range(0, len(test_idx), eval_batch_size):
            batch_end = min(s + eval_batch_size, len(test_idx))
            batch_test_idx = test_idx[s:batch_end]
            
            # Load batch of true data
            theta_batch = X_theta[batch_test_idx].astype(np.float32)
            phi_batch = X_phi[batch_test_idx].astype(np.float32)
            p_batch = theta_batch + phi_batch
            
            p_true_parts.append(p_batch)
            e_theta_true_parts.append(theta_batch)
            e_phi_true_parts.append(phi_batch)
            
            if n_eval_batches > 10 and (s // eval_batch_size + 1) % (n_eval_batches // 5) == 0:
                progress = 100 * (s + batch_end - s) / len(test_idx)
                print(f"  Loading true test data: {progress:.0f}%")
        
        p_true_np = np.concatenate(p_true_parts, axis=0)
        e_theta_true = np.concatenate(e_theta_true_parts, axis=0)
        e_phi_true = np.concatenate(e_phi_true_parts, axis=0)
        
        print("Computing final physics metrics...")
        p_metrics = _metrics(
            p_true_np, p_pred_np, basis["sin_theta"],
            e_theta_true, e_phi_true, e_theta_pred, e_phi_pred
        )
        
        # Add a note about full evaluation
        p_metrics["evaluation_note"] = f"Computed on full test set ({n_test:,} samples)"
    
    print("✅ Physics metrics complete!")

    metrics = {
        "coeff_mse_train": coeff_mse_train,
        "coeff_mse_val": coeff_mse_val,
        "coeff_mse_test": coeff_mse_test,
        **p_metrics,
    }
    log_basic_metrics(metrics)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    artifact_dir = _artifact_dir(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Detailed coefficient validation for test set
    Y_test_true = Y[test_idx]  # True generating coefficients
    a_e_true, a_m_true = unpack_coeffs(Y_test_true, n_modes=n_modes)
    mode_list = _mode_list(config.maxorder)
    
    coeff_validation = coefficient_validation_summary(
        a_e_true, a_m_true, a_e_hat, a_m_hat, mode_list,
        n_sample_previews=min(10, len(test_idx)),
        output_dir=artifact_dir / "coefficient_analysis"
    )
    
    # Add coefficient validation metrics to the main metrics
    metrics.update({
        'coeff_mag_error_rms': coeff_validation['table_stats']['mag_error_rms'],
        'coeff_phase_error_rms_deg': coeff_validation['table_stats']['phase_error_rms_deg'],
        'coeff_complex_error_rms': coeff_validation['table_stats']['complex_error_rms'],
    })

    torch.save(model.state_dict(), artifact_dir / "model.pt")
    np.savez_compressed(
        artifact_dir / "preprocess.npz",
        pca_mean=pca["mean"],
        pca_components=pca["components"],
        y_mean=norms["y_mean"],
        y_std=norms["y_std"],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": config.__dict__,
                "model": "PhysicsMLP",
                "architecture": {
                    "in_dim": int(z_train.shape[1]),
                    "hidden_size": config.hidden_size,
                    "n_hidden_layers": config.n_hidden_layers,
                    "dropout": config.dropout,
                    "out_dim": int(y_train_np.shape[1]),
                },
                "dataset_x_theta": str(x_theta_path),
                "dataset_x_phi": str(x_phi_path),
                "dataset_y_true": str(y_true_path),
                "dataset_y_proj": str(y_proj_path),
                "dataset_meta": str(meta_path),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _log_validation_images(
        config,  # type: ignore[arg-type]
        X_theta, X_phi, Y, test_idx, yhat_test_denorm, artifact_dir,
    )

    return metrics
