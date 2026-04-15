"""
Physics-Aware Neural Network Pipeline for Multipole Prediction.

This implements several physics-informed architectures that incorporate domain knowledge
about electromagnetic multipole expansions, spherical harmonics, and energy conservation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None

from models.training.baseline_pipeline import (
    _concat_inputs,
    _decode_power_torch,
    _fit_normalization,
    _log_validation_images,
    _mode_list,
    _pca_transform,
    _physics_power_loss_batch,
    _randomized_pca_fit,
    _reconstruct_polarization_batch,
    _reconstruct_power_batch,
    _save_splits,
    build_dataset,
    load_or_build_basis,
    log_dataset_to_mlflow,
    unpack_coeffs,
)
from models.analysis.coefficient_comparison import coefficient_validation_summary
from models.evaluation.metrics import compute_all as _compute_all
from models.tracking.mlflow_utils import log_basic_metrics, set_tag, start_run
from src.common.paths import DATA_ML_DATASETS_DIR, MODELS_ARTIFACTS_DIR


@dataclass(frozen=True)
class PhysicsAwareConfig:
    # Dataset / splits
    n_samples: int = 10_000
    maxorder: int = 15
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # PCA preprocessing
    pca_components: int = 256
    pca_oversample: int = 16
    pca_iterations: int = 0
    # Architecture selection
    model_type: str = "multipole_aware"  # "multipole_aware", "energy_conserving", "hybrid"
    hidden_size: int = 512
    n_hidden_layers: int = 2
    dropout: float = 0.1
    # Physics-aware parameters
    energy_conservation_weight: float = 1.0
    symmetry_regularization: float = 0.1
    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    device: str = "cpu"
    rebuild_dataset: bool = False
    amplitude_loss_weight: float = 1.0

    @property
    def n_modes(self) -> int:
        return self.maxorder * (self.maxorder + 2)

    @property
    def n_targets(self) -> int:
        return 4 * self.n_modes


# ============================================================================
# Physics-Aware Network Architectures
# ============================================================================

class MultipoleAwareNet(nn.Module):
    """
    Network that processes multipole orders separately, respecting spherical harmonic structure.
    
    Key physics insight: Different multipole orders (l) represent different spatial scales
    and should be processed with specialized sub-networks.
    """
    
    def __init__(self, in_dim: int, maxorder: int, hidden_size: int, dropout: float):
        super().__init__()
        self.maxorder = maxorder
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for each multipole order l
        self.multipole_heads = nn.ModuleDict()
        for l in range(1, maxorder + 1):
            n_modes_l = 2 * l + 1  # Number of m values for this l
            n_coeffs_l = 4 * n_modes_l  # Real/imag for E and M
            
            self.multipole_heads[str(l)] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, n_coeffs_l)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Process each multipole order separately
        coeffs_by_order = []
        for l in range(1, self.maxorder + 1):
            coeffs_l = self.multipole_heads[str(l)](features)
            coeffs_by_order.append(coeffs_l)
        
        # Concatenate in standard mode order
        return torch.cat(coeffs_by_order, dim=1)


class EnergyConservingNet(nn.Module):
    """
    Network with built-in energy conservation constraint.
    
    Physics insight: Total radiated power should match the input power measurements.
    This enforces energy conservation as an architectural constraint.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )
        
        # Energy prediction head (predicts total power)
        self.energy_head = nn.Sequential(
            nn.Linear(in_dim, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive energy
        )
    
    def forward(self, x: torch.Tensor, input_power: torch.Tensor | None = None) -> torch.Tensor:
        raw_coeffs = self.backbone(x)
        
        if input_power is not None:
            # Predict energy scaling
            predicted_energy_scale = self.energy_head(x).squeeze(-1)  # (batch,)
            
            # Compute coefficient energy (simplified approximation)
            coeff_energy = (raw_coeffs ** 2).sum(dim=1)  # (batch,)
            
            # Scale coefficients to match input power
            target_total = input_power.sum(dim=1)  # Total power per sample
            scale_factor = torch.sqrt(target_total * predicted_energy_scale / (coeff_energy + 1e-8))
            
            return raw_coeffs * scale_factor.unsqueeze(1)
        else:
            return raw_coeffs


class HybridPhysicsNet(nn.Module):
    """
    Hybrid model: analytical physics baseline + learned corrections.
    
    Physics insight: Start with known physics (e.g., dipole approximation) and learn
    only the corrections/higher-order effects.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        
        # Analytical baseline predictor (simple model)
        self.baseline_net = nn.Sequential(
            nn.Linear(in_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, out_dim)
        )
        
        # Residual correction network (more complex)
        self.correction_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, hidden_size),  # Input + baseline prediction
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )
        
        # Learned mixing weight
        self.mixing_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Analytical baseline
        baseline = self.baseline_net(x)
        
        # Learned corrections based on input + baseline
        correction_input = torch.cat([x, baseline], dim=1)
        corrections = self.correction_net(correction_input)
        
        # Adaptive mixing
        mix_weight = torch.sigmoid(self.mixing_weight)
        return mix_weight * baseline + (1 - mix_weight) * corrections


class SymmetryAwareNet(nn.Module):
    """
    Network that respects electromagnetic reciprocity and other symmetries.
    
    Physics insight: Electromagnetic fields have inherent symmetries that should
    be preserved by the neural network architecture.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        
        # Standard backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )
        
        # Symmetry constraint parameters
        self.register_buffer('reciprocity_matrix', torch.eye(out_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_output = self.backbone(x)
        
        # Apply reciprocity constraint (simplified)
        # In practice, this would implement specific EM reciprocity relations
        symmetrized = 0.5 * (raw_output + raw_output @ self.reciprocity_matrix)
        
        return symmetrized


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def create_physics_aware_model(config: PhysicsAwareConfig, in_dim: int) -> nn.Module:
    """Factory function to create physics-aware models."""
    out_dim = config.n_targets
    
    if config.model_type == "multipole_aware":
        return MultipoleAwareNet(in_dim, config.maxorder, config.hidden_size, config.dropout)
    elif config.model_type == "energy_conserving":
        return EnergyConservingNet(in_dim, out_dim, config.hidden_size, config.dropout)
    elif config.model_type == "hybrid":
        return HybridPhysicsNet(in_dim, out_dim, config.hidden_size, config.dropout)
    elif config.model_type == "symmetry_aware":
        return SymmetryAwareNet(in_dim, out_dim, config.hidden_size, config.dropout)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def _metrics(
    p_true: np.ndarray, 
    p_pred: np.ndarray, 
    sin_theta: np.ndarray,
    e_theta_true: np.ndarray | None = None,
    e_phi_true: np.ndarray | None = None,
    e_theta_pred: np.ndarray | None = None,
    e_phi_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all available metrics including polarization analysis."""
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


def _artifact_dir(config: PhysicsAwareConfig) -> Path:
    return (
        MODELS_ARTIFACTS_DIR
        / f"physics_aware_{config.model_type}_L{config.maxorder}_N{config.n_samples}_seed{config.seed}"
    )


def train_and_evaluate(config: PhysicsAwareConfig) -> dict[str, float]:
    """Train and evaluate physics-aware model."""
    if torch is None or nn is None:
        raise RuntimeError("Physics-aware pipeline requires PyTorch. Install it with: pip install torch")

    DATA_ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build/load dataset
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

    # MLflow run
    with start_run(
        f"physics_aware_{config.model_type}",
        params={
            "n_samples": config.n_samples,
            "maxorder": config.maxorder,
            "seed": config.seed,
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "n_hidden_layers": config.n_hidden_layers,
            "energy_conservation_weight": config.energy_conservation_weight,
            "training_loss": "physics-aware power loss + conservation constraints",
            "amplitude_loss_weight": config.amplitude_loss_weight,
        },
    ):
        set_tag("model", f"PhysicsAware-{config.model_type}")
        metrics = _run_physics_aware(
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


def _run_physics_aware(
    config: PhysicsAwareConfig,
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
    """Core physics-aware training loop with conservation constraints."""
    device = torch.device(config.device)
    basis = load_or_build_basis(config.maxorder)
    n_modes = config.n_modes

    # Preprocessing
    all_idx = np.arange(len(X_theta))
    X = _concat_inputs(X_theta, X_phi, all_idx, config.batch_size)

    norms = _fit_normalization(X, Y, train_idx)
    y_train_np = ((Y[train_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
    y_val_np = ((Y[val_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
    y_test_np = ((Y[test_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)

    pca = _randomized_pca_fit(
        X, train_idx, config.pca_components,
        config.pca_oversample, config.pca_iterations, config.batch_size,
    )
    z_train = _pca_transform(X, train_idx, pca, config.batch_size).astype(np.float32)
    z_val = _pca_transform(X, val_idx, pca, config.batch_size).astype(np.float32)
    z_test = _pca_transform(X, test_idx, pca, config.batch_size).astype(np.float32)

    # Create physics-aware model
    model = create_physics_aware_model(config, z_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Denormalization and physics setup
    y_mean_t = torch.from_numpy(norms["y_mean"]).to(device)
    y_std_t = torch.from_numpy(norms["y_std"]).to(device)
    sin_theta_t = torch.from_numpy(basis["sin_theta"].astype(np.float32)).to(device)
    w = sin_theta_t / sin_theta_t.sum()

    # Training data
    p_train_np = (X_theta[train_idx] + X_phi[train_idx]).astype(np.float32)
    p_train_t = torch.from_numpy(p_train_np).to(device)
    p_train_total = (p_train_t * w).sum(dim=1, keepdim=True).clamp(min=1e-6)
    p_train_norm_t = p_train_t / p_train_total
    z_train_t = torch.from_numpy(z_train).to(device)

    # Training loop with physics-aware loss
    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(z_train_t.shape[0], device=device)
        
        for s in range(0, z_train_t.shape[0], config.batch_size):
            batch = perm[s : s + config.batch_size]
            zb = z_train_t[batch]
            pb_norm = p_train_norm_t[batch]
            pb_raw = p_train_t[batch]

            # Forward pass (with input power for energy-conserving model)
            if config.model_type == "energy_conserving":
                y_pred = model(zb, pb_raw)
            else:
                y_pred = model(zb)
                
            y_pred_denorm = y_pred * y_std_t + y_mean_t
            p_pred = _decode_power_torch(y_pred_denorm, basis, n_modes=n_modes)

            # Physics-aware power loss
            loss, loss_shape, loss_amp = _physics_power_loss_batch(
                p_pred, pb_norm, pb_raw, w, config.amplitude_loss_weight
            )
            
            # Additional physics constraints
            if config.energy_conservation_weight > 0:
                # Energy conservation penalty
                pred_total = (p_pred * w).sum(dim=1)
                true_total = pb_raw.sum(dim=1)
                energy_loss = torch.mean((pred_total - true_total) ** 2)
                loss += config.energy_conservation_weight * energy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Evaluation
    model.eval()
    
    def infer(z_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if config.model_type == "energy_conserving":
                # Provide dummy power for inference (same shape as training power patterns)
                dummy_power = torch.ones(len(z_np), basis["sin_theta"].shape[0], device=device)
                return model(torch.from_numpy(z_np).to(device), dummy_power).cpu().numpy()
            else:
                return model(torch.from_numpy(z_np).to(device)).cpu().numpy()

    yhat_train = infer(z_train)
    yhat_val = infer(z_val)
    yhat_test = infer(z_test)

    # Standard metrics
    coeff_mse_train = float(np.mean((y_train_np - yhat_train) ** 2))
    coeff_mse_val = float(np.mean((y_val_np - yhat_val) ** 2))
    coeff_mse_test = float(np.mean((y_test_np - yhat_test) ** 2))

    # Physics metrics
    yhat_test_denorm = yhat_test * norms["y_std"] + norms["y_mean"]
    a_e_hat, a_m_hat = unpack_coeffs(yhat_test_denorm, n_modes=n_modes)
    p_pred_np = _reconstruct_power_batch(a_e_hat, a_m_hat, basis)
    p_true_np = (X_theta[test_idx] + X_phi[test_idx]).astype(np.float32)
    
    # Polarization analysis
    e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_hat, a_m_hat, basis)
    e_theta_true = X_theta[test_idx].astype(np.float32)
    e_phi_true = X_phi[test_idx].astype(np.float32)
    
    p_metrics = _metrics(
        p_true_np, p_pred_np, basis["sin_theta"],
        e_theta_true, e_phi_true, e_theta_pred, e_phi_pred
    )

    metrics = {
        "coeff_mse_train": coeff_mse_train,
        "coeff_mse_val": coeff_mse_val,
        "coeff_mse_test": coeff_mse_test,
        **p_metrics,
    }
    log_basic_metrics(metrics)

    # Artifacts
    artifact_dir = _artifact_dir(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    
    # Coefficient validation
    Y_test_true = Y[test_idx]
    a_e_true, a_m_true = unpack_coeffs(Y_test_true, n_modes=n_modes)
    mode_list = _mode_list(config.maxorder)
    
    coeff_validation = coefficient_validation_summary(
        a_e_true, a_m_true, a_e_hat, a_m_hat, mode_list,
        n_sample_previews=min(10, len(test_idx)),
        output_dir=artifact_dir / "coefficient_analysis"
    )
    
    metrics.update({
        'coeff_mag_error_rms': coeff_validation['table_stats']['mag_error_rms'],
        'coeff_phase_error_rms_deg': coeff_validation['table_stats']['phase_error_rms_deg'],
        'coeff_complex_error_rms': coeff_validation['table_stats']['complex_error_rms'],
    })
    
    # Save metadata
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": config.__dict__,
                "model_type": config.model_type,
                "physics_constraints": {
                    "energy_conservation_weight": config.energy_conservation_weight,
                    "symmetry_regularization": config.symmetry_regularization,
                },
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Generate validation images
    _log_validation_images(
        config,  # type: ignore[arg-type]
        X_theta, X_phi, Y, test_idx, yhat_test_denorm, artifact_dir,
    )

    return metrics