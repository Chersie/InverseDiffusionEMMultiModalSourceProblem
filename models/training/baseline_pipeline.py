from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from models.analysis.coefficient_comparison import coefficient_validation_summary
from models.evaluation.metrics import compute_all as _compute_all
from models.tracking.mlflow_utils import log_basic_metrics, log_images, set_tag, start_run
from models.visualization.plot_fields import save_sample_preview
from src.common.paths import (
    CHERSIE_DIR,
    DATA_ML_DATASETS_DIR,
    DATA_ML_FEATURES_DIR,
    DATA_ML_SPLITS_DIR,
    MODELS_ARTIFACTS_DIR,
)
from src.pipeline.generate_fields import _build_output_grid, _load_fast_module


@dataclass(frozen=True)
class BaselineConfig:
    n_samples: int = 10_000
    maxorder: int = 15
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    ridge_alpha: float = 1.0
    pca_components: int = 256
    pca_oversample: int = 16
    pca_iterations: int = 0
    batch_size: int = 64
    rebuild_dataset: bool = False
    trainer: str = "ridge"
    epochs: int = 40
    learning_rate: float = 1e-3
    coeff_loss_weight: float = 1.0
    power_loss_weight: float = 0.1
    # Physics / MLP trainer: weight on log-total-power term (see _physics_power_loss_batch).
    amplitude_loss_weight: float = 1.0
    device: str = "cpu"

    @property
    def n_modes(self) -> int:
        return self.maxorder * (self.maxorder + 2)

    @property
    def n_targets(self) -> int:
        return 4 * self.n_modes


def _mode_list(maxorder: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(1, maxorder + 1) for m in range(-l, l + 1)]


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for s in range(0, len(indices), batch_size):
        yield indices[s : s + batch_size]


def pack_coeffs(a_e: np.ndarray, a_m: np.ndarray) -> np.ndarray:
    """Pack complex coeff arrays `(n_samples, n_modes)` into real matrix `(n_samples, 4*n_modes)`."""
    return np.concatenate([a_e.real, a_e.imag, a_m.real, a_m.imag], axis=1).astype(np.float32)


def unpack_coeffs(y: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_coeffs`."""
    a_e = y[:, :n_modes] + 1j * y[:, n_modes : 2 * n_modes]
    a_m = y[:, 2 * n_modes : 3 * n_modes] + 1j * y[:, 3 * n_modes : 4 * n_modes]
    return a_e.astype(np.complex64), a_m.astype(np.complex64)


def _coeffs_from_latin_sample(
    maxorder: int, sample_id: int, seed: int, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic per-sample latin-square-style coefficients.
    Returns arrays `(n_modes,)` for electric and magnetic complex coefficients.
    """
    n = 2 * maxorder + 1
    mode_pairs = _mode_list(maxorder)
    n_modes = len(mode_pairs)

    rng = np.random.default_rng(seed + sample_id * 104729)
    seed_e_re = int(rng.integers(0, n))
    seed_e_im = int(rng.integers(0, n))
    seed_m_re = int(rng.integers(0, n))
    seed_m_im = int(rng.integers(0, n))
    row_perm_e = rng.permutation(n)
    col_perm_e = rng.permutation(n)
    row_perm_m = rng.permutation(n)
    col_perm_m = rng.permutation(n)

    def latin_value(row: int, col: int, shift: int) -> float:
        v = (col + row + shift) % n
        return scale * (2.0 * (v + 1) / (n + 1) - 1.0)

    a_e = np.zeros(n_modes, dtype=np.complex64)
    a_m = np.zeros(n_modes, dtype=np.complex64)
    for k, (l, m) in enumerate(mode_pairs):
        row = l - 1
        col = m + maxorder
        row_e = int(row_perm_e[row])
        col_e = int(col_perm_e[col])
        row_m = int(row_perm_m[row])
        col_m = int(col_perm_m[col])
        a_e[k] = latin_value(row_e, col_e, seed_e_re) + 1j * latin_value(row_e, col_e, seed_e_im)
        a_m[k] = latin_value(row_m, col_m, seed_m_re) + 1j * latin_value(row_m, col_m, seed_m_im)
    return a_e, a_m


def _basis_cache_path(maxorder: int) -> Path:
    return DATA_ML_FEATURES_DIR / f"basis_L{maxorder}.npz"


def load_or_build_basis(maxorder: int) -> dict[str, np.ndarray]:
    """
    Load or compute multipole basis tensors on canonical grid.
    Returns:
      - e_theta, e_phi, m_theta, m_phi: shape (n_modes, n_points), complex64
      - sin_theta: shape (n_points,), float32
    """
    DATA_ML_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _basis_cache_path(maxorder)
    if cache_path.exists():
        data = np.load(cache_path)
        return {
            "e_theta": data["e_theta"],
            "e_phi": data["e_phi"],
            "m_theta": data["m_theta"],
            "m_phi": data["m_phi"],
            "sin_theta": data["sin_theta"],
        }

    mp_module = _load_fast_module()
    size_phi = 360
    size_theta = 179
    theta_2d, phi_2d = _build_output_grid(
        type("Grid", (), {"size_phi": size_phi, "size_theta": size_theta})()
    )
    n_points = size_phi * size_theta
    modes = _mode_list(maxorder)
    n_modes = len(modes)

    e_theta = np.zeros((n_modes, n_points), dtype=np.complex64)
    e_phi = np.zeros((n_modes, n_points), dtype=np.complex64)
    m_theta = np.zeros((n_modes, n_points), dtype=np.complex64)
    m_phi = np.zeros((n_modes, n_points), dtype=np.complex64)
    sin_theta = np.sin(theta_2d).reshape(-1).astype(np.float32)

    for k, (l, m) in enumerate(modes):
        e_mode = mp_module.field_for_multipole(l, m, theta_2d, phi_2d, electric=True)
        m_mode = mp_module.field_for_multipole(l, m, theta_2d, phi_2d, electric=False)
        e_theta[k] = e_mode[..., 0].reshape(-1).astype(np.complex64)
        e_phi[k] = e_mode[..., 1].reshape(-1).astype(np.complex64)
        m_theta[k] = m_mode[..., 0].reshape(-1).astype(np.complex64)
        m_phi[k] = m_mode[..., 1].reshape(-1).astype(np.complex64)

    np.savez_compressed(
        cache_path,
        e_theta=e_theta,
        e_phi=e_phi,
        m_theta=m_theta,
        m_phi=m_phi,
        sin_theta=sin_theta,
    )
    return {
        "e_theta": e_theta,
        "e_phi": e_phi,
        "m_theta": m_theta,
        "m_phi": m_phi,
        "sin_theta": sin_theta,
    }


def _dataset_dir(config: BaselineConfig) -> Path:
    return DATA_ML_DATASETS_DIR / f"baseline_L{config.maxorder}_N{config.n_samples}_seed{config.seed}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(config: BaselineConfig) -> tuple[Path, Path, Path, Path, Path]:
    """
    Build the dataset:
      - X_power_theta.npy : (N, n_points) float32  |E_theta|^2 — model input component 1
      - X_power_phi.npy   : (N, n_points) float32  |E_phi|^2   — model input component 2
      - Y_coeff_true.npy  : (N, 4*n_modes) float32 true generating coefficients (TRAINING TARGET)
      - Y_coeff_proj.npy  : (N, 4*n_modes) float32 analytic inner-product coefficients (REFERENCE)
      - meta.json

    Model input is the concatenation of X_power_theta and X_power_phi along axis=1,
    giving shape (N, 2*n_points = 128880). Both components are phase-free intensities,
    consistent with a dual-polarisation power-only measurement.

    Returns (x_theta_path, x_phi_path, y_true_path, y_proj_path, meta_path).
    """
    out_dir = _dataset_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_theta_path = out_dir / "X_power_theta.npy"
    x_phi_path = out_dir / "X_power_phi.npy"
    y_true_path = out_dir / "Y_coeff_true.npy"
    y_proj_path = out_dir / "Y_coeff_proj.npy"
    meta_path = out_dir / "meta.json"

    all_exist = (
        x_theta_path.exists()
        and x_phi_path.exists()
        and y_true_path.exists()
        and y_proj_path.exists()
        and meta_path.exists()
    )
    if all_exist and not config.rebuild_dataset:
        return x_theta_path, x_phi_path, y_true_path, y_proj_path, meta_path

    basis = load_or_build_basis(config.maxorder)
    e_theta_b = basis["e_theta"]   # (n_modes, n_points)
    e_phi_b = basis["e_phi"]
    m_theta_b = basis["m_theta"]
    m_phi_b = basis["m_phi"]
    sin_theta = basis["sin_theta"]
    n_points = e_theta_b.shape[1]
    n_modes = e_theta_b.shape[0]
    d_omega = (np.pi / 180.0) ** 2

    x_theta_mm = np.lib.format.open_memmap(
        x_theta_path, mode="w+", dtype=np.float32, shape=(config.n_samples, n_points)
    )
    x_phi_mm = np.lib.format.open_memmap(
        x_phi_path, mode="w+", dtype=np.float32, shape=(config.n_samples, n_points)
    )
    y_true_mm = np.lib.format.open_memmap(
        y_true_path, mode="w+", dtype=np.float32, shape=(config.n_samples, 4 * n_modes)
    )
    y_proj_mm = np.lib.format.open_memmap(
        y_proj_path, mode="w+", dtype=np.float32, shape=(config.n_samples, 4 * n_modes)
    )

    # Pre-compute weighted conjugated basis for inner-product projection.
    proj_e_theta = np.conj(e_theta_b) * sin_theta[np.newaxis, :]
    proj_e_phi = np.conj(e_phi_b) * sin_theta[np.newaxis, :]
    proj_m_theta = np.conj(m_theta_b) * sin_theta[np.newaxis, :]
    proj_m_phi = np.conj(m_phi_b) * sin_theta[np.newaxis, :]

    for i in range(config.n_samples):
        a_e_true, a_m_true = _coeffs_from_latin_sample(
            config.maxorder, sample_id=i, seed=config.seed
        )
        e_ut_theta = a_e_true @ e_theta_b + a_m_true @ m_theta_b  # (n_points,) complex
        e_ut_phi = a_e_true @ e_phi_b + a_m_true @ m_phi_b

        # Per-polarisation intensities — the model inputs.
        x_theta_mm[i] = np.abs(e_ut_theta).astype(np.float32) ** 2
        x_phi_mm[i] = np.abs(e_ut_phi).astype(np.float32) ** 2

        # True generating coefficients — the training target.
        y_true_mm[i] = pack_coeffs(a_e_true[np.newaxis, :], a_m_true[np.newaxis, :])[0]

        # Projected coefficients — analytic recovery from complex field (reference only).
        a_e_proj = (proj_e_theta @ e_ut_theta + proj_e_phi @ e_ut_phi) * d_omega
        a_m_proj = (proj_m_theta @ e_ut_theta + proj_m_phi @ e_ut_phi) * d_omega
        y_proj_mm[i] = pack_coeffs(a_e_proj[np.newaxis, :], a_m_proj[np.newaxis, :])[0]

    meta = {
        "n_samples": config.n_samples,
        "maxorder": config.maxorder,
        "n_modes": int(n_modes),
        "n_points": int(n_points),
        "grid": {"size_phi": 360, "size_theta": 179},
        "d_omega": d_omega,
        "seed": config.seed,
        "basis_cache": str(_basis_cache_path(config.maxorder)),
        "coeff_packing": "Re(a_E) | Im(a_E) | Re(a_M) | Im(a_M), modes ordered l=1..L m=-l..l",
        "model_input": "concatenate(X_power_theta, X_power_phi) along axis=1 → shape (N, 2*n_points)",
        "files": {
            "X_power_theta": "|E_theta|^2, shape (N, n_points), phi-outer theta-inner — model input 1",
            "X_power_phi": "|E_phi|^2, shape (N, n_points), phi-outer theta-inner — model input 2",
            "Y_coeff_true": "True generating coefficients — TRAINING TARGET",
            "Y_coeff_proj": "Analytic inner-product coefficients from complex E_UT — REFERENCE ONLY",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return x_theta_path, x_phi_path, y_true_path, y_proj_path, meta_path


# ---------------------------------------------------------------------------
# MLflow dataset logging
# ---------------------------------------------------------------------------

def log_dataset_to_mlflow(
    config: BaselineConfig,
    x_theta_path: Path,
    x_phi_path: Path,
    y_true_path: Path,
    y_proj_path: Path,
    meta_path: Path,
    val_idx: "np.ndarray | None" = None,
    test_idx: "np.ndarray | None" = None,
    n_preview: int = 16,
) -> None:
    """
    Open a dedicated MLflow run (type=dataset) and log:
      - params  : dataset configuration
      - metrics : power and coefficient statistics including coeff_proj_vs_true_mse
      - artifacts: meta.json + n_preview sample preview images from val/test splits
    """
    basis = load_or_build_basis(config.maxorder)
    n_modes = config.n_modes

    X_theta = np.load(x_theta_path, mmap_mode="r")
    X_phi = np.load(x_phi_path, mmap_mode="r")
    Y_true = np.load(y_true_path, mmap_mode="r")
    Y_proj = np.load(y_proj_path, mmap_mode="r")

    sample_rows = min(1000, len(X_theta))
    stat_idx = np.linspace(0, len(X_theta) - 1, sample_rows, dtype=int)
    x_th_s = X_theta[stat_idx].astype(np.float64)
    x_ph_s = X_phi[stat_idx].astype(np.float64)
    yt_s = Y_true[stat_idx].astype(np.float64)
    yp_s = Y_proj[stat_idx].astype(np.float64)

    p_total = x_th_s + x_ph_s
    stats: dict[str, float] = {
        "power_total_mean": float(p_total.mean()),
        "power_total_std": float(p_total.std()),
        "power_theta_fraction_mean": float((x_th_s / (p_total + 1e-30)).mean()),
        "coeff_true_norm_mean": float(np.linalg.norm(yt_s, axis=1).mean()),
        "coeff_proj_norm_mean": float(np.linalg.norm(yp_s, axis=1).mean()),
        "coeff_proj_vs_true_mse": float(np.mean((yt_s - yp_s) ** 2)),
    }

    # Prefer val then test for preview; fall back to evenly-spaced full set.
    if val_idx is not None and len(val_idx) > 0:
        combined = [(int(i), "val") for i in val_idx]
    else:
        combined = []
    if test_idx is not None and len(test_idx) > 0:
        combined += [(int(i), "test") for i in test_idx]
    if not combined:
        combined = [(int(i), "") for i in np.linspace(0, len(X_theta) - 1, n_preview, dtype=int)]

    step = max(1, len(combined) // n_preview)
    preview_pool = combined[::step][:n_preview]

    preview_dir = _dataset_dir(config) / "preview"
    if preview_dir.exists():
        for old in preview_dir.glob("*.png"):
            old.unlink()
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_paths: list[Path] = []
    for idx_i, split_name in preview_pool:
        a_e_true, a_m_true = unpack_coeffs(Y_true[[idx_i]], n_modes=n_modes)
        a_e_proj, a_m_proj = unpack_coeffs(Y_proj[[idx_i]], n_modes=n_modes)
        p_display = (X_theta[idx_i] + X_phi[idx_i]).astype(np.float32)
        paths = save_sample_preview(
            sample_idx=idx_i,
            p_true=p_display,
            a_e_true=a_e_true[0],
            a_m_true=a_m_true[0],
            a_e_ref=a_e_proj[0],
            a_m_ref=a_m_proj[0],
            basis=basis,
            out_dir=preview_dir,
            ref_label="proj",
            split_label=split_name,
        )
        preview_paths.extend(paths)

    dataset_name = f"dataset_{_dataset_dir(config).name}"
    with start_run(dataset_name, params={
        "maxorder": config.maxorder,
        "n_samples": config.n_samples,
        "seed": config.seed,
        "n_modes": config.n_modes,
        "n_points": 360 * 179,
        "model_input_width": 2 * 360 * 179,
        "n_targets": config.n_targets,
        "coeff_packing": "Re_E|Im_E|Re_M|Im_M",
    }):
        set_tag("type", "dataset")
        log_basic_metrics(stats)
        log_images([meta_path], artifact_subdir="")
        log_images(preview_paths, artifact_subdir="preview")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _save_splits(
    config: BaselineConfig, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    DATA_ML_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = (
        DATA_ML_SPLITS_DIR
        / f"baseline_L{config.maxorder}_N{config.n_samples}_seed{config.seed}.npz"
    )
    if split_path.exists():
        s = np.load(split_path)
        return s["train"], s["val"], s["test"]

    rng = np.random.default_rng(config.seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(config.train_ratio * n_samples)
    n_val = int(config.val_ratio * n_samples)
    if n_samples >= 3:
        n_train = min(max(n_train, 1), n_samples - 2)
        n_val = min(max(n_val, 1), n_samples - n_train - 1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Helpers shared by trainers
# ---------------------------------------------------------------------------

def _concat_inputs(
    X_theta: np.ndarray, X_phi: np.ndarray, idx: np.ndarray, batch_size: int
) -> np.ndarray:
    """Return (|idx|, 2*n_points) float32 by concatenating the two intensity arrays."""
    parts: list[np.ndarray] = []
    for batch in _iter_batches(idx, batch_size):
        parts.append(
            np.concatenate([X_theta[batch].astype(np.float32),
                            X_phi[batch].astype(np.float32)], axis=1)
        )
    return np.concatenate(parts, axis=0)


def _fit_normalization(
    X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray
) -> dict[str, np.ndarray]:
    x_mean = X[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = X[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = Y[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = Y[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _randomized_pca_fit_split(
    X_theta: np.ndarray,
    X_phi: np.ndarray, 
    idx: np.ndarray,
    n_components: int,
    oversample: int,
    n_iter: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """
    Memory-efficient randomized PCA for split theta/phi data.
    Avoids creating large concatenated arrays by processing theta and phi separately.
    """
    # Compute mean from batched data to avoid memory spike
    print("  PCA: Computing means...")
    theta_sum = np.zeros(X_theta.shape[1], dtype=np.float64)
    phi_sum = np.zeros(X_phi.shape[1], dtype=np.float64)
    n_samples = len(idx)
    
    for batch in _iter_batches(idx, batch_size):
        theta_sum += X_theta[batch].sum(axis=0, dtype=np.float64)
        phi_sum += X_phi[batch].sum(axis=0, dtype=np.float64)
    
    theta_mean = (theta_sum / n_samples).astype(np.float32)
    phi_mean = (phi_sum / n_samples).astype(np.float32)
    x_mean = np.concatenate([theta_mean, phi_mean])
    
    print("  PCA: Initializing random projections...")
    n_features = X_theta.shape[1] + X_phi.shape[1]
    l_dim = min(n_components + oversample, n_features)
    rng = np.random.default_rng(12345)
    omega = rng.standard_normal((n_features, l_dim), dtype=np.float32)
    omega_theta = omega[:X_theta.shape[1]]
    omega_phi = omega[X_theta.shape[1]:]

    def _sketch_split(om_theta: np.ndarray, om_phi: np.ndarray) -> np.ndarray:
        """Sketch using split theta/phi data without concatenation."""
        out = np.zeros((len(idx), om_theta.shape[1]), dtype=np.float32)
        row = 0
        for batch in _iter_batches(idx, batch_size):
            # Process theta and phi separately, then combine results
            theta_centered = X_theta[batch].astype(np.float32) - theta_mean
            phi_centered = X_phi[batch].astype(np.float32) - phi_mean
            
            theta_contrib = theta_centered @ om_theta
            phi_contrib = phi_centered @ om_phi
            
            out[row : row + len(batch)] = theta_contrib + phi_contrib
            row += len(batch)
        return out

    print("  PCA: Initial sketch and QR decomposition...")
    Q, _ = np.linalg.qr(_sketch_split(omega_theta, omega_phi), mode="reduced")
    
    print(f"  PCA: Power iterations (n_iter={n_iter})...")
    for iter_i in range(n_iter):
        if n_iter > 0:
            print(f"    Iteration {iter_i+1}/{n_iter}")
        Z = np.zeros((n_features, Q.shape[1]), dtype=np.float32)
        row = 0
        for batch in _iter_batches(idx, batch_size):
            theta_centered = X_theta[batch].astype(np.float32) - theta_mean
            phi_centered = X_phi[batch].astype(np.float32) - phi_mean
            
            # Split Z update
            Z[:X_theta.shape[1]] += theta_centered.T @ Q[row : row + len(batch)]
            Z[X_theta.shape[1]:] += phi_centered.T @ Q[row : row + len(batch)]
            row += len(batch)
            
        Q, _ = np.linalg.qr(_sketch_split(Z[:X_theta.shape[1]], Z[X_theta.shape[1]:]), mode="reduced")

    print("  PCA: Final projection...")
    B = np.zeros((Q.shape[1], n_features), dtype=np.float32)
    row = 0
    for batch in _iter_batches(idx, batch_size):
        theta_centered = X_theta[batch].astype(np.float32) - theta_mean
        phi_centered = X_phi[batch].astype(np.float32) - phi_mean
        
        # Split B update
        B[:, :X_theta.shape[1]] += Q[row : row + len(batch)].T @ theta_centered
        B[:, X_theta.shape[1]:] += Q[row : row + len(batch)].T @ phi_centered
        row += len(batch)

    print("  PCA: SVD decomposition...")
    _, S, Vt = np.linalg.svd(B, full_matrices=False)
    print(f"  PCA: Complete! Extracted {n_components} components.")
    return {
        "mean": x_mean,
        "components": Vt[:n_components].astype(np.float32),
        "singular_values": S[:n_components].astype(np.float32),
    }


def _randomized_pca_fit(
    X: np.ndarray,
    idx: np.ndarray,
    n_components: int,
    oversample: int,
    n_iter: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """
    Legacy randomized PCA for concatenated data (backward compatibility).
    For memory efficiency with large datasets, use _randomized_pca_fit_split instead.
    """
    x_mean = X[idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    n_features = X.shape[1]
    l_dim = min(n_components + oversample, n_features)
    rng = np.random.default_rng(12345)
    omega = rng.standard_normal((n_features, l_dim), dtype=np.float32)

    def _sketch(om: np.ndarray) -> np.ndarray:
        out = np.zeros((len(idx), om.shape[1]), dtype=np.float32)
        row = 0
        for batch in _iter_batches(idx, batch_size):
            xb = X[batch].astype(np.float32) - x_mean
            out[row : row + len(batch)] = xb @ om
            row += len(batch)
        return out

    Q, _ = np.linalg.qr(_sketch(omega), mode="reduced")
    for _ in range(n_iter):
        Z = np.zeros((n_features, Q.shape[1]), dtype=np.float32)
        row = 0
        for batch in _iter_batches(idx, batch_size):
            xb = X[batch].astype(np.float32) - x_mean
            Z += xb.T @ Q[row : row + len(batch)]
            row += len(batch)
        Q, _ = np.linalg.qr(_sketch(Z), mode="reduced")

    B = np.zeros((Q.shape[1], n_features), dtype=np.float32)
    row = 0
    for batch in _iter_batches(idx, batch_size):
        xb = X[batch].astype(np.float32) - x_mean
        B += Q[row : row + len(batch)].T @ xb
        row += len(batch)

    _, S, Vt = np.linalg.svd(B, full_matrices=False)
    return {
        "mean": x_mean,
        "components": Vt[:n_components].astype(np.float32),
        "singular_values": S[:n_components].astype(np.float32),
    }


def _pca_transform_split(
    X_theta: np.ndarray,
    X_phi: np.ndarray, 
    idx: np.ndarray, 
    pca: dict[str, np.ndarray], 
    batch_size: int
) -> np.ndarray:
    """
    Memory-efficient PCA transform for split theta/phi data.
    Avoids creating large concatenated arrays.
    """
    out = np.zeros((len(idx), pca["components"].shape[0]), dtype=np.float32)
    
    # Split mean and components
    theta_mean = pca["mean"][:X_theta.shape[1]]
    phi_mean = pca["mean"][X_theta.shape[1]:]
    theta_components = pca["components"][:, :X_theta.shape[1]]
    phi_components = pca["components"][:, X_theta.shape[1]:]
    
    row = 0
    for batch in _iter_batches(idx, batch_size):
        theta_centered = X_theta[batch].astype(np.float32) - theta_mean
        phi_centered = X_phi[batch].astype(np.float32) - phi_mean
        
        # Transform each part and combine
        theta_proj = theta_centered @ theta_components.T
        phi_proj = phi_centered @ phi_components.T
        
        out[row : row + len(batch)] = theta_proj + phi_proj
        row += len(batch)
    return out


def _pca_transform(
    X: np.ndarray, idx: np.ndarray, pca: dict[str, np.ndarray], batch_size: int
) -> np.ndarray:
    """Legacy PCA transform for concatenated data (backward compatibility).""" 
    out = np.zeros((len(idx), pca["components"].shape[0]), dtype=np.float32)
    row = 0
    for batch in _iter_batches(idx, batch_size):
        xb = X[batch].astype(np.float32) - pca["mean"]
        out[row : row + len(batch)] = xb @ pca["components"].T
        row += len(batch)
    return out


def _fit_ridge_multioutput(Xz: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = Xz.T @ Xz
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    xty = Xz.T @ Y
    return np.linalg.solve(xtx + reg, xty).astype(np.float32)


def _predict_ridge(Xz: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (Xz @ W).astype(np.float32)


def _reconstruct_power_batch(
    a_e: np.ndarray, a_m: np.ndarray, basis: dict[str, np.ndarray]
) -> np.ndarray:
    ehat_theta = a_e @ basis["e_theta"] + a_m @ basis["m_theta"]
    ehat_phi = a_e @ basis["e_phi"] + a_m @ basis["m_phi"]
    return (np.abs(ehat_theta) ** 2 + np.abs(ehat_phi) ** 2).astype(np.float32)


def _reconstruct_polarization_batch(
    a_e: np.ndarray, a_m: np.ndarray, basis: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct |E_θ|² and |E_φ|² components separately.
    
    Returns
    -------
    e_theta_power, e_phi_power : (N, n_points) float arrays
        Polarization component powers
    """
    ehat_theta = a_e @ basis["e_theta"] + a_m @ basis["m_theta"]
    ehat_phi = a_e @ basis["e_phi"] + a_m @ basis["m_phi"]
    return (
        np.abs(ehat_theta) ** 2,
        np.abs(ehat_phi) ** 2
    )


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
    
    Parameters
    ----------
    p_true, p_pred : (N, n_points) float arrays
        Total power patterns
    sin_theta : (n_points,) float array
        Area weighting factors
    e_theta_true, e_phi_true : (N, n_points) float arrays, optional
        True |E_θ|² and |E_φ|² components
    e_theta_pred, e_phi_pred : (N, n_points) float arrays, optional
        Predicted |E_θ|² and |E_φ|² components
        
    Returns
    -------
    dict
        All available metrics
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


def _physics_power_loss_batch(
    p_pred: "torch.Tensor",
    p_true_norm: "torch.Tensor",
    p_true_raw: "torch.Tensor",
    w: "torch.Tensor",
    amplitude_loss_weight: float,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Combined physics loss on the sphere.

    *Shape* (beam pattern): area-weighted MSE after per-sample normalisation
    so polar angle density is corrected and overall scale is factored out.

    *Amplitude*: mean squared error of log(P), where P = Σ_i w_i p_i is the
    same area-weighted total power as in the dataset (observable from |E_θ|²+|E_φ|²).

    Returns (loss_total, loss_shape, loss_amp).
    """
    P_true = (p_true_raw * w).sum(dim=1).clamp(min=1e-12)
    P_pred = (p_pred * w).sum(dim=1).clamp(min=1e-12)
    p_pred_norm = p_pred / P_pred.unsqueeze(1)
    loss_shape = ((p_pred_norm - p_true_norm) ** 2 * w).sum(dim=1).mean()
    loss_amp = torch.mean((torch.log(P_pred) - torch.log(P_true)) ** 2)
    loss_total = loss_shape + amplitude_loss_weight * loss_amp
    return loss_total, loss_shape, loss_amp


def _decode_power_torch(
    y_pred_denorm: "torch.Tensor", basis: dict[str, np.ndarray], n_modes: int
) -> "torch.Tensor":
    """Frozen physics decoder: coefficient vector → reconstructed total power."""
    a_e_r = y_pred_denorm[:, :n_modes]
    a_e_i = y_pred_denorm[:, n_modes : 2 * n_modes]
    a_m_r = y_pred_denorm[:, 2 * n_modes : 3 * n_modes]
    a_m_i = y_pred_denorm[:, 3 * n_modes : 4 * n_modes]
    a_e = torch.complex(a_e_r, a_e_i)
    a_m = torch.complex(a_m_r, a_m_i)
    dev = y_pred_denorm.device
    et = torch.from_numpy(basis["e_theta"]).to(dev)
    ep = torch.from_numpy(basis["e_phi"]).to(dev)
    mt = torch.from_numpy(basis["m_theta"]).to(dev)
    mp = torch.from_numpy(basis["m_phi"]).to(dev)
    ehat_theta = a_e @ et + a_m @ mt
    ehat_phi = a_e @ ep + a_m @ mp
    return torch.abs(ehat_theta) ** 2 + torch.abs(ehat_phi) ** 2  # (batch, n_points)


def _log_validation_images(
    config: BaselineConfig,
    X_theta: np.ndarray,
    X_phi: np.ndarray,
    Y_true: np.ndarray,
    test_idx: np.ndarray,
    yhat_test_denorm: np.ndarray,
    artifact_dir: Path,
    n_images: int = 16,
) -> None:
    """Save true-vs-predicted comparison images for test samples and log to MLflow."""
    basis = load_or_build_basis(config.maxorder)
    n_modes = config.n_modes
    val_dir = artifact_dir / "validation"

    preview_indices = np.linspace(0, len(test_idx) - 1, min(n_images, len(test_idx)), dtype=int)
    all_paths: list[Path] = []
    for local_i in preview_indices:
        global_i = int(test_idx[local_i])
        a_e_true, a_m_true = unpack_coeffs(Y_true[[global_i]], n_modes=n_modes)
        a_e_pred, a_m_pred = unpack_coeffs(yhat_test_denorm[[local_i]], n_modes=n_modes)
        p_display = (X_theta[global_i] + X_phi[global_i]).astype(np.float32)
        paths = save_sample_preview(
            sample_idx=global_i,
            p_true=p_display,
            a_e_true=a_e_true[0],
            a_m_true=a_m_true[0],
            a_e_ref=a_e_pred[0],
            a_m_ref=a_m_pred[0],
            basis=basis,
            out_dir=val_dir,
            ref_label="model",
            split_label="test",
            include_difference_maps=True,
        )
        all_paths.extend(paths)
    log_images(all_paths, artifact_subdir="validation")


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def _run_ridge(
    config: BaselineConfig,
    X_theta: np.ndarray,
    X_phi: np.ndarray,
    Y: np.ndarray,           # Y_coeff_true — the training target
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    meta_path: Path,
    x_theta_path: Path,
    x_phi_path: Path,
    y_true_path: Path,
    y_proj_path: Path,
) -> dict[str, float]:
    all_idx = np.arange(len(X_theta))
    X = _concat_inputs(X_theta, X_phi, all_idx, config.batch_size)  # (N, 2*n_points)

    norms = _fit_normalization(X, Y, train_idx)
    y_train = ((Y[train_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
    y_val = ((Y[val_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)
    y_test = ((Y[test_idx] - norms["y_mean"]) / norms["y_std"]).astype(np.float32)

    pca = _randomized_pca_fit(
        X, train_idx, config.pca_components,
        config.pca_oversample, config.pca_iterations, config.batch_size,
    )
    z_train = _pca_transform(X, train_idx, pca, config.batch_size)
    z_val = _pca_transform(X, val_idx, pca, config.batch_size)
    z_test = _pca_transform(X, test_idx, pca, config.batch_size)

    W = _fit_ridge_multioutput(z_train, y_train, alpha=config.ridge_alpha)
    yhat_train = _predict_ridge(z_train, W)
    yhat_val = _predict_ridge(z_val, W)
    yhat_test = _predict_ridge(z_test, W)

    coeff_mse_train = float(np.mean((y_train - yhat_train) ** 2))
    coeff_mse_val = float(np.mean((y_val - yhat_val) ** 2))
    coeff_mse_test = float(np.mean((y_test - yhat_test) ** 2))

    yhat_test_denorm = yhat_test * norms["y_std"] + norms["y_mean"]
    a_e_hat, a_m_hat = unpack_coeffs(yhat_test_denorm, n_modes=config.n_modes)
    _basis_ridge = load_or_build_basis(config.maxorder)
    p_pred = _reconstruct_power_batch(a_e_hat, a_m_hat, _basis_ridge)
    p_true = (X_theta[test_idx] + X_phi[test_idx]).astype(np.float32)
    
    # Reconstruct polarization components for detailed analysis
    e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_hat, a_m_hat, _basis_ridge)
    e_theta_true = X_theta[test_idx].astype(np.float32)
    e_phi_true = X_phi[test_idx].astype(np.float32)
    
    p_metrics = _metrics(
        p_true, p_pred, _basis_ridge["sin_theta"],
        e_theta_true, e_phi_true, e_theta_pred, e_phi_pred
    )

    # Detailed coefficient validation for test set
    Y_test_true = Y[test_idx]  # True generating coefficients
    a_e_true, a_m_true = unpack_coeffs(Y_test_true, n_modes=config.n_modes)
    
    artifact_dir = (
        MODELS_ARTIFACTS_DIR
        / f"baseline_{config.trainer}_L{config.maxorder}_N{config.n_samples}_seed{config.seed}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mode_list = _mode_list(config.maxorder)
    
    coeff_validation = coefficient_validation_summary(
        a_e_true, a_m_true, a_e_hat, a_m_hat, mode_list,
        n_sample_previews=min(10, len(test_idx)),
        output_dir=artifact_dir / "coefficient_analysis"
    )
    
    # Add coefficient validation metrics to the main metrics  
    p_metrics.update({
        'coeff_mag_error_rms': coeff_validation['table_stats']['mag_error_rms'],
        'coeff_phase_error_rms_deg': coeff_validation['table_stats']['phase_error_rms_deg'],
        'coeff_complex_error_rms': coeff_validation['table_stats']['complex_error_rms'],
    })
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "model.npz",
        ridge_W=W,
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
                "dataset_x_theta": str(x_theta_path),
                "dataset_x_phi": str(x_phi_path),
                "dataset_y_true": str(y_true_path),
                "dataset_y_proj": str(y_proj_path),
                "dataset_meta": str(meta_path),
                "metrics": p_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log_validation_images(
        config, X_theta, X_phi, Y, test_idx, yhat_test_denorm, artifact_dir
    )
    return p_metrics


def _run_physics(
    config: BaselineConfig,
    X_theta: np.ndarray,
    X_phi: np.ndarray,
    Y: np.ndarray,           # Y_coeff_true — used for normalisation stats and post-training eval
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
    Train a linear model (PCA → Linear) using area-weighted power loss on the
    sphere: beam *shape* plus *amplitude* of the total radiated power.

    Training objective:
        L = L_shape + λ_amp · L_amp
        L_shape = Σ_j w_j (p̂_norm_j - p_true_norm_j)²   [per batch sample, then mean]
        L_amp   = mean_b ( log P̂_b - log P_b )²
        P_b = Σ_j w_j p_j   [area-weighted power integral, same units as inputs]

    Weights w sum to 1 (pole-corrected).  p_norm = p / P recovers the previous
    scale-free pattern loss when λ_amp = 0.

    The frozen physics decoder (coefficients → power) is the only link between
    the predicted coefficients and the training signal.  No coefficient MSE
    is used — the model discovers coefficients that reproduce observed power.
    """
    if torch is None or nn is None:
        raise RuntimeError(
            "Physics trainer requires PyTorch. Install torch first, "
            "then rerun with --trainer physics."
        )
    device = torch.device(config.device)
    basis = load_or_build_basis(config.maxorder)
    n_modes = config.n_modes

    all_idx = np.arange(len(X_theta))
    X = _concat_inputs(X_theta, X_phi, all_idx, config.batch_size)  # (N, 2*n_points)

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

    model = nn.Linear(z_train.shape[1], y_train_np.shape[1], bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Denormalization tensors: model predicts in normalised coeff space;
    # we denorm before the frozen physics decoder so units are correct.
    y_mean_t = torch.from_numpy(norms["y_mean"]).to(device)
    y_std_t = torch.from_numpy(norms["y_std"]).to(device)

    # Area weights w_i = sin(θ_i) / Σ sin(θ_j) — pole-corrected, sum to 1.
    sin_theta_t = torch.from_numpy(basis["sin_theta"].astype(np.float32)).to(device)
    w = sin_theta_t / sin_theta_t.sum()  # (n_points,)

    # True training power patterns normalised to unit area-weighted total.
    # Normalising makes the loss a beam-shape comparison (scale-independent).
    p_train_np = (X_theta[train_idx] + X_phi[train_idx]).astype(np.float32)
    p_train_t = torch.from_numpy(p_train_np).to(device)
    p_train_total = (p_train_t * w).sum(dim=1, keepdim=True).clamp(min=1e-6)
    p_train_norm_t = p_train_t / p_train_total  # (n_train, n_points)

    z_train_t = torch.from_numpy(z_train).to(device)

    for epoch in range(config.epochs):
        perm = torch.randperm(z_train_t.shape[0], device=device)
        for s in range(0, z_train_t.shape[0], config.batch_size):
            batch = perm[s : s + config.batch_size]
            zb = z_train_t[batch]
            pb_norm = p_train_norm_t[batch]

            y_pred = model(zb)
            y_pred_denorm = y_pred * y_std_t + y_mean_t
            p_pred = _decode_power_torch(y_pred_denorm, basis, n_modes=n_modes)

            # Normalise predicted power the same way as the target.
            pb_raw = p_train_t[batch]
            loss, _, _ = _physics_power_loss_batch(
                p_pred, pb_norm, pb_raw, w, config.amplitude_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    def infer(z_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(z_np).to(device)).cpu().numpy()

    yhat_train = infer(z_train)
    yhat_val = infer(z_val)
    yhat_test = infer(z_test)

    coeff_mse_train = float(np.mean((y_train_np - yhat_train) ** 2))
    coeff_mse_val = float(np.mean((y_val_np - yhat_val) ** 2))
    coeff_mse_test = float(np.mean((y_test_np - yhat_test) ** 2))

    yhat_test_denorm = yhat_test * norms["y_std"] + norms["y_mean"]
    a_e_hat, a_m_hat = unpack_coeffs(yhat_test_denorm, n_modes=n_modes)
    p_pred = _reconstruct_power_batch(a_e_hat, a_m_hat, basis)
    p_true = (X_theta[test_idx] + X_phi[test_idx]).astype(np.float32)
    
    # Reconstruct polarization components for detailed analysis
    e_theta_pred, e_phi_pred = _reconstruct_polarization_batch(a_e_hat, a_m_hat, basis)
    e_theta_true = X_theta[test_idx].astype(np.float32)
    e_phi_true = X_phi[test_idx].astype(np.float32)
    
    p_metrics = _metrics(
        p_true, p_pred, basis["sin_theta"],
        e_theta_true, e_phi_true, e_theta_pred, e_phi_pred
    )

    metrics = {
        "coeff_mse_train": coeff_mse_train,
        "coeff_mse_val": coeff_mse_val,
        "coeff_mse_test": coeff_mse_test,
        **p_metrics,
    }
    log_basic_metrics(metrics)

    artifact_dir = (
        MODELS_ARTIFACTS_DIR
        / f"baseline_{config.trainer}_L{config.maxorder}_N{config.n_samples}_seed{config.seed}"
    )
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
        config, X_theta, X_phi, Y, test_idx, yhat_test_denorm, artifact_dir
    )
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_and_evaluate(config: BaselineConfig) -> dict[str, float]:
    DATA_ML_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    x_theta_path, x_phi_path, y_true_path, y_proj_path, meta_path = build_dataset(config)

    X_theta = np.load(x_theta_path, mmap_mode="r")
    X_phi = np.load(x_phi_path, mmap_mode="r")
    Y_true = np.load(y_true_path, mmap_mode="r")

    train_idx, val_idx, test_idx = _save_splits(config, X_theta.shape[0])
    log_dataset_to_mlflow(
        config,
        x_theta_path, x_phi_path,
        y_true_path, y_proj_path,
        meta_path,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    with start_run(
        "baseline_power_to_multipoles",
        params={
            "n_samples": config.n_samples,
            "maxorder": config.maxorder,
            "seed": config.seed,
            "pca_components": config.pca_components,
            "ridge_alpha": config.ridge_alpha,
            "trainer": config.trainer,
            "model_input": "[X_theta || X_phi], shape (N, 128880)",
            "training_target": "Y_coeff_true",
            "amplitude_loss_weight": config.amplitude_loss_weight,
        },
    ):
        trainer_kwargs = dict(
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
        if config.trainer == "ridge":
            metrics = _run_ridge(**trainer_kwargs)
        elif config.trainer == "physics":
            metrics = _run_physics(**trainer_kwargs)
        else:
            raise ValueError(f"Unsupported trainer: {config.trainer!r}")

    return metrics
