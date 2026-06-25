"""Composition helpers used by Hydra ``_target_`` directives.

These functions take *plain* Python objects (already instantiated by Hydra) plus a
small number of structural parameters and assemble the framework's higher-order
objects: dataset wrappers, the physics loss (which depends on grid + l_max), and
the in-memory dataset arrays.

The CLI ``train.py`` calls these via ``hydra.utils.instantiate(cfg.x)`` only at
leaves; the compositions between leaves are plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.augment import apply_augmentation, build_augmentation
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig


@dataclass(slots=True)
class DataPipeline:
    """Materialised in-memory training/validation arrays + dataloaders.

    All payloads live on CPU; the Trainer moves them to the configured device.
    """

    grid: GridSpec
    l_max: int
    n_train: int
    n_val: int
    P_train: np.ndarray  # (n_train, n_theta, n_phi)
    packed_train: np.ndarray  # (n_train, 4 K)
    P_val: np.ndarray
    packed_val: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader


class _ArrayDataset(Dataset):
    """Dataset returning ``(features, packed_coeffs, P_pattern)`` triples."""

    def __init__(self, features: np.ndarray, packed: np.ndarray, P: np.ndarray):
        if not (len(features) == len(packed) == len(P)):
            raise ValueError("array lengths disagree")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.packed = torch.as_tensor(packed, dtype=torch.float32)
        self.P = torch.as_tensor(P, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.packed[idx], self.P[idx]


def build_data_pipeline(
    grid,
    l_max: int,
    n_train: int,
    n_val: int,
    seed_train: int,
    seed_val: int,
    generator: dict,
    batch_size: int = 32,
    num_workers: int = 0,
    n_test: int = 0,
    seed_test: int = 9876,
    augmentation: dict | None = None,
    aug_seed: int = 4242,
) -> dict:
    """Build the in-memory synthetic pipeline.

    Returns a *plain dict* (not a dataset object) so the CLI can pull out individual
    arrays and inject them into the feature pipeline before constructing dataloaders.
    Hydra's instantiate magic is therefore only applied to leaves, not to this whole
    composition.

    Optional ``augmentation`` (a Hydra-style dict with at least ``name``) is applied
    to the **train** ``(P, packed)`` pair only, after generation. Validation and
    synthetic-test pairs are kept clean. See :mod:`mpinv.data.augment` for the
    consistency contract of each augmentation.
    """
    grid_obj = GridSpec(
        n_phi=grid["n_phi"],
        n_theta=grid["n_theta"],
        theta_start_deg=grid.get("theta_start_deg", 1.0),
        theta_end_deg=grid.get("theta_end_deg", 179.0),
    )
    try:
        basis = load_basis(grid_obj, l_max)
    except Exception:
        basis = build_basis(grid_obj, l_max)
    cfg = SyntheticGeneratorConfig(
        grid=grid_obj,
        l_max=l_max,
        mode=generator.get("mode", "gaussian"),
        family_balance=generator.get("family_balance", 0.5),
        coef_scale=generator.get("coef_scale", 1.0),
        coef_scale_log_uniform_range=generator.get("coef_scale_log_uniform_range"),
        color_alpha=generator.get("color_alpha", 1.0),
        sparse_active_fraction=generator.get("sparse_active_fraction", 0.1),
        mode_dropout_prob=generator.get("mode_dropout_prob", 0.0),
        family_balance_jitter=generator.get("family_balance_jitter", 0.0),
    )
    gen = SyntheticGenerator(cfg=cfg, basis=basis)

    rng_train = np.random.default_rng(seed_train)
    rng_val = np.random.default_rng(seed_val)
    P_train, packed_train = gen.generate_batch(n_train, rng_train)
    P_val, packed_val = gen.generate_batch(n_val, rng_val)

    aug_cfg = build_augmentation(augmentation) if augmentation else None
    if aug_cfg is not None:
        rng_aug = np.random.default_rng(aug_seed)
        P_train, packed_train = apply_augmentation(
            P_train,
            packed_train,
            cfg=aug_cfg,
            rng=rng_aug,
            basis=basis,
            l_max=l_max,
        )

    out = {
        "grid": grid_obj,
        "l_max": l_max,
        "n_train": n_train,
        "n_val": n_val,
        "P_train": P_train,
        "packed_train": packed_train,
        "P_val": P_val,
        "packed_val": packed_val,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "basis": basis,
        "augmentation": aug_cfg,
    }

    if n_test and n_test > 0:
        rng_test = np.random.default_rng(seed_test)
        P_test, packed_test = gen.generate_batch(n_test, rng_test)
        out["n_test"] = n_test
        out["P_test"] = P_test
        out["packed_test"] = packed_test

    return out


def build_physics_power_loss(
    grid: GridSpec,
    l_max: int,
    log_ratio: bool = False,
    log_eps: float = 1e-12,
    coef_aux_weight: float = 0.0,
    rank_bin_weight: float = 0.0,
    rank_bin_n_bins: int | None = None,
    rank_bin_beta: float = 10.0,
    truncate_target_to_band: int | None = None,
    decoder: DifferentiableMultipoleField | None = None,
) -> PhysicsPowerLoss:
    """Build the physics power loss with a fresh decoder if none is provided."""
    cfg = PhysicsPowerLossConfig(
        log_ratio=log_ratio,
        log_eps=log_eps,
        coef_aux_weight=coef_aux_weight,
        rank_bin_weight=rank_bin_weight,
        rank_bin_n_bins=rank_bin_n_bins,
        rank_bin_beta=rank_bin_beta,
        truncate_target_to_band=truncate_target_to_band,
    )
    return PhysicsPowerLoss(cfg=cfg, grid=grid, l_max=l_max, decoder=decoder)


def make_loaders(
    P_train: np.ndarray,
    packed_train: np.ndarray,
    z_train: np.ndarray,
    P_val: np.ndarray,
    packed_val: np.ndarray,
    z_val: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train / val dataloaders from in-memory arrays."""
    train_ds = _ArrayDataset(z_train, packed_train, P_train)
    val_ds = _ArrayDataset(z_val, packed_val, P_val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader
