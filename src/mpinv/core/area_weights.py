"""Spherical area weights ``mu(theta) = sin(theta) * d_theta * d_phi``.

These weights appear inside the inner products of presentation/ch1_full.md §1.3 and
inside the data-fidelity norm of §1.8 (``||.||_w``).
"""

from __future__ import annotations

import numpy as np
import torch

from mpinv.core.grid import GRID_DEFAULT, GridSpec


def area_weights(grid: GridSpec = GRID_DEFAULT) -> np.ndarray:
    """Per-pixel spherical area element on the (n_theta, n_phi) grid.

    Returns a ``(n_theta,)`` 1-D vector since the weight depends only on theta;
    callers broadcast it across the phi axis as needed.
    """
    theta = grid.theta_axis()
    return np.sin(theta) * grid.dtheta * grid.dphi


def normalised_area_weights(grid: GridSpec = GRID_DEFAULT) -> np.ndarray:
    """Area weights rescaled to mean 1, matching the practice in losses/physics_power.

    Normalising to mean 1 keeps the loss magnitude comparable across grids.
    """
    w = area_weights(grid)
    w_full = np.broadcast_to(w[:, None], (grid.n_theta, grid.n_phi))
    return w_full / w_full.mean()


def torch_area_weights(
    grid: GridSpec = GRID_DEFAULT,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    normalised: bool = True,
) -> torch.Tensor:
    """Same as :func:`area_weights` but returns a torch tensor of shape ``(n_theta, n_phi)``."""
    w = normalised_area_weights(grid) if normalised else area_weights(grid)
    if w.ndim == 1:
        w = np.broadcast_to(w[:, None], (grid.n_theta, grid.n_phi))
    return torch.as_tensor(np.ascontiguousarray(w), dtype=dtype, device=device)
