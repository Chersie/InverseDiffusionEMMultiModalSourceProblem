"""Soft rank-bin loss for power patterns.

Motivation
----------

The standard ``physics_power`` loss penalises the absolute value mismatch
between the predicted and target power patterns. That couples two
properties: (a) the *shape* of the radiation pattern (which mode is up,
where the nulls and lobes are) and (b) the absolute *magnitude* of the
power. On real-antenna data with very small absolute scales (``||P||~1e-3``)
the shape signal can be dominated by amplitude noise.

This module implements an explicit *rank-only* regulariser:

1. For each sample, sort the pixels of ``P_true`` by value and bin them
   into ``num_bins`` (default ``2 * l_max + 1``) equal-population bins.
   Each pixel gets a bin index in ``{0, 1, …, num_bins-1}``.
2. Do the same for ``P_pred``, using its *own* bin thresholds (so the
   loss is invariant to monotone affine transforms ``P → α·P + β``).
3. Compute MSE between the two per-pixel bin indices.

Differentiability
-----------------

The naive sort-and-bin route uses ``torch.argsort`` which is
non-differentiable (gradient is zero almost everywhere). To preserve
gradients we use the standard *soft binning* trick: for each sample
compute the ``num_bins - 1`` quantile thresholds via ``torch.quantile``
(piecewise-linear, sub-differentiable) and then assign each pixel a
*soft bin index* via

    soft_bin(P_i) = Σ_b σ(β · (P_i - τ_b) / s)

where ``τ_b`` are the thresholds, ``s`` is a per-sample scale equal to the
threshold range, and ``σ`` is the logistic sigmoid. As ``β → ∞`` this
converges to the hard rank-bin index. ``β = 10`` (default) gives a sharp
but smooth assignment with stable gradients.

Properties
----------

* ``rank_bin_mse(P, P) = 0``.
* ``rank_bin_mse(α·P + β, P) ≈ 0`` for any ``α > 0`` and finite ``β``
  (scale and shift invariance — the whole point of the regulariser).
* ``rank_bin_mse(P_perm, P) > 0`` when the model permutes pixel ranks.
* On the project's ``(179, 360)`` grid: O(B · N · num_bins) with
  N = 64 440 and num_bins ≈ 11 — about 45 M float ops per step, comparable
  to the basis-decoder einsum the physics_power loss already runs.

This loss is registered as ``"rank_bin_p"`` and can also be folded into
``physics_power`` via the new ``rank_bin_weight`` config field (see
:mod:`mpinv.losses.physics_power`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX
from mpinv.core.shapes import assert_packed_coeffs, assert_power_pattern
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.registry import register_loss


def _soft_bin_indices(
    P: torch.Tensor,
    n_bins: int,
    beta: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute differentiable per-pixel soft bin indices.

    ``P`` has shape ``(B, n_pixels)``. Returns a tensor of the same shape
    whose entries are continuous values in approximately ``[0, n_bins-1]``
    (sigmoid-soft bin assignments). At the limit ``β → ∞`` these collapse
    onto the hard quantile-bin indices.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    B, n_pixels = P.shape
    # n_bins - 1 quantile thresholds at fractions 1/n_bins, 2/n_bins, ...
    qs = torch.linspace(
        1.0 / n_bins, (n_bins - 1) / n_bins, n_bins - 1,
        device=P.device, dtype=P.dtype,
    )
    # torch.quantile(input, q, dim) returns shape (n_quantiles, *input.shape[no dim])
    # Setting dim=1 gives (n_bins-1, B); transpose to (B, n_bins-1).
    thresh = torch.quantile(P, qs, dim=1).transpose(0, 1)  # (B, n_bins - 1)
    # Per-sample scale: range of thresholds (a robust IQR-like proxy).
    # Adding ``eps`` keeps the sigmoid well-defined when P is constant.
    scale = (thresh[:, -1:] - thresh[:, :1]).abs() + eps  # (B, 1)
    # Broadcast: (B, n_pixels, 1) - (B, 1, n_bins-1) → (B, n_pixels, n_bins-1)
    diffs = P.unsqueeze(-1) - thresh.unsqueeze(1)
    soft_bins = torch.sigmoid(beta * diffs / scale.unsqueeze(-1)).sum(dim=-1)
    return soft_bins


def rank_bin_mse(
    P_pred: torch.Tensor,
    P_true: torch.Tensor,
    n_bins: int,
    beta: float = 10.0,
) -> torch.Tensor:
    """Per-pixel rank-bin MSE on power patterns.

    Parameters
    ----------
    P_pred, P_true : torch.Tensor
        Real tensors of shape ``(B, n_theta, n_phi)``. Shapes must match.
    n_bins : int
        Number of rank bins to use. Common choice: ``2 * l_max + 1``.
    beta : float
        Sigmoid temperature. Higher = sharper bin transitions, lower =
        smoother but less faithful to true ranks. ``β = 10`` is a
        reasonable default; values in ``[5, 50]`` are typical.

    Returns
    -------
    torch.Tensor
        Scalar tensor: ``mean_pixels((soft_bin(P_pred) - soft_bin(P_true))²)``.
    """
    if P_pred.shape != P_true.shape:
        raise ValueError(f"shape mismatch: {P_pred.shape} vs {P_true.shape}")
    if P_pred.ndim != 3:
        raise ValueError(f"expected (B, n_theta, n_phi); got {P_pred.shape}")
    B = P_pred.shape[0]
    P_pred_flat = P_pred.reshape(B, -1)
    P_true_flat = P_true.reshape(B, -1)
    bin_pred = _soft_bin_indices(P_pred_flat, n_bins=n_bins, beta=beta)
    bin_true = _soft_bin_indices(P_true_flat, n_bins=n_bins, beta=beta)
    return ((bin_pred - bin_true) ** 2).mean()


@dataclass(slots=True)
class RankBinPLossConfig:
    """Knobs for :class:`RankBinPLoss`.

    ``n_bins`` defaults to ``2 * l_max + 1`` (matched to the angular
    resolution of order-``l_max`` spherical harmonics) when ``None``.
    """

    n_bins: int | None = None
    beta: float = 10.0
    coef_aux_weight: float = 0.0
    """Optional coef-MSE auxiliary term, mirroring the same field on
    :class:`mpinv.losses.physics_power.PhysicsPowerLossConfig` so this loss
    can also act as a warm-up anchor."""


@register_loss("rank_bin_p")
class RankBinPLoss(nn.Module):
    """Standalone rank-bin loss on power patterns.

    The loss owns its own differentiable VSH decoder so the model output
    (packed coefficients) can be turned into ``P_pred`` before the rank
    comparison. Mirrors the ownership pattern in
    :class:`mpinv.losses.physics_power.PhysicsPowerLoss`.
    """

    def __init__(
        self,
        cfg: RankBinPLossConfig | None = None,
        grid: GridSpec = GRID_DEFAULT,
        l_max: int = L_MAX,
        decoder: DifferentiableMultipoleField | None = None,
    ):
        super().__init__()
        self.cfg = cfg or RankBinPLossConfig()
        self.grid = grid
        self.l_max = l_max
        self.decoder = (
            decoder if decoder is not None
            else DifferentiableMultipoleField(grid, l_max)
        )
        self._n_bins = int(self.cfg.n_bins) if self.cfg.n_bins is not None \
            else (2 * l_max + 1)
        if self._n_bins < 2:
            raise ValueError(
                f"rank_bin_p needs n_bins >= 2 (got {self._n_bins} — "
                f"with l_max={l_max} the default 2*l_max+1 already gives 11)"
            )
        self._last_components: dict[str, float] = {}

    @property
    def last_components(self) -> Mapping[str, float]:
        return self._last_components

    @property
    def n_bins(self) -> int:
        return self._n_bins

    def forward(
        self,
        pred_packed: torch.Tensor,
        target: torch.Tensor,
        target_packed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert_packed_coeffs(pred_packed, name="pred_packed")
        assert_power_pattern(target, grid=self.grid, name="target_P")

        P_pred = self.decoder(pred_packed)
        primary = rank_bin_mse(
            P_pred, target, n_bins=self._n_bins, beta=self.cfg.beta
        )
        components: dict[str, float] = {"rank_bin_p": primary.detach().item()}

        if self.cfg.coef_aux_weight > 0:
            if target_packed is None:
                raise ValueError(
                    "RankBinPLoss configured with coef_aux_weight > 0 needs target_packed"
                )
            assert_packed_coeffs(target_packed, name="target_packed")
            aux = (pred_packed - target_packed).pow(2).mean()
            components["coef_aux"] = aux.detach().item()
            total = primary + self.cfg.coef_aux_weight * aux
        else:
            total = primary

        components["total"] = total.detach().item()
        self._last_components = components
        return total
