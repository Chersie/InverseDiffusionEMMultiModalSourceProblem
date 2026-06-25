"""Physics-aware loss: ``sin θ``-weighted MSE in power-pattern space.

Replaces the legacy ``PhysicsPowerLoss``. Differences from the legacy:

- **No silent shape coercion**. We assert the predicted and target shapes match
  the canonical layout ``(B, n_theta, n_phi)``; mismatches raise ``ValueError``.
- **No bilinear resize fallback**. The legacy resized the target if it disagreed
  with the prediction; that hid bugs.
- **Single source of truth** for the area weights via
  :func:`mpinv.core.area_weights.normalised_area_weights`.

Optional log-ratio variant (``log_ratio=True``) computes the loss in
``log(P_pred + eps) - log(P_true + eps)`` space, which is helpful when the dynamic
range of P is large; off by default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from mpinv.core.area_weights import torch_area_weights
from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX, zero_above_band
from mpinv.core.shapes import assert_packed_coeffs, assert_power_pattern
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.rank_bin import rank_bin_mse
from mpinv.losses.registry import register_loss


@dataclass(slots=True)
class PhysicsPowerLossConfig:
    """Knobs for :class:`PhysicsPowerLoss`."""

    log_ratio: bool = False
    log_eps: float = 1e-12
    coef_aux_weight: float = 0.0
    """If > 0, add ``coef_aux_weight * MSE(packed_pred, packed_target)`` as a
    secondary term. Useful for warm-up training to escape flat regions.
    The auxiliary term is *always* taken against the full ``target_packed``;
    truncation (see :attr:`truncate_target_to_band`) only affects the primary
    field-space term."""
    rank_bin_weight: float = 0.0
    """If > 0, add ``rank_bin_weight * rank_bin_mse(P_pred, P_true)`` as a
    third regulariser. The rank-bin term is scale- and shift-invariant on
    P, so it focuses the gradient on pattern *shape* (which pixel ranks
    higher than which) rather than absolute amplitude. See
    :mod:`mpinv.losses.rank_bin` for the soft-binning formulation."""
    rank_bin_n_bins: int | None = None
    """Number of rank bins. Defaults to ``2 * l_max + 1`` if ``None``,
    matching the angular resolution of order-``l_max`` spherical harmonics."""
    rank_bin_beta: float = 10.0
    """Sigmoid temperature for the rank-bin soft binning."""
    truncate_target_to_band: int | None = None
    """If set, the target P is recomputed at every forward pass from the
    bands ``l ≤ truncate_target_to_band`` of ``target_packed`` (zeroing all
    higher-l contributions before passing through the differentiable VSH
    decoder). This implements the proposal-§"loss на голове k" branch where
    head ``k`` is supervised against ``P`` synthesised from coefficients up
    to band ``k`` inclusive — useful in staged training so the active head
    is not penalised for content that future heads will produce.

    When set, ``forward`` requires ``target_packed`` (it cannot recompute
    the truncated P from ``target`` alone). Setting back to ``None`` (or to
    ``l_max``) reverts to the canonical "loss against the full ground-truth
    P" mode."""


@register_loss("physics_power")
class PhysicsPowerLoss(nn.Module):
    """Sin θ-weighted MSE in power-pattern space, computed through a differentiable
    VSH decoder.

    The decoder is owned by this loss (not the model), because the loss needs it
    every step but the model does not. This keeps the model architecture-agnostic.
    """

    def __init__(
        self,
        cfg: PhysicsPowerLossConfig | None = None,
        grid: GridSpec = GRID_DEFAULT,
        l_max: int = L_MAX,
        decoder: DifferentiableMultipoleField | None = None,
    ):
        super().__init__()
        self.cfg = cfg or PhysicsPowerLossConfig()
        self.grid = grid
        self.l_max = l_max
        self.decoder = decoder if decoder is not None else DifferentiableMultipoleField(grid, l_max)
        # area weights of shape (n_theta, n_phi); registered as buffer for device tracking
        self.register_buffer(
            "area_w",
            torch_area_weights(grid=grid, dtype=torch.float32, normalised=True),
            persistent=False,
        )
        self._last_components: dict[str, float] = {}

    @property
    def last_components(self) -> Mapping[str, float]:
        return self._last_components

    def forward(
        self,
        pred_packed: torch.Tensor,
        target: torch.Tensor,
        target_packed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        pred_packed : torch.Tensor
            Predicted packed coefficients ``(B, 4 K)``.
        target : torch.Tensor
            Target power pattern ``(B, n_theta, n_phi)``.
        target_packed : torch.Tensor or None
            Optional ground-truth packed coefficients, used only when the auxiliary
            coefficient term is enabled (``cfg.coef_aux_weight > 0``).
        """
        assert_packed_coeffs(pred_packed, name="pred_packed")
        assert_power_pattern(target, grid=self.grid, name="target_P")

        P_pred = self.decoder(pred_packed)  # (B, n_theta, n_phi)

        # Truncated-target mode: replace the ground-truth P with one synthesised
        # from coefficients up to band k inclusive. The differentiable decoder is
        # the same module used for P_pred, so ``target_truncated`` lives on the
        # same device / dtype as ``P_pred``. The recomputation is detached from
        # the graph (target P is a fixed quantity for the loss).
        target_for_field = target
        truncate_k = self.cfg.truncate_target_to_band
        if truncate_k is not None and truncate_k < self.l_max:
            if target_packed is None:
                raise ValueError(
                    "PhysicsPowerLoss configured with truncate_target_to_band "
                    "needs target_packed to recompute the truncated target P"
                )
            assert_packed_coeffs(target_packed, name="target_packed")
            with torch.no_grad():
                packed_truncated = zero_above_band(
                    target_packed, int(truncate_k), self.l_max
                )
                target_for_field = self.decoder(packed_truncated)

        if self.cfg.log_ratio:
            log_pred = torch.log(P_pred + self.cfg.log_eps)
            log_true = torch.log(target_for_field + self.cfg.log_eps)
            sq = (log_pred - log_true).pow(2)
        else:
            sq = (P_pred - target_for_field).pow(2)
        weighted = sq * self.area_w
        primary = weighted.mean()

        components: dict[str, float] = {"physics_power": primary.detach().item()}

        total = primary
        if self.cfg.coef_aux_weight > 0:
            if target_packed is None:
                raise ValueError(
                    "PhysicsPowerLoss configured with coef_aux_weight > 0 needs target_packed"
                )
            assert_packed_coeffs(target_packed, name="target_packed")
            aux = (pred_packed - target_packed).pow(2).mean()
            components["coef_aux"] = aux.detach().item()
            total = total + self.cfg.coef_aux_weight * aux

        if self.cfg.rank_bin_weight > 0:
            n_bins = (
                int(self.cfg.rank_bin_n_bins)
                if self.cfg.rank_bin_n_bins is not None
                else (2 * self.l_max + 1)
            )
            rank = rank_bin_mse(
                P_pred, target, n_bins=n_bins, beta=self.cfg.rank_bin_beta
            )
            components["rank_bin_p"] = rank.detach().item()
            total = total + self.cfg.rank_bin_weight * rank

        components["total"] = total.detach().item()
        self._last_components = components
        return total
