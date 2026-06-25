"""Loss layer: registry, coefficient MSE, physics power loss, differentiable VSH decoder."""

from mpinv.losses.coef_mse import CoefMSE, CoefMSEConfig
from mpinv.losses.differentiable_field import DifferentiableMultipoleField
from mpinv.losses.physics_power import PhysicsPowerLoss, PhysicsPowerLossConfig
from mpinv.losses.rank_bin import RankBinPLoss, RankBinPLossConfig, rank_bin_mse
from mpinv.losses.registry import LOSSES, register_loss

__all__ = [
    "LOSSES",
    "CoefMSE",
    "CoefMSEConfig",
    "DifferentiableMultipoleField",
    "PhysicsPowerLoss",
    "PhysicsPowerLossConfig",
    "RankBinPLoss",
    "RankBinPLossConfig",
    "rank_bin_mse",
    "register_loss",
]
