"""Training layer: Trainer, StagedTrainer, optimiser builder, AMP, sanity checks."""

from mpinv.training.amp import AMPConfig, build_amp
from mpinv.training.optim import OptimiserConfig, SchedulerConfig, build_optimiser, build_scheduler
from mpinv.training.sanity import sanity_check_loss_participation, sanity_check_optimiser_coverage
from mpinv.training.staged import (
    BackbonePolicy,
    StagedTrainer,
    StagedTrainerConfig,
    StageReport,
    apply_stage_policy,
    build_stage_optimiser,
)
from mpinv.training.trainer import Trainer, TrainerConfig, TrainingContext

__all__ = [
    "AMPConfig",
    "BackbonePolicy",
    "OptimiserConfig",
    "SchedulerConfig",
    "StageReport",
    "StagedTrainer",
    "StagedTrainerConfig",
    "Trainer",
    "TrainerConfig",
    "TrainingContext",
    "apply_stage_policy",
    "build_amp",
    "build_optimiser",
    "build_scheduler",
    "build_stage_optimiser",
    "sanity_check_loss_participation",
    "sanity_check_optimiser_coverage",
]
