"""Stage-wise training of :class:`mpinv.models.multi_head_mlp.MultiHeadMLP`.

This module composes (does not replace) :mod:`mpinv.training.trainer`. A
*stage* trains a single per-l-band head on top of the shared backbone:

- Heads with index ``< k`` (already trained in earlier stages) are
  ``requires_grad = False``, weights left at their trained values.
- Head with index ``k`` (the active one) is set ``requires_grad = True``.
  Optionally re-initialised to PyTorch's default ``Linear`` distribution at the
  stage boundary (``reinit_active_head``, default ``True``) so each head
  starts from non-zero weights for faster convergence (the user's preferred
  recipe).
- Heads with index ``> k`` (the future ones) have weight + bias forced to
  zero AND ``requires_grad = False``, so they emit identically zero in the
  forward pass and contribute nothing to ``P_pred``.
- The shared backbone follows ``backbone_policy``:

  * ``trainable_always`` — backbone tracks every stage with the same LR.
  * ``freeze_after_stage1`` (default) — backbone trains only in stage 1; from
    stage 2 onwards it is ``requires_grad = False`` and disappears from the
    optimiser.
  * ``lower_lr_after_stage1`` — backbone keeps training but in a separate
    optimiser parameter group with LR scaled by ``backbone_lr_factor``.

A fresh optimiser and scheduler are built per stage (over the new set of
``requires_grad=True`` parameters), then ``Trainer.fit(...)`` runs for
``stage_max_epochs`` epochs with the user-supplied callbacks and sinks.

The framework's ``sanity_check_optimiser_coverage`` (run from ``Trainer.fit``)
verifies after each stage transition that every trainable parameter is in the
optimiser exactly once, which is the binding correctness check on the freezing
logic. See R4 in [research/framework-rebuild/manifest.md](../../research/framework-rebuild/manifest.md).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.optim import Optimizer

from mpinv.callbacks.base import Callback
from mpinv.models.multi_head_mlp import MultiHeadMLP
from mpinv.training.optim import OptimiserConfig, SchedulerConfig, build_scheduler
from mpinv.training.trainer import Trainer, TrainerConfig, TrainingContext

logger = logging.getLogger(__name__)

BackbonePolicy = Literal[
    "trainable_always",
    "freeze_after_stage1",
    "lower_lr_after_stage1",
    "all_trainable_active_boost",
]


@dataclass(slots=True)
class StagedTrainerConfig:
    """Knobs for :class:`StagedTrainer`.

    Attributes
    ----------
    stage_max_epochs : int
        Per-stage epoch budget. Identical across stages by default; if a user
        wants different budgets per stage they can wrap
        :meth:`StagedTrainer.fit_one_stage` directly.
    backbone_policy : BackbonePolicy
        How the shared backbone is treated across stages. Default
        ``freeze_after_stage1`` matches the user's stability requirement.
    backbone_lr_factor : float
        Used only when ``backbone_policy == "lower_lr_after_stage1"``: backbone
        param-group runs at ``backbone_lr_factor * head_lr``.
    reinit_active_head : bool
        Run ``head.reset_parameters()`` on the active head at every stage
        boundary (the user's "re-initialise head 2 with non-zero weights" path).
    zero_init_future_heads : bool
        Apply ``head.zero_init_head(idx)`` to every head with index ``> k`` at
        the start of stage ``k``. ``True`` by default — this is the
        "zero modes 2-5 while training mode 1" branch of the user's plan; turn
        off only for the explicit "frozen-random heads" ablation.
    starting_stage : int
        1-indexed first stage to run. Useful with a transplanted model
        (e.g. heads 1..5 transplanted from a smaller-L run, ``starting_stage=6``
        kicks off training of head 6 onwards).
    checkpoint_root : str | None
        If set, write ``<checkpoint_root>/stage_<k>/best.pt`` etc. via the
        per-stage ``CheckpointCallback`` instances supplied by the
        callbacks factory. Otherwise checkpoints land wherever the callbacks
        the factory returns choose to put them.
    """

    stage_max_epochs: int = 30
    backbone_policy: BackbonePolicy = "freeze_after_stage1"
    backbone_lr_factor: float = 0.1
    reinit_active_head: bool = True
    zero_init_future_heads: bool = True
    starting_stage: int = 1
    checkpoint_root: str | None = None
    truncate_target_to_active_band: bool = False
    """If ``True``, at every stage transition set
    ``loss_fn.cfg.truncate_target_to_band`` to ``max(group)`` of the active
    head's l-band group, so :class:`mpinv.losses.physics_power.PhysicsPowerLoss`
    supervises the active head against P synthesised only from
    ``l ≤ active_band`` (proposal §"loss на голове k"). Requires the loss to
    expose ``cfg.truncate_target_to_band`` (i.e. it is a PhysicsPowerLoss);
    silently ignored for any loss that does not have such a knob."""


@dataclass(slots=True)
class StageReport:
    """Per-stage outcome for downstream logging / aggregation."""

    stage_idx: int
    active_head_idx: int
    group: list[int]
    trainable_summary: dict[str, bool]
    epochs_run: int
    last_eval_metrics: dict[str, float]
    stop_requested: bool
    n_trainable_params: int


def apply_stage_policy(
    model: MultiHeadMLP,
    stage_idx: int,
    *,
    backbone_policy: BackbonePolicy = "freeze_after_stage1",
    reinit_active_head: bool = True,
    zero_init_future_heads: bool = True,
) -> None:
    """Apply the freezing / zeroing / reinit policy for stage ``stage_idx`` (1-indexed).

    Heads ``0 .. stage_idx-2`` (1-indexed: ``1 .. stage_idx-1``) → ``requires_grad=False``
    (left at whatever state they were in — typically trained in earlier
    stages). Head ``stage_idx-1`` (1-indexed: ``stage_idx``) → trainable, optionally
    re-initialised. Heads ``stage_idx .. n_heads-1`` → optionally zeroed,
    always frozen.

    Backbone follows ``backbone_policy``.
    """
    if not (1 <= stage_idx <= model.n_heads):
        raise ValueError(
            f"stage_idx={stage_idx} out of range [1, {model.n_heads}]"
        )
    # Under "all_trainable_active_boost" the previous heads stay trainable; the
    # optimiser builder gives them a lower LR. Every other policy freezes them.
    keep_previous_trainable = backbone_policy == "all_trainable_active_boost"
    active = stage_idx - 1
    for j in range(model.n_heads):
        if j < active:
            model.set_head_trainable(j, keep_previous_trainable)
        elif j == active:
            if reinit_active_head:
                model.reinit_head(j)
            model.set_head_trainable(j, True)
        else:
            if zero_init_future_heads:
                model.zero_init_head(j)
            model.set_head_trainable(j, False)

    if backbone_policy == "trainable_always":
        model.set_backbone_trainable(True)
    elif backbone_policy == "freeze_after_stage1":
        model.set_backbone_trainable(stage_idx == 1)
    elif backbone_policy == "lower_lr_after_stage1":
        # Backbone is trainable, but the optimiser builder will scale its LR.
        model.set_backbone_trainable(True)
    elif backbone_policy == "all_trainable_active_boost":
        # Backbone trains at backbone_lr_factor * lr; the optimiser builder
        # also keeps previous heads at backbone_lr_factor * lr, leaving the
        # active head at the full lr.
        model.set_backbone_trainable(True)
    else:
        raise ValueError(f"unknown backbone_policy {backbone_policy!r}")


def build_stage_optimiser(
    model: MultiHeadMLP,
    optim_cfg: OptimiserConfig,
    *,
    stage_idx: int,
    backbone_policy: BackbonePolicy = "freeze_after_stage1",
    backbone_lr_factor: float = 0.1,
) -> Optimizer:
    """Build an optimiser over the trainable params at the start of stage ``stage_idx``.

    For ``lower_lr_after_stage1`` and ``stage_idx > 1`` the optimiser has two
    parameter groups: backbone (at ``backbone_lr_factor * lr``) and active
    head (at ``lr``). For all other cases there is a single group at ``lr``.

    The returned optimiser is fully covered: every ``requires_grad=True``
    parameter is in exactly one group, which the framework's
    ``sanity_check_optimiser_coverage`` verifies on the warm-up step.
    """
    active = stage_idx - 1
    active_head_params = [
        p for p in model.heads[active].parameters() if p.requires_grad
    ]
    other_head_params = [
        p
        for j, head in enumerate(model.heads)
        if j != active
        for p in head.parameters()
        if p.requires_grad
    ]
    head_params = active_head_params + other_head_params
    backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    if not head_params and not backbone_params:
        raise ValueError("no trainable parameters at this stage")

    fused_supported = torch.cuda.is_available()
    fused = optim_cfg.fused and fused_supported

    use_split_groups_lower = (
        backbone_policy == "lower_lr_after_stage1"
        and stage_idx > 1
        and bool(backbone_params)
    )
    use_split_groups_active_boost = (
        backbone_policy == "all_trainable_active_boost"
        and stage_idx > 1
    )

    if use_split_groups_active_boost:
        # Three groups: backbone @ lr*factor, previous heads @ lr*factor,
        # active head @ lr (the "boost"). Previous heads may be empty at
        # stage_idx==2 with a single previous head; we still emit the group
        # so the param coverage check runs through it.
        groups: list[dict[str, Any]] = []
        slow_group_params: list[Any] = list(backbone_params) + list(other_head_params)
        if slow_group_params:
            groups.append({
                "params": slow_group_params,
                "lr": optim_cfg.lr * backbone_lr_factor,
                "weight_decay": optim_cfg.weight_decay,
            })
        if active_head_params:
            groups.append({
                "params": active_head_params,
                "lr": optim_cfg.lr,
                "weight_decay": optim_cfg.weight_decay,
            })
    elif use_split_groups_lower:
        groups = [
            {
                "params": backbone_params,
                "lr": optim_cfg.lr * backbone_lr_factor,
                "weight_decay": optim_cfg.weight_decay,
            },
            {
                "params": head_params,
                "lr": optim_cfg.lr,
                "weight_decay": optim_cfg.weight_decay,
            },
        ]
    else:
        groups = [
            {
                "params": backbone_params + head_params,
                "lr": optim_cfg.lr,
                "weight_decay": optim_cfg.weight_decay,
            }
        ]

    if optim_cfg.name == "adamw":
        return torch.optim.AdamW(
            groups, betas=optim_cfg.betas, eps=optim_cfg.eps, fused=fused
        )
    if optim_cfg.name == "adam":
        return torch.optim.Adam(
            groups, betas=optim_cfg.betas, eps=optim_cfg.eps, fused=fused
        )
    if optim_cfg.name == "sgd":
        nesterov = bool(optim_cfg.nesterov) and optim_cfg.momentum > 0.0
        return torch.optim.SGD(
            groups, momentum=optim_cfg.momentum, nesterov=nesterov
        )
    raise ValueError(f"unknown optimiser {optim_cfg.name!r}")


class StagedTrainer:
    """Run a multi-head MLP through stage-by-stage training.

    The trainer keeps state minimal: each stage is a fresh
    :class:`Trainer.fit` call with a fresh optimiser, fresh scheduler, and
    fresh callback instances (provided by ``callbacks_factory(stage_idx)``).
    Stateful callbacks (``EarlyStoppingCallback``,
    ``CheckpointCallback``) are therefore NOT shared across stages — a
    stage-1 plateau does not stop stage 2, and stage 2 starts from a
    fresh patience counter.

    Returns a list of :class:`StageReport` (one per stage that actually ran;
    stages skipped via ``starting_stage`` are not reported). The model is
    mutated in place across stages.
    """

    def __init__(
        self,
        cfg: StagedTrainerConfig | None = None,
        *,
        trainer: Trainer | None = None,
    ):
        self.cfg = cfg or StagedTrainerConfig()
        self.trainer = trainer or Trainer(
            TrainerConfig(max_epochs=self.cfg.stage_max_epochs)
        )
        # Force the inner trainer's max_epochs to match the staged budget so a
        # caller who passes a custom Trainer with stale max_epochs cannot
        # accidentally truncate or extend a stage.
        self.trainer.cfg.max_epochs = self.cfg.stage_max_epochs

    def fit(
        self,
        model: MultiHeadMLP,
        train_loader: Iterable[Any],
        loss_fn: Any,
        optim_cfg: OptimiserConfig,
        *,
        loss_kind: str = "physics",
        sched_cfg: SchedulerConfig | None = None,
        val_loader: Iterable[Any] | None = None,
        callbacks_factory: Callable[[int], Sequence[Callback]] = lambda _stage: (),
        sinks_factory: Callable[[int], Sequence[Any]] = lambda _stage: (),
        unpack_batch: Any = None,
        steps_per_epoch: int | None = None,
    ) -> list[StageReport]:
        """Run stages ``starting_stage .. n_heads`` and return per-stage reports.

        Parameters
        ----------
        model : MultiHeadMLP
            The multi-head model to train. Mutated in place.
        train_loader, val_loader : iterables
            Same shape as :meth:`Trainer.fit`.
        loss_fn : callable
            Same shape as :meth:`Trainer.fit`.
        optim_cfg : OptimiserConfig
            Used for every stage. Per-stage variation is achieved by wrapping
            this method or by providing a different ``optim_cfg`` and calling
            :meth:`fit_one_stage` directly.
        loss_kind : "coef" | "physics"
            Same as :class:`Trainer`.
        sched_cfg : SchedulerConfig or None
            Optional. ``total_steps`` is recomputed per stage if
            ``steps_per_epoch`` is provided.
        callbacks_factory, sinks_factory : Callable
            Called once per stage with the 1-indexed stage number; expected to
            return a fresh list of callbacks / sinks.
        unpack_batch : optional callable
            Forwarded to :meth:`Trainer.fit`.
        steps_per_epoch : int or None
            If given, used together with ``stage_max_epochs`` to compute
            ``total_steps`` for the per-stage scheduler.
        """
        n_stages = model.n_heads
        if not (1 <= self.cfg.starting_stage <= n_stages):
            raise ValueError(
                f"starting_stage={self.cfg.starting_stage} out of range "
                f"[1, {n_stages}]"
            )

        reports: list[StageReport] = []
        for stage_idx in range(self.cfg.starting_stage, n_stages + 1):
            report = self.fit_one_stage(
                model,
                stage_idx=stage_idx,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optim_cfg=optim_cfg,
                loss_kind=loss_kind,
                sched_cfg=sched_cfg,
                val_loader=val_loader,
                callbacks=list(callbacks_factory(stage_idx)),
                sinks=list(sinks_factory(stage_idx)),
                unpack_batch=unpack_batch,
                steps_per_epoch=steps_per_epoch,
            )
            reports.append(report)
        return reports

    def fit_one_stage(
        self,
        model: MultiHeadMLP,
        *,
        stage_idx: int,
        train_loader: Iterable[Any],
        loss_fn: Any,
        optim_cfg: OptimiserConfig,
        loss_kind: str = "physics",
        sched_cfg: SchedulerConfig | None = None,
        val_loader: Iterable[Any] | None = None,
        callbacks: Sequence[Callback] = (),
        sinks: Sequence[Any] = (),
        unpack_batch: Any = None,
        steps_per_epoch: int | None = None,
    ) -> StageReport:
        """Run a single stage. Useful for tests and for callers that want
        per-stage knob variation that the loop in :meth:`fit` cannot express.
        """
        active_head = stage_idx - 1
        group = list(model.groups[active_head])
        logger.info(
            "STAGE %d/%d  active head idx=%d (l-band group %s)  policy=%s",
            stage_idx, model.n_heads, active_head, group,
            self.cfg.backbone_policy,
        )
        apply_stage_policy(
            model,
            stage_idx,
            backbone_policy=self.cfg.backbone_policy,
            reinit_active_head=self.cfg.reinit_active_head,
            zero_init_future_heads=self.cfg.zero_init_future_heads,
        )
        summary = model.trainable_summary()
        logger.info("  trainable: %s", summary)

        # Optional: tell the loss to compute its primary term against a target P
        # synthesised only from the bands up to (and including) max(group). The
        # loss is mutated in place and reused across stages — the next stage
        # boundary will overwrite the value, so we don't restore here.
        if self.cfg.truncate_target_to_active_band:
            cfg_attr = getattr(loss_fn, "cfg", None)
            if cfg_attr is not None and hasattr(cfg_attr, "truncate_target_to_band"):
                k_band = int(max(group))
                cfg_attr.truncate_target_to_band = k_band
                logger.info(
                    "  truncate_target_to_band <- %d (active band group max)",
                    k_band,
                )
            else:
                logger.warning(
                    "  truncate_target_to_active_band requested but loss_fn has "
                    "no cfg.truncate_target_to_band knob; skipping"
                )

        optimiser = build_stage_optimiser(
            model,
            optim_cfg,
            stage_idx=stage_idx,
            backbone_policy=self.cfg.backbone_policy,
            backbone_lr_factor=self.cfg.backbone_lr_factor,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  optimiser: %s | trainable params: %d", optim_cfg.name, n_trainable)

        scheduler = None
        if sched_cfg is not None and sched_cfg.name != "none":
            scheduler_cfg = SchedulerConfig(
                name=sched_cfg.name,
                total_steps=(
                    steps_per_epoch * self.cfg.stage_max_epochs
                    if steps_per_epoch is not None
                    else sched_cfg.total_steps
                ),
                warmup_steps=sched_cfg.warmup_steps,
                min_lr=sched_cfg.min_lr,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
                plateau_patience=sched_cfg.plateau_patience,
                plateau_factor=sched_cfg.plateau_factor,
            )
            scheduler = build_scheduler(optimiser, scheduler_cfg)

        # Run the stage. Trainer.fit returns the final TrainingContext.
        ctx: TrainingContext = self.trainer.fit(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimiser=optimiser,
            loss_kind=loss_kind,  # type: ignore[arg-type]
            scheduler=scheduler,
            val_loader=val_loader,
            callbacks=callbacks,
            sinks=sinks,
            **({"unpack_batch": unpack_batch} if unpack_batch is not None else {}),
        )

        return StageReport(
            stage_idx=stage_idx,
            active_head_idx=active_head,
            group=group,
            trainable_summary=summary,
            epochs_run=ctx.epoch,
            last_eval_metrics=dict(ctx.last_eval_metrics or {}),
            stop_requested=ctx.stop_requested,
            n_trainable_params=int(n_trainable),
        )


__all__ = [
    "BackbonePolicy",
    "StageReport",
    "StagedTrainer",
    "StagedTrainerConfig",
    "apply_stage_policy",
    "build_stage_optimiser",
]
