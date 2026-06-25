"""Optimiser and learning-rate scheduler builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(slots=True)
class OptimiserConfig:
    """Knobs for :func:`build_optimiser`."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-15
    momentum: float = 0.9
    nesterov: bool = False
    """Only meaningful when ``name == 'sgd'`` and ``momentum > 0``. PyTorch
    requires ``momentum > 0`` to enable Nesterov, so :func:`build_optimiser`
    silently disables it when ``momentum == 0``."""
    fused: bool = False
    """Use ``fused=True`` on Adam/AdamW; per practice.pdf p. 13 this is a free
    memory + speed win on CUDA. Has no effect on CPU/MPS."""


@dataclass(slots=True)
class SchedulerConfig:
    """Knobs for :func:`build_scheduler`. Set ``name='none'`` for no scheduling."""

    name: Literal["none", "cosine", "cosine_with_warmup", "step", "plateau"] = "none"
    total_steps: int = 0
    warmup_steps: int = 0
    min_lr: float = 1e-6
    step_size: int = 10
    gamma: float = 0.1
    plateau_patience: int = 5
    plateau_factor: float = 0.5


def _filter_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def build_optimiser(model: nn.Module, cfg: OptimiserConfig) -> Optimizer:
    params = list(_filter_params(model))
    if not params:
        raise ValueError("model has no trainable parameters")
    common = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    fused_supported = torch.cuda.is_available()
    fused = cfg.fused and fused_supported
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, betas=cfg.betas, eps=cfg.eps, fused=fused, **common)
    if cfg.name == "adam":
        return torch.optim.Adam(params, betas=cfg.betas, eps=cfg.eps, fused=fused, **common)
    if cfg.name == "sgd":
        nesterov = bool(cfg.nesterov) and cfg.momentum > 0.0
        return torch.optim.SGD(
            params, momentum=cfg.momentum, nesterov=nesterov, **common
        )
    raise ValueError(f"unknown optimiser {cfg.name!r}")


def build_scheduler(opt: Optimizer, cfg: SchedulerConfig) -> LRScheduler | None:
    if cfg.name == "none":
        return None
    if cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(cfg.total_steps, 1), eta_min=cfg.min_lr
        )
    if cfg.name == "cosine_with_warmup":
        if cfg.total_steps <= cfg.warmup_steps:
            raise ValueError("total_steps must exceed warmup_steps for cosine_with_warmup")
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, total_iters=max(cfg.warmup_steps, 1)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(cfg.total_steps - cfg.warmup_steps, 1), eta_min=cfg.min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, [warm, cosine], milestones=[cfg.warmup_steps]
        )
    if cfg.name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=cfg.plateau_factor, patience=cfg.plateau_patience, min_lr=cfg.min_lr
        )
    raise ValueError(f"unknown scheduler {cfg.name!r}")
