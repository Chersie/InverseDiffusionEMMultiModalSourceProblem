"""Automatic mixed-precision config & helpers (per practice.pdf p. 8).

Default precision is bf16 (no GradScaler needed). fp16 path uses GradScaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(slots=True)
class AMPConfig:
    """Knobs for :func:`build_amp`."""

    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    """Per practice.pdf p. 8: bf16 is preferred on supported hardware (Ampere+, Apple
    Silicon doesn't support bf16 autocast on MPS yet so fp32 stays the default)."""


def _device_supports_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8
    # CPU supports bf16 autocast via oneDNN; MPS does not as of torch 2.6+.
    return device.type == "cpu"


def build_amp(
    cfg: AMPConfig, device: torch.device
) -> tuple[torch.cuda.amp.autocast, torch.amp.GradScaler | None]:
    """Return ``(autocast_ctx, grad_scaler)``.

    The autocast context is a no-op for ``fp32``. The grad scaler is None for ``fp32``
    and ``bf16``, and a real :class:`torch.amp.GradScaler` for ``fp16``.
    """
    if cfg.precision == "fp32":
        ctx = torch.autocast(device_type=device.type, enabled=False)
        return ctx, None
    if cfg.precision == "bf16":
        if not _device_supports_bf16(device):
            raise RuntimeError(f"bf16 not supported on device {device}")
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
        return ctx, None
    if cfg.precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError("fp16 autocast + GradScaler is meaningful only on CUDA")
        ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        scaler = torch.amp.GradScaler("cuda")
        return ctx, scaler
    raise ValueError(f"unknown precision {cfg.precision!r}")
