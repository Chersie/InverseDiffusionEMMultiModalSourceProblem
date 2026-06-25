"""Sanity assertions per practice.pdf p. 15.

- Every trainable parameter is in *some* optimiser group exactly once.
- Every trainable parameter receives a non-zero gradient on a single warm-up step
  (catches detached losses and parameters not connected to the loss).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


def sanity_check_optimiser_coverage(model: nn.Module, optimiser: Optimizer) -> None:
    """Raise ``ValueError`` if any trainable parameter is missing from the optimiser
    or assigned to more than one group.
    """
    model_params = {id(p) for p in model.parameters() if p.requires_grad}
    optim_params: list[int] = []
    for group in optimiser.param_groups:
        for p in group["params"]:
            optim_params.append(id(p))

    if len(optim_params) != len(set(optim_params)):
        seen: set[int] = set()
        dups: list[int] = []
        for pid in optim_params:
            if pid in seen:
                dups.append(pid)
            seen.add(pid)
        raise ValueError(f"optimiser has {len(dups)} duplicated parameter id(s)")

    optim_set = set(optim_params)
    missing = model_params - optim_set
    extra = optim_set - model_params
    if missing:
        raise ValueError(f"{len(missing)} model parameters not assigned to optimiser")
    if extra:
        raise ValueError(f"optimiser holds {len(extra)} parameters not in the model")


def sanity_check_loss_participation(
    model: nn.Module,
    loss_fn,
    sample_input,
    sample_target,
    *args,
    **kwargs,
) -> None:
    """Run one forward/backward and assert every trainable parameter has a gradient."""
    model.zero_grad(set_to_none=True)
    pred = model(sample_input)
    loss = loss_fn(pred, sample_target, *args, **kwargs)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise ValueError(
            f"{len(missing)} trainable parameter(s) received no gradient on warm-up: "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}"
        )
    model.zero_grad(set_to_none=True)


def grad_norm(parameters: Iterable[nn.Parameter], norm_type: float = 2.0) -> float:
    """Compute the global gradient norm across the supplied parameters."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)
    total = torch.norm(torch.stack([torch.norm(g, p=norm_type) for g in grads]), p=norm_type)
    return total.item()
