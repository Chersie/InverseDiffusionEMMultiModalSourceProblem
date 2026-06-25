"""Single-source-of-truth registry for trainable losses."""

from __future__ import annotations

from typing import Any

LOSSES: dict[str, type[Any]] = {}
"""Mapping ``name -> class``."""


def register_loss(name: str):
    def _wrap(cls: type[Any]) -> type[Any]:
        if name in LOSSES and LOSSES[name] is not cls:
            raise ValueError(f"loss {name!r} already registered to a different class")
        LOSSES[name] = cls
        return cls

    return _wrap
