"""Single-source-of-truth registry for trainable model classes.

The legacy framework had two competing ``get_model_registry`` symbols that disagreed.
This module is the *only* place the framework registers model implementations.
"""

from __future__ import annotations

from typing import Any

MODELS: dict[str, type[Any]] = {}
"""Mapping ``name -> class``."""


def register_model(name: str):
    """Decorator: register a model class under ``name``."""

    def _wrap(cls: type[Any]) -> type[Any]:
        if name in MODELS and MODELS[name] is not cls:
            raise ValueError(f"model {name!r} already registered to a different class")
        MODELS[name] = cls
        return cls

    return _wrap
