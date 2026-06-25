"""Single-source-of-truth registry for feature extractors."""

from __future__ import annotations

from typing import Any

FEATURE_EXTRACTORS: dict[str, type[Any]] = {}
"""Mapping ``name -> class`` used to instantiate feature extractors from configs."""


def register_feature(name: str):
    """Decorator: register a feature-extractor class under ``name``."""

    def _wrap(cls: type[Any]) -> type[Any]:
        if name in FEATURE_EXTRACTORS and FEATURE_EXTRACTORS[name] is not cls:
            raise ValueError(f"feature extractor {name!r} already registered to a different class")
        FEATURE_EXTRACTORS[name] = cls
        return cls

    return _wrap
