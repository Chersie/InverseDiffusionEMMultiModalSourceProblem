"""Hydra/OmegaConf → MLflow params flattening.

MLflow params have a 6,000-char limit per value; we truncate after that and rely on
the run's resolved-config artifact for the full record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MLFLOW_PARAM_VALUE_MAX = 6000


def flatten_for_mlflow(d: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested mapping into ``"a.b.c" -> str(value)`` form.

    Lists become ``"[item, item, ...]"`` strings. None becomes the literal string
    ``"None"``. Long strings are truncated to MLflow's per-value limit.
    """
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, Mapping):
            out.update(flatten_for_mlflow(v, prefix=key))
        elif isinstance(v, (list, tuple)):
            out[key] = _to_str(v)
        else:
            out[key] = _to_str(v)
    return out


def _to_str(v: Any) -> str:
    s = repr(v) if not isinstance(v, str) else v
    if len(s) > _MLFLOW_PARAM_VALUE_MAX:
        s = s[: _MLFLOW_PARAM_VALUE_MAX - 3] + "..."
    return s
