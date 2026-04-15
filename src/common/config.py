from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import LIBRARY_FAST_DIR


VALID_PIPELINE_TEST_STEPS = ("1", "2", "3", "4a", "4b")
DEFAULT_MAXORDER = 15
ANGLE_STEP_DEG = 1
LIBRARY_HEADER_LINES = 43


@dataclass(frozen=True)
class PipelineDefaults:
    field_file: str = "Fields.txt"
    result_file: str = "Results_Fields.txt"
    field_file_stem_env: str = "FIELD_FILE"
    library_env: str = "MULTIPOLAR_LIBRARY"
    default_library: Path = LIBRARY_FAST_DIR


DEFAULTS = PipelineDefaults()


def as_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
