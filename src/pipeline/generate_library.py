from __future__ import annotations

from pathlib import Path

from src.common.io_utils import run_python_script
from src.common.paths import CHERSIE_DIR


FAST_SCRIPT = CHERSIE_DIR / "MPField_Spherical_Fast.py"
SLOW_SCRIPT = CHERSIE_DIR / "MPField_Spherical_Write_Updated.py"


def generate_fast_library() -> Path:
    run_python_script(FAST_SCRIPT, cwd=CHERSIE_DIR)
    return CHERSIE_DIR / "FieldsFast0.5"


def generate_slow_library() -> Path:
    run_python_script(SLOW_SCRIPT, cwd=CHERSIE_DIR)
    return CHERSIE_DIR / "Fields0.5"
