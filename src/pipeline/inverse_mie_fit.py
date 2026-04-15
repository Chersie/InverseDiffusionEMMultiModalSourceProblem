from __future__ import annotations

from pathlib import Path

from src.common.io_utils import run_python_script
from src.common.paths import NAIVE_DIR


def run_inverse_mie(results_file: Path) -> None:
    script = NAIVE_DIR / "inverse_mie.py"
    run_python_script(script, cwd=NAIVE_DIR, args=[str(results_file)])
