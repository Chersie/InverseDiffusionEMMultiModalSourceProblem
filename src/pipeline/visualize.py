from __future__ import annotations

from pathlib import Path

from src.common.io_utils import run_python_script
from src.common.paths import NAIVE_DIR


def show_multipoles() -> None:
    run_python_script(NAIVE_DIR / "4 ShowMultipoles.py", cwd=NAIVE_DIR)


def plot_multipoles_3d(kind: str = "Axial_Ratio", *, results_file: Path | str = "Results_Fields.txt") -> None:
    run_python_script(
        NAIVE_DIR / "4 Plot3DMultipoles.py",
        cwd=NAIVE_DIR,
        args=[kind, "--results", str(results_file)],
    )
