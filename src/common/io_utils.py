from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_python_script(
    script_path: Path,
    *,
    cwd: Path | None = None,
    args: Iterable[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a Python script with explicit path and cwd."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    subprocess.check_call(cmd, cwd=str(cwd or script_path.parent), env=env or os.environ.copy())


def safe_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    out = os.environ.copy()
    if extra_env:
        out.update(extra_env)
    return out
