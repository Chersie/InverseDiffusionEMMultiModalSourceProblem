#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.common.config import ANGLE_STEP_DEG, LIBRARY_HEADER_LINES
from src.common.paths import LIBRARY_FAST_DIR, NAIVE_DIR


def _read_rows(path: Path, *, skip_lines: int = 0) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < skip_lines:
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                rows.append(parts)
    return rows


def _assert_close(value: float, expected: float, *, label: str) -> None:
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label}: got {value}, expected {expected}.")


def _validate_order(rows: list[list[str]], *, size_theta: int, legacy_grid: bool) -> None:
    if not rows:
        raise ValueError("No data rows found.")

    theta0 = float(rows[0][0])
    phi0 = float(rows[0][1])
    expected_theta0 = 0.0 if legacy_grid else ANGLE_STEP_DEG
    expected_phi0 = 0.0
    use_radians = False
    if math.isclose(theta0, math.radians(expected_theta0), rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        phi0, math.radians(expected_phi0), rel_tol=1e-9, abs_tol=1e-9
    ):
        use_radians = True
    else:
        _assert_close(theta0, expected_theta0, label="first theta")
        _assert_close(phi0, expected_phi0, label="first phi")

    jump_idx = size_theta + 2 if legacy_grid else size_theta
    theta_jump = float(rows[jump_idx][0])
    phi_jump = float(rows[jump_idx][1])
    expected_theta_jump = 0.0 if legacy_grid else ANGLE_STEP_DEG
    expected_phi_jump = ANGLE_STEP_DEG
    if use_radians:
        expected_theta_jump = math.radians(expected_theta_jump)
        expected_phi_jump = math.radians(expected_phi_jump)
    _assert_close(theta_jump, expected_theta_jump, label="jump theta")
    _assert_close(phi_jump, expected_phi_jump, label="jump phi")


def validate_fields_file(path: Path) -> None:
    size_phi = int(360 / ANGLE_STEP_DEG)
    size_theta = int(180 / ANGLE_STEP_DEG) - 1
    expected_rows = size_phi * size_theta

    rows = _read_rows(path)
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: got {len(rows)} rows, expected {expected_rows}.")
    _validate_order(rows, size_theta=size_theta, legacy_grid=False)


def validate_library_file(path: Path) -> None:
    size_phi = int(360 / ANGLE_STEP_DEG)
    size_theta = int(180 / ANGLE_STEP_DEG) - 1
    expected_rows = size_phi * size_theta
    legacy_rows = (size_phi + 1) * (size_theta + 2)

    rows = _read_rows(path, skip_lines=LIBRARY_HEADER_LINES)
    if len(rows) not in {expected_rows, legacy_rows}:
        raise ValueError(
            f"{path}: got {len(rows)} rows, expected {expected_rows} (canonical) "
            f"or {legacy_rows} (legacy)."
        )
    _validate_order(rows, size_theta=size_theta, legacy_grid=(len(rows) == legacy_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate field/library row-count and ordering invariants.")
    parser.add_argument("--field-file", default=str(NAIVE_DIR / "Fields.txt"))
    parser.add_argument("--library-file", default=str(LIBRARY_FAST_DIR / "E_l1_m0.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_fields_file(Path(args.field_file))
    validate_library_file(Path(args.library_file))
    print("Grid file invariants: OK")


if __name__ == "__main__":
    main()
