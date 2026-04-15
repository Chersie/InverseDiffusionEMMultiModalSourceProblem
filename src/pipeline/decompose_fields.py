from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.config import ANGLE_STEP_DEG, DEFAULT_MAXORDER, LIBRARY_HEADER_LINES
from src.common.paths import NAIVE_DIR


def _validate_row_order(
    rows: list[list[str]],
    *,
    size_phi: int,
    size_theta: int,
    legacy_grid: bool = False,
) -> None:
    """Validate expected theta/phi ordering in serialized field rows."""
    if not rows:
        return

    theta0 = float(rows[0][0])
    phi0 = float(rows[0][1])
    if legacy_grid:
        expected_theta0 = 0.0
        expected_phi0 = 0.0
    else:
        expected_theta0 = ANGLE_STEP_DEG
        expected_phi0 = 0.0
    use_radians = False
    if np.isclose(theta0, np.deg2rad(expected_theta0)) and np.isclose(phi0, np.deg2rad(expected_phi0)):
        use_radians = True
    elif not (np.isclose(theta0, expected_theta0) and np.isclose(phi0, expected_phi0)):
        raise ValueError(
            "Unexpected row ordering in field/library file. "
            f"First row has (theta,phi)=({theta0},{phi0}), expected ({expected_theta0},{expected_phi0})."
        )

    if legacy_grid:
        jump_idx = size_theta + 2
        next_theta = float(rows[jump_idx][0])
        next_phi = float(rows[jump_idx][1])
        expected_next_theta = 0.0
        expected_next_phi = ANGLE_STEP_DEG
    else:
        jump_idx = size_theta
        next_theta = float(rows[jump_idx][0])
        next_phi = float(rows[jump_idx][1])
        expected_next_theta = ANGLE_STEP_DEG
        expected_next_phi = ANGLE_STEP_DEG

    if use_radians:
        expected_next_theta = np.deg2rad(expected_next_theta)
        expected_next_phi = np.deg2rad(expected_next_phi)

    if not (np.isclose(next_theta, expected_next_theta) and np.isclose(next_phi, expected_next_phi)):
        raise ValueError(
            "Unexpected theta/phi row progression in field/library file. "
            f"Row {jump_idx} has (theta,phi)=({next_theta},{next_phi}), "
            f"expected ({expected_next_theta},{expected_next_phi})."
        )


def _load_amplitudes_from_field_file(field_path: Path) -> np.ndarray:
    size_phi = int(360 / ANGLE_STEP_DEG)
    size_theta = int(180 / ANGLE_STEP_DEG) - 1
    expected_rows = size_phi * size_theta

    values: list[list[str]] = []
    with field_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                values.append(parts)
    if len(values) < expected_rows:
        raise ValueError(f"Field file {field_path} has {len(values)} rows; expected at least {expected_rows}.")
    _validate_row_order(values, size_phi=size_phi, size_theta=size_theta, legacy_grid=False)

    a_study = np.zeros((size_phi, size_theta, 2), dtype=complex)
    for j in range(size_phi):
        for i in range(size_theta):
            idx = size_theta * j + i
            row = values[idx]
            a_study[j, i, 0] = float(row[3]) * np.exp(1j * float(row[4]))
            a_study[j, i, 1] = float(row[5]) * np.exp(1j * float(row[6]))
    return a_study


def _load_library_mode(library_path: Path) -> tuple[np.ndarray, np.ndarray]:
    size_phi = int(360 / ANGLE_STEP_DEG)
    size_theta = int(180 / ANGLE_STEP_DEG) - 1

    rows: list[list[str]] = []
    with library_path.open() as f:
        for line in f.readlines()[LIBRARY_HEADER_LINES:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                rows.append(parts)

    expected_rows = size_phi * size_theta
    legacy_rows = (size_phi + 1) * (size_theta + 2)
    if len(rows) < expected_rows:
        raise ValueError(
            f"Library file {library_path} has {len(rows)} rows; expected at least {expected_rows}."
        )

    amp = np.zeros((size_phi, size_theta, 2), dtype=complex)
    theta = np.zeros((size_phi, size_theta), dtype=float)
    use_legacy_grid = len(rows) >= legacy_rows
    _validate_row_order(rows, size_phi=size_phi, size_theta=size_theta, legacy_grid=use_legacy_grid)
    for j in range(size_phi):
        for i in range(size_theta):
            if use_legacy_grid:
                # Legacy Chersie libraries use 361x181 including poles.
                idx = (size_theta + 2) * j + (i + 1)
            else:
                idx = size_theta * j + i
            row = rows[idx]
            amp[j, i, 0] = float(row[3]) * np.exp(1j * float(row[4]))
            amp[j, i, 1] = float(row[5]) * np.exp(1j * float(row[6]))
            theta[j, i] = float(row[0])
    if np.nanmax(theta) <= np.pi + 1e-6:
        theta = np.rad2deg(theta)
    return amp, theta


def decompose_fields(*, field_file: str = "Fields.txt", library_dir: Path) -> Path:
    field_path = Path(field_file)
    if not field_path.is_absolute():
        field_path = NAIVE_DIR / field_path
    stem = field_path.stem
    result_path = NAIVE_DIR / f"Results_{stem}.txt"

    a_study = _load_amplitudes_from_field_file(field_path)
    d_omega = (ANGLE_STEP_DEG * np.pi / 180.0) ** 2

    with result_path.open("w") as out:
        for l in range(1, DEFAULT_MAXORDER + 1):
            for m in range(-l, l + 1):
                e_amp, e_theta = _load_library_mode(library_dir / f"E_l{l}_m{m}.txt")
                e_j = np.sum(
                    (
                        np.conj(e_amp[..., 0]) * a_study[..., 0]
                        + np.conj(e_amp[..., 1]) * a_study[..., 1]
                    )
                    * np.sin(np.deg2rad(e_theta))
                )
                out.write(f"E {l} {m} {np.real(e_j * d_omega)} {np.imag(e_j * d_omega)}\n")

                m_amp, m_theta = _load_library_mode(library_dir / f"M_l{l}_m{m}.txt")
                m_j = np.sum(
                    (
                        np.conj(m_amp[..., 0]) * a_study[..., 0]
                        + np.conj(m_amp[..., 1]) * a_study[..., 1]
                    )
                    * np.sin(np.deg2rad(m_theta))
                )
                out.write(f"M {l} {m} {np.real(m_j * d_omega)} {np.imag(m_j * d_omega)}\n")

    return result_path
