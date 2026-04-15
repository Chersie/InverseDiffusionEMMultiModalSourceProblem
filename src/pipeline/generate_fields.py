from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import ANGLE_STEP_DEG, DEFAULT_MAXORDER
from src.common.paths import CHERSIE_DIR, NAIVE_DIR


@dataclass(frozen=True)
class GridShape:
    size_phi: int = int(360 / ANGLE_STEP_DEG)
    size_theta: int = int(180 / ANGLE_STEP_DEG) - 1


def _load_fast_module() -> Any:
    module_path = CHERSIE_DIR / "MPField_Spherical_Fast.py"
    spec = importlib.util.spec_from_file_location("legacy_mpfield_fast", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coefficients_from_latin_square(
    maxorder: int,
    *,
    scale: float = 1.0,
    seed_e_re: int = 0,
    seed_e_im: int = 1,
    seed_m_re: int = 2,
    seed_m_im: int = 3,
) -> tuple[dict[int, dict[int, complex]], dict[int, dict[int, complex]]]:
    n = 2 * maxorder + 1

    def latin_square(seed: int) -> np.ndarray:
        ls = np.array([(np.arange(n) + i + seed) % n for i in range(n)], dtype=np.float64)
        return scale * (2.0 * (ls + 1) / (n + 1) - 1.0)

    ls_e_re = latin_square(seed_e_re)
    ls_e_im = latin_square(seed_e_im)
    ls_m_re = latin_square(seed_m_re)
    ls_m_im = latin_square(seed_m_im)

    a_e: dict[int, dict[int, complex]] = {}
    a_m: dict[int, dict[int, complex]] = {}
    for l in range(1, maxorder + 1):
        a_e[l] = {}
        a_m[l] = {}
        for m in range(-l, l + 1):
            row = l - 1
            col = m + maxorder
            a_e[l][m] = ls_e_re[row, col] + 1j * ls_e_im[row, col]
            a_m[l][m] = ls_m_re[row, col] + 1j * ls_m_im[row, col]
    return a_e, a_m


def _build_output_grid(grid: GridShape) -> tuple[np.ndarray, np.ndarray]:
    phi_deg = np.arange(grid.size_phi) * ANGLE_STEP_DEG
    theta_deg = (np.arange(grid.size_theta) + 1) * ANGLE_STEP_DEG
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    phi_2d = np.broadcast_to(phi_rad[:, np.newaxis], (grid.size_phi, grid.size_theta))
    theta_2d = np.broadcast_to(theta_rad[np.newaxis, :], (grid.size_phi, grid.size_theta))
    return theta_2d, phi_2d


def _compute_field(
    mpfield_module: Any,
    a_e: dict[int, dict[int, complex]],
    a_m: dict[int, dict[int, complex]],
    theta: np.ndarray,
    phi: np.ndarray,
    maxorder: int,
) -> np.ndarray:
    amplitude = np.zeros((theta.shape[0], theta.shape[1], 2), dtype=complex)
    for l in range(1, maxorder + 1):
        for m in range(-l, l + 1):
            amp_e = mpfield_module.field_for_multipole(l, m, theta, phi, electric=True)
            amp_m = mpfield_module.field_for_multipole(l, m, theta, phi, electric=False)
            amplitude += a_e[l][m] * amp_e + a_m[l][m] * amp_m
    return amplitude


def generate_fields_latin_square(output_path: Path | None = None) -> Path:
    out_path = output_path or (NAIVE_DIR / "Fields.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = GridShape()
    maxorder = DEFAULT_MAXORDER
    mpfield_module = _load_fast_module()
    a_e, a_m = _coefficients_from_latin_square(maxorder)
    theta, phi = _build_output_grid(grid)
    amplitude = _compute_field(mpfield_module, a_e, a_m, theta, phi, maxorder)
    power = np.sum(np.abs(amplitude) ** 2, axis=-1)

    with out_path.open("w") as f:
        for j in range(grid.size_phi):
            for i in range(grid.size_theta):
                c_theta = (i + 1) * ANGLE_STEP_DEG
                c_phi = j * ANGLE_STEP_DEG
                f.write(
                    f"{c_theta} {c_phi} {power[j, i]} "
                    f"{np.abs(amplitude[j, i, 0])} {np.angle(amplitude[j, i, 0])} "
                    f"{np.abs(amplitude[j, i, 1])} {np.angle(amplitude[j, i, 1])}\n"
                )
    return out_path


def calculate_fields_from_tables() -> Path:
    z0 = 377.0
    threshold = 0.01
    tables_dir = NAIVE_DIR / "Tables"
    radiation_path = tables_dir / "RadiationPattern.txt"
    axial_path = tables_dir / "AxialRatio.txt"
    output_path = NAIVE_DIR / "Fields.txt"

    theta: list[float] = []
    phi: list[float] = []
    p_vals: list[float] = []
    gamma: list[float] = []
    theta_gamma: list[float] = []
    phi_gamma: list[float] = []

    with radiation_path.open() as f:
        for line in f:
            parts = line.strip().split()
            theta.append(float(parts[0]))
            phi.append(float(parts[1]))
            p_vals.append(float(parts[2]))

    with axial_path.open() as f:
        for line in f:
            parts = line.strip().split()
            theta_gamma.append(float(parts[0]))
            phi_gamma.append(float(parts[1]))
            gamma.append(float(parts[2]))

    if len(theta) != len(gamma):
        raise ValueError(
            f"Tables length mismatch: RadiationPattern has {len(theta)} rows, AxialRatio has {len(gamma)} rows."
        )

    n = len(theta)
    for i in range(n):
        if not (np.isclose(theta[i], theta_gamma[i]) and np.isclose(phi[i], phi_gamma[i])):
            raise ValueError(
                "Tables row-order mismatch between RadiationPattern and AxialRatio "
                f"at row {i}: ({theta[i]}, {phi[i]}) != ({theta_gamma[i]}, {phi_gamma[i]})."
            )

    def c_coeff(y: float) -> float:
        if abs(y) > threshold and abs(y) < 1 - threshold:
            denominator = y**8 - 2 * y**4 + 1
            if denominator <= 0:
                return 0.0
            c_root = 2 * np.sqrt((y**6 + 2 * y**4 + y**2) / denominator)
            return float(np.sign(y) * c_root * 1.00001)
        return 0.0

    def a_coeff(y: float, c_val: float) -> float:
        if abs(y) > threshold and abs(y) < 1 - threshold:
            discriminant = (2 * y**2 - c_val**2 - c_val**2 * y**4) ** 2 - 4 * y**4 * (c_val**4 + 2 * c_val**2 + 1)
            if discriminant < 0:
                if discriminant > -1e-12:
                    discriminant = 0.0
                else:
                    return 0.0
            d = np.sqrt(discriminant)
            a_root1 = np.sqrt((-2 * y**2 + c_val**2 + c_val**2 * y**4 + d) / (2 * y**2 * (c_val**4 + 2 * c_val**2 + 1)))
            return float(a_root1)
        return 0.0

    def e_components(p_val: float, y: float, c_val: float, a_val: float) -> tuple[complex, complex]:
        if abs(y) <= threshold:
            return np.sqrt(p_val * z0), np.sqrt(p_val * z0)
        if y >= 1 - threshold:
            return np.sqrt(p_val * z0), np.sqrt(p_val * z0) * (-1j)
        if y <= -1 + threshold:
            return np.sqrt(p_val * z0), np.sqrt(p_val * z0) * (1j)
        base = np.sqrt((2 * p_val * z0) / (1 + a_val**2 * (1 + c_val**2)))
        return base, base * (a_val * (1 + 1j * c_val))

    with output_path.open("w") as f:
        for i in range(n):
            y_val = gamma[i]
            p_val = p_vals[i]
            c_val = c_coeff(y_val)
            a_val = a_coeff(y_val, c_val)
            e_theta, e_phi = e_components(p_val, y_val, c_val, a_val)
            f.write(
                f"{theta[i]} {phi[i]} {p_vals[i]} "
                f"{np.abs(e_theta)} {np.angle(e_theta)} {np.abs(e_phi)} {np.angle(e_phi)}\n"
            )
    return output_path
