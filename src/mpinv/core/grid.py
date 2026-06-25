"""Angular grid specification.

The project's grid (cf. presentation/ch1_full.md §1.2) is the 1° equiangular grid with
the polar caps excluded:

    phi   in 0°, 1°, ..., 359°       (n_phi   = 360)
    theta in 1°, 2°, ..., 179°       (n_theta = 179)

The two polar samples theta = 0° and theta = 180° are dropped because the spherical area
element sin(theta) d_theta d_phi vanishes there.

This file is the single source of truth for the grid; every other module imports
``GRID_DEFAULT`` rather than re-declaring shapes inline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Specification of a uniform spherical sampling grid in (phi, theta).

    Attributes
    ----------
    n_phi : int
        Number of azimuthal samples; equally spaced in [0, 2 pi).
    n_theta : int
        Number of polar samples; equally spaced in (theta_start, theta_end).
    theta_start_deg : float
        First polar sample in degrees (>= 0). For the project default this is 1.0.
    theta_end_deg : float
        Last polar sample in degrees (<= 180). For the project default this is 179.0.
    """

    n_phi: int = 360
    n_theta: int = 179
    theta_start_deg: float = 1.0
    theta_end_deg: float = 179.0

    @property
    def n_pixels(self) -> int:
        return self.n_phi * self.n_theta

    @property
    def real_dof_power(self) -> int:
        """Real degrees of freedom of the dual-polarisation power pattern P."""
        return self.n_pixels

    @property
    def real_dof_complex_field(self) -> int:
        """Real degrees of freedom of the complex tangential field (E_theta, E_phi)."""
        return 4 * self.n_pixels

    @property
    def dphi(self) -> float:
        """Azimuthal sample spacing in radians."""
        return 2.0 * np.pi / self.n_phi

    @property
    def dtheta(self) -> float:
        """Polar sample spacing in radians."""
        if self.n_theta == 1:
            return np.pi
        return np.deg2rad(self.theta_end_deg - self.theta_start_deg) / (self.n_theta - 1)

    def phi_axis(self) -> np.ndarray:
        """Return phi samples in radians, shape (n_phi,)."""
        return np.linspace(0.0, 2.0 * np.pi, self.n_phi, endpoint=False)

    def theta_axis(self) -> np.ndarray:
        """Return theta samples in radians, shape (n_theta,)."""
        return np.deg2rad(np.linspace(self.theta_start_deg, self.theta_end_deg, self.n_theta))

    def th_padded_nlat(self) -> int:
        """Polar dimension after padding to a poles-included equiangular grid.

        For the default 1° grid we pad to nlat = 181 (theta = 0°..180° in 1° steps);
        the inner slice ``[1:180]`` reproduces the project grid. See R1 in
        ``research/framework-rebuild/manifest.md``.
        """
        if self.theta_start_deg == 1.0 and self.theta_end_deg == 179.0:
            return self.n_theta + 2
        raise ValueError(
            "th_padded_nlat is defined only for the canonical 1° pole-excluded grid; "
            f"got theta in [{self.theta_start_deg}, {self.theta_end_deg}]"
        )

    def th_inner_slice(self) -> tuple[int, int]:
        """Slice indices ``(start, stop)`` into the padded grid that recover this grid."""
        if self.theta_start_deg == 1.0 and self.theta_end_deg == 179.0:
            return (1, self.n_theta + 1)
        raise ValueError("th_inner_slice is defined only for the canonical 1° pole-excluded grid")


GRID_DEFAULT = GridSpec()
"""Project default: ``n_phi=360``, ``n_theta=179``, theta in [1°, 179°]."""
