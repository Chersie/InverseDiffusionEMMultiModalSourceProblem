"""Tests for ``mpinv.core.grid``."""

from __future__ import annotations

import numpy as np

from mpinv.core.grid import GRID_DEFAULT


def test_default_dimensions():
    assert GRID_DEFAULT.n_phi == 360
    assert GRID_DEFAULT.n_theta == 179
    assert GRID_DEFAULT.n_pixels == 360 * 179
    assert GRID_DEFAULT.real_dof_complex_field == 4 * 360 * 179
    assert GRID_DEFAULT.real_dof_power == 360 * 179


def test_axes_endpoints():
    phi = GRID_DEFAULT.phi_axis()
    assert phi[0] == 0.0
    assert np.isclose(phi[-1], 2 * np.pi - 2 * np.pi / 360)
    th = GRID_DEFAULT.theta_axis()
    assert np.isclose(np.rad2deg(th[0]), 1.0)
    assert np.isclose(np.rad2deg(th[-1]), 179.0)


def test_padded_nlat_and_inner_slice():
    assert GRID_DEFAULT.th_padded_nlat() == 181
    s, e = GRID_DEFAULT.th_inner_slice()
    assert (s, e) == (1, 180)
