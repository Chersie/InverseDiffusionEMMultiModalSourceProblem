"""Shared pytest fixtures.

Use a tiny grid (n_phi=12, n_theta=8, L=4) for unit tests so the basis is built in
milliseconds and tests run on CPU without breaking determinism.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from mpinv.core.grid import GridSpec
from mpinv.data._basis_cache import VSHBasis, build_basis
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig


@pytest.fixture(scope="session")
def tiny_grid() -> GridSpec:
    return GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)


@pytest.fixture(scope="session")
def tiny_l_max() -> int:
    return 4


@pytest.fixture(scope="session")
def tiny_basis(tiny_grid: GridSpec, tiny_l_max: int) -> VSHBasis:
    return build_basis(tiny_grid, tiny_l_max)


@pytest.fixture
def tiny_generator(
    tiny_grid: GridSpec, tiny_l_max: int, tiny_basis: VSHBasis
) -> SyntheticGenerator:
    cfg = SyntheticGeneratorConfig(grid=tiny_grid, l_max=tiny_l_max, mode="gaussian")
    return SyntheticGenerator(cfg=cfg, basis=tiny_basis)


@pytest.fixture
def rng() -> Iterator[np.random.Generator]:
    return np.random.default_rng(0)
