"""Tests for the real-antenna loader (synthetic file fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpinv.core.grid import GridSpec
from mpinv.core.packing import iter_modes
from mpinv.data._basis_cache import build_basis
from mpinv.data.real_antenna_loader import (
    RealAntennaLoaderConfig,
    list_real_antenna_samples,
    load_real_antenna,
)
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig


def _write_fake_corpus(tmp_path: Path, grid: GridSpec, l_max: int, n: int = 3) -> None:
    feat_dir = tmp_path / "E_in_plane"
    feat_dir.mkdir()
    basis = build_basis(grid, l_max)
    gen = SyntheticGenerator(SyntheticGeneratorConfig(grid=grid, l_max=l_max), basis=basis)
    rng = np.random.default_rng(0)
    for i in range(n):
        a_e, a_m = gen.sample_coefficients(1, rng)
        E, P = gen.synthesize(a_e, a_m)
        # Reshape into the 7-column file layout (phi outer, theta inner)
        E_th = E[0, 0].T  # (n_phi, n_theta) complex
        E_ph = E[0, 1].T
        rows = []
        for p_idx in range(grid.n_phi):
            for t_idx in range(grid.n_theta):
                theta_deg = grid.theta_start_deg + t_idx * (
                    (grid.theta_end_deg - grid.theta_start_deg) / max(grid.n_theta - 1, 1)
                )
                phi_deg = p_idx * 360.0 / grid.n_phi
                Et = E_th[p_idx, t_idx]
                Ep = E_ph[p_idx, t_idx]
                rows.append(
                    (
                        theta_deg,
                        phi_deg,
                        abs(Et) ** 2 + abs(Ep) ** 2,
                        abs(Et),
                        float(np.rad2deg(np.angle(Et))),
                        abs(Ep),
                        float(np.rad2deg(np.angle(Ep))),
                    )
                )
        arr = np.array(rows, dtype=np.float64)
        np.savetxt(feat_dir / f"sample_{i:03d}.txt", arr, fmt="%.6e", delimiter=" ")
        # Target file
        with open(tmp_path / f"Results_sample_{i:03d}.txt", "w") as f:
            f.write("# Type l m Re Im\n")
            for k, (l, m) in enumerate(iter_modes(l_max)):
                f.write(f"E {l} {m} {a_e[0, k].real:.6e} {a_e[0, k].imag:.6e}\n")
                f.write(f"M {l} {m} {a_m[0, k].real:.6e} {a_m[0, k].imag:.6e}\n")


def test_loader_round_trip(tmp_path):
    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    l_max = 4
    _write_fake_corpus(tmp_path, grid, l_max, n=2)
    cfg = RealAntennaLoaderConfig(
        root=tmp_path, grid=grid, l_max=l_max, max_samples=2, shuffle_seed=0
    )
    samples = load_real_antenna(cfg)
    assert len(samples) == 2
    for s in samples:
        assert s.P.shape == (grid.n_theta, grid.n_phi)
        assert s.E.shape == (2, grid.n_theta, grid.n_phi)
        assert s.packed.shape == (4 * l_max * (l_max + 2),)


def test_loader_shuffle_seed_deterministic(tmp_path):
    grid = GridSpec(n_phi=12, n_theta=8, theta_start_deg=15.0, theta_end_deg=165.0)
    _write_fake_corpus(tmp_path, grid, l_max=4, n=4)
    cfg = RealAntennaLoaderConfig(root=tmp_path, grid=grid, l_max=4, shuffle_seed=42)
    a = [s for _, _, s in list_real_antenna_samples(cfg)]
    b = [s for _, _, s in list_real_antenna_samples(cfg)]
    assert a == b
    cfg2 = RealAntennaLoaderConfig(root=tmp_path, grid=grid, l_max=4, shuffle_seed=99)
    c = [s for _, _, s in list_real_antenna_samples(cfg2)]
    assert sorted(a) == sorted(c)
