"""Real-antenna far-field loader.

The reference data layout (defined in
``presentation/figures.md`` / R5 of the research manifest) is a directory of plain-
text files with a seven-column row format::

    theta_deg  phi_deg  power  |E_theta|  arg(E_theta)_deg  |E_phi|  arg(E_phi)_deg

with one row per ``(theta, phi)`` grid sample. Phase columns are in **degrees** in
the file format and are converted to **radians** here, exactly once. After this
loader, all downstream code sees radians (see [AGENTS.md](../../../AGENTS.md)).

Targets are paired ``Results_*.txt`` files of the form::

    Type  l  m  Re  Im

where ``Type ∈ {E, M}``. Each row is one complex multipole coefficient.

This loader fixes the legacy framework's lexicographic-ordering bias by accepting
a deterministic ``shuffle_seed`` and applying a stable shuffle to the file list
before sub-sampling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import iter_modes, pack_coefficients

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RealAntennaLoaderConfig:
    """Where to find the real-antenna corpus and how to subsample it."""

    root: str | Path
    feature_subdir: str = "E_in_plane"
    target_glob: str = "Results_*.txt"
    grid: GridSpec = field(default_factory=lambda: GRID_DEFAULT)
    l_max: int = 15
    shuffle_seed: int = 42
    max_samples: int | None = None


@dataclass(slots=True, frozen=True)
class RealAntennaSample:
    """One loaded sample with both the field and the target coefficients."""

    sample_id: str
    P: np.ndarray  # (n_theta, n_phi)
    E: np.ndarray  # (2, n_theta, n_phi) complex
    packed: np.ndarray  # (4 K,)


def _parse_feature_file(path: Path, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Parse a 7-column field file. Returns ``(P, E)`` with E complex.

    Phase units in the file are degrees; we convert to radians here, once.
    """
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"{path}: expected 7-column file, got shape {arr.shape}")
    if arr.shape[0] != grid.n_phi * grid.n_theta:
        raise ValueError(
            f"{path}: row count {arr.shape[0]} != grid.n_phi * grid.n_theta = "
            f"{grid.n_phi * grid.n_theta}"
        )
    # Reshape into (n_phi, n_theta) outer-loop (matches legacy convention) then
    # transpose into the framework's canonical (n_theta, n_phi) layout.
    rows = arr.reshape(grid.n_phi, grid.n_theta, 7)
    P = rows[..., 2].astype(np.float32).T  # (n_theta, n_phi)
    abs_th = rows[..., 3].astype(np.float64).T
    arg_th = np.deg2rad(rows[..., 4]).astype(np.float64).T
    abs_ph = rows[..., 5].astype(np.float64).T
    arg_ph = np.deg2rad(rows[..., 6]).astype(np.float64).T
    E_theta = (abs_th * np.exp(1j * arg_th)).astype(np.complex64)
    E_phi = (abs_ph * np.exp(1j * arg_ph)).astype(np.complex64)
    E = np.stack((E_theta, E_phi), axis=0)  # (2, n_theta, n_phi)
    return P, E


def _parse_target_file(path: Path, l_max: int) -> np.ndarray:
    """Parse a ``Results_*.txt`` target file into a packed real coefficient vector."""
    K = l_max * (l_max + 2)
    a_e = np.zeros(K, dtype=np.complex64)
    a_m = np.zeros(K, dtype=np.complex64)
    mode_idx = {(l, m): k for k, (l, m) in enumerate(iter_modes(l_max))}
    rx = re.compile(r"^\s*([EM])\s+(-?\d+)\s+(-?\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    with path.open() as f:
        for line in f:
            m = rx.match(line)
            if not m:
                continue
            kind, l_s, m_s, re_s, im_s = m.groups()
            l = int(l_s)
            mm = int(m_s)
            if l < 1 or l > l_max:
                continue
            if abs(mm) > l:
                continue
            k = mode_idx[(l, mm)]
            v = complex(float(re_s), float(im_s))
            if kind == "E":
                a_e[k] = v
            else:
                a_m[k] = v
    return pack_coefficients(a_e[None], a_m[None])[0]


def list_real_antenna_samples(cfg: RealAntennaLoaderConfig) -> list[tuple[Path, Path, str]]:
    """Discover ``(feature_path, target_path, sample_id)`` triples."""
    root = Path(cfg.root)
    feat_dir = root / cfg.feature_subdir
    feat_paths = sorted(feat_dir.glob("*.txt"))
    pairs: list[tuple[Path, Path, str]] = []
    for fp in feat_paths:
        sid = fp.stem
        target = root / f"Results_{sid}.txt"
        if not target.exists():
            target_alt = next(root.glob(f"Results_*{sid}*.txt"), None)
            if target_alt is None:
                logger.debug("skipping %s: no matching target file", fp.name)
                continue
            target = target_alt
        pairs.append((fp, target, sid))
    if cfg.shuffle_seed >= 0:
        rng = np.random.default_rng(cfg.shuffle_seed)
        idx = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idx]
    if cfg.max_samples is not None:
        pairs = pairs[: cfg.max_samples]
    return pairs


def load_real_antenna(cfg: RealAntennaLoaderConfig) -> list[RealAntennaSample]:
    """Load every paired ``(feature, target)`` file under ``cfg.root``.

    Returns an in-memory list of :class:`RealAntennaSample`. For very large corpora
    callers should iterate :func:`iter_real_antenna` instead.
    """
    return list(iter_real_antenna(cfg))


def iter_real_antenna(cfg: RealAntennaLoaderConfig):
    """Yield :class:`RealAntennaSample` one at a time."""
    for fp, tp, sid in list_real_antenna_samples(cfg):
        try:
            P, E = _parse_feature_file(fp, cfg.grid)
        except Exception as exc:
            logger.warning("skipping %s: %s", fp, exc)
            continue
        try:
            packed = _parse_target_file(tp, cfg.l_max)
        except Exception as exc:
            logger.warning("skipping %s: target parse failed: %s", tp, exc)
            continue
        yield RealAntennaSample(sample_id=sid, P=P, E=E, packed=packed)
