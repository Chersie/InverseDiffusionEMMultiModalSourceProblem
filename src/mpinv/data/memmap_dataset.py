"""Memmap-backed dataset for large synthetic / mixed corpora.

Uses ``numpy.lib.format.open_memmap`` so multiple worker processes can read the same
shard without copying. Each shard owns three arrays:

- ``P.npy``       shape ``(N_shard, n_theta, n_phi)`` float32
- ``packed.npy``  shape ``(N_shard, 4 K)``           float32
- (optional) ``E.npy`` shape ``(N_shard, 2, n_theta, n_phi)`` complex64

Shards live under ``<root>/<split>/`` and are named ``shard_{pid}_{ms}_{idx:06d}.npy``
where ``pid`` and ``ms`` are the producing process id and millisecond timestamp at
shard-creation time. This naming defends against parallel HPO trials that would
otherwise collide if shards were named only by ``idx`` (legacy bug).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset


@dataclass(slots=True, frozen=True)
class MemmapShard:
    """A single shard's three (or two) memmap arrays."""

    P_path: Path
    packed_path: Path
    E_path: Path | None = None

    def open_P(self, mode: str = "r") -> np.ndarray:
        return open_memmap(self.P_path, mode=mode)

    def open_packed(self, mode: str = "r") -> np.ndarray:
        return open_memmap(self.packed_path, mode=mode)

    def open_E(self, mode: str = "r") -> np.ndarray | None:
        if self.E_path is None:
            return None
        return open_memmap(self.E_path, mode=mode)


def shard_token() -> str:
    """A unique-per-process, unique-per-millisecond shard prefix."""
    return f"{os.getpid()}_{int(time.time() * 1000)}"


def write_shard(
    out_dir: Path | str,
    P: np.ndarray,
    packed: np.ndarray,
    E: np.ndarray | None = None,
    shard_idx: int = 0,
    token: str | None = None,
) -> MemmapShard:
    """Write a single shard to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = token or shard_token()
    P_path = out_dir / f"P_{tok}_{shard_idx:06d}.npy"
    pk_path = out_dir / f"packed_{tok}_{shard_idx:06d}.npy"
    np.save(P_path, P.astype(np.float32, copy=False))
    np.save(pk_path, packed.astype(np.float32, copy=False))
    E_path: Path | None = None
    if E is not None:
        E_path = out_dir / f"E_{tok}_{shard_idx:06d}.npy"
        np.save(E_path, E.astype(np.complex64, copy=False))
    return MemmapShard(P_path=P_path, packed_path=pk_path, E_path=E_path)


def list_shards(root: Path | str, kind: str = "P") -> list[Path]:
    """List all shard paths matching ``kind`` (``P`` | ``packed`` | ``E``) under ``root``."""
    root = Path(root)
    return sorted(root.glob(f"{kind}_*_*.npy"))


class MemmapDataset(Dataset):
    """Torch dataset over a list of shards.

    All shards must have matching ``(n_theta, n_phi)``. Indexing computes
    ``(shard_idx, local_idx)`` via the prefix-sum of shard lengths.
    """

    def __init__(self, shards: list[MemmapShard], features: np.ndarray | None = None):
        if not shards:
            raise ValueError("MemmapDataset needs at least one shard")
        self.shards = shards
        self._mm_P: list[np.ndarray] = [s.open_P() for s in shards]
        self._mm_pk: list[np.ndarray] = [s.open_packed() for s in shards]
        # Optional: precomputed feature vectors (e.g. PCA projections); same length.
        self.features = features
        self._lengths = np.array([len(p) for p in self._mm_P], dtype=np.int64)
        self._cumlen = np.concatenate(([0], np.cumsum(self._lengths)))
        self._total = int(self._cumlen[-1])
        if features is not None and len(features) != self._total:
            raise ValueError(f"features length {len(features)} != total samples {self._total}")

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> tuple[int, int]:
        shard_idx = int(np.searchsorted(self._cumlen, idx, side="right") - 1)
        local = idx - int(self._cumlen[shard_idx])
        return shard_idx, local

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx, loc = self._locate(idx)
        P = np.array(self._mm_P[s_idx][loc], copy=True)
        pk = np.array(self._mm_pk[s_idx][loc], copy=True)
        # If no precomputed features are provided, fall back to the flattened P
        # (callers should usually transform with a feature pipeline upstream).
        x = np.array(self.features[idx], copy=True) if self.features is not None else P.reshape(-1)
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(pk, dtype=torch.float32),
            torch.as_tensor(P, dtype=torch.float32),
        )
