"""Train / validation / holdout / dummy split specifications.

A split is described by counts plus the seed used to draw it. The framework's
``mpinv-generate-data`` CLI materialises splits to memmap shards under
``data/processed/<split_name>/``; the in-memory generator path streams them
on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SplitConfig:
    """Counts and seeds for the train/val/holdout-synthetic/dummy splits."""

    n_train: int = 4096
    n_val: int = 512
    n_dummy: int = 64
    seed_train: int = 1234
    seed_val: int = 5678
    seed_dummy: int = 9999


@dataclass(slots=True, frozen=True)
class HoldoutConfig:
    """Real-antenna holdout layout. ``shuffle_seed`` fixes the file ordering bias."""

    root: str = "data/raw/real_antenna"
    feature_subdir: str = "E_in_plane"
    target_glob: str = "Results_*.txt"
    shuffle_seed: int = 42
    max_samples: int | None = None


@dataclass(slots=True, frozen=True)
class FullSplits:
    """All splits of an experiment."""

    train: SplitConfig = field(default_factory=SplitConfig)
    holdout: HoldoutConfig | None = None


def build_splits(cfg: FullSplits | None = None) -> FullSplits:
    return cfg or FullSplits()
