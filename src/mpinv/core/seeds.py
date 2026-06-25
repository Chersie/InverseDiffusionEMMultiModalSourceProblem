"""Global seed management with a context manager for reproducible local seeding."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def RNGContext(seed: int) -> Iterator[np.random.Generator]:
    """Context manager that yields a NumPy ``Generator`` seeded with ``seed``.

    Useful for local determinism without polluting the global RNG state.
    """
    rng = np.random.default_rng(seed)
    yield rng
