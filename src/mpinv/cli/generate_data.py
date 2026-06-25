"""``mpinv-generate-data``: offline materialisation of a synthetic dataset to memmap shards.

Usage:

    uv run mpinv-generate-data --output-dir data/processed/train --n-total 100000 \
        --n-per-shard 4096 --grid full --l-max 15 --seed 1234

Each shard yields three ``.npy`` files (``P``, ``packed``, optionally ``E``) named
with a ``{pid}_{ms}_{idx}`` token so concurrent producers do not collide.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.memmap_dataset import shard_token, write_shard
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)

logger = logging.getLogger(__name__)


def _grid_from_name(name: str) -> GridSpec:
    if name == "full":
        return GRID_DEFAULT
    if name == "tiny":
        return GridSpec(n_phi=24, n_theta=12, theta_start_deg=7.5, theta_end_deg=172.5)
    raise ValueError(f"unknown grid name {name!r}; use 'full' or 'tiny'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialise a synthetic dataset to memmap shards."
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--n-total", type=int, default=4096)
    parser.add_argument("--n-per-shard", type=int, default=1024)
    parser.add_argument("--grid", type=str, default="full")
    parser.add_argument("--l-max", type=int, default=15)
    parser.add_argument("--mode", type=str, default="gaussian")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--store-fields", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    grid = _grid_from_name(args.grid)
    try:
        basis = load_basis(grid, args.l_max)
    except Exception:
        basis = build_basis(grid, args.l_max)
    cfg = SyntheticGeneratorConfig(grid=grid, l_max=args.l_max, mode=args.mode)
    gen = SyntheticGenerator(cfg=cfg, basis=basis)

    rng = np.random.default_rng(args.seed)
    tok = shard_token()
    n_done = 0
    shard_idx = 0
    while n_done < args.n_total:
        n_this = min(args.n_per_shard, args.n_total - n_done)
        if args.store_fields:
            E, P, packed = gen.generate_batch_with_field(n_this, rng)
        else:
            E = None
            P, packed = gen.generate_batch(n_this, rng)
        shard = write_shard(
            args.output_dir,
            P=P,
            packed=packed,
            E=E,
            shard_idx=shard_idx,
            token=tok,
        )
        logger.info("wrote shard %d (n=%d): %s", shard_idx, n_this, shard.P_path.name)
        n_done += n_this
        shard_idx += 1
    logger.info("done: %d samples in %d shards under %s", n_done, shard_idx, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
