"""``mpinv-data-stats``: dataset distribution diagnostic.

Compare two datasets ``A`` and ``B`` (each a synthetic generator regime, a
generated memmap shard, or the real-antenna holdout corpus) along the four
axes that matter for the baseline-experiments S0 stage:

1. Per-block statistics on the packed coefficient vector (``Re aE``, ``Im aE``,
   ``Re aM``, ``Im aM``): mean, std, min, max, quantiles (5/50/95).
2. Per-degree (`l`) energy spectrum: ``sum_m (|aE_lm|^2 + |aM_lm|^2)``.
3. Per-sample power-pattern statistics: mean, std, max, polar-vs-equatorial
   energy ratio (sin theta-weighted bands).
4. Wasserstein-1 distance between A and B for each of the four packed blocks
   (computed via :func:`scipy.stats.wasserstein_distance`).

Usage:

    uv run mpinv-data-stats \\
        --a-spec spec.yaml --b-spec spec.yaml --output-dir experiments/baseline/S0

Each ``spec`` is a small YAML with a ``kind`` key:

- ``kind: synthetic`` — extra keys: ``mode``, ``n``, ``seed``, ``l_max``, ``grid``
  (one of ``full`` | ``tiny``), plus optional ``color_alpha``,
  ``sparse_active_fraction``, ``family_balance``, ``mode_dropout_prob``.
- ``kind: real_antenna`` — extra keys: ``root``, ``feature_subdir``,
  ``target_glob``, ``shuffle_seed``, ``max_samples``.

Output: a directory with ``stats_a.json``, ``stats_b.json``, ``compare.json``
(distances), and three figures (``hist_compare.pdf``, ``per_l_spectrum.pdf``,
``power_summary.pdf``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import wasserstein_distance

from mpinv.core.area_weights import normalised_area_weights
from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import iter_modes
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.real_antenna_loader import RealAntennaLoaderConfig, load_real_antenna
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetSummary:
    """A small JSON-serialisable summary of one dataset."""

    name: str
    n: int
    coef_block_mean: list[float]
    coef_block_std: list[float]
    coef_block_min: list[float]
    coef_block_max: list[float]
    coef_block_q05: list[float]
    coef_block_q50: list[float]
    coef_block_q95: list[float]
    per_l_energy_mean: list[float]
    P_mean: float
    P_std: float
    P_min: float
    P_max: float
    pole_vs_equator_ratio: float


def _grid_from_name(name: str) -> GridSpec:
    if name == "full":
        return GRID_DEFAULT
    if name == "tiny":
        return GridSpec(n_phi=24, n_theta=12, theta_start_deg=7.5, theta_end_deg=172.5)
    raise ValueError(f"unknown grid name {name!r}; use 'full' or 'tiny'")


def _load_dataset(spec: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, GridSpec, int]:
    """Materialise ``(P, packed)`` for the dataset described by ``spec``.

    Returns ``(P, packed, grid, l_max)`` with ``P`` shape ``(N, n_theta, n_phi)``
    and ``packed`` shape ``(N, 4 K)``.
    """
    kind = spec["kind"]
    if kind == "synthetic":
        grid = _grid_from_name(str(spec.get("grid", "full")))
        l_max = int(spec.get("l_max", 15))
        try:
            basis = load_basis(grid, l_max)
        except Exception:
            basis = build_basis(grid, l_max)
        gen_cfg = SyntheticGeneratorConfig(
            grid=grid,
            l_max=l_max,
            mode=spec.get("mode", "gaussian"),
            family_balance=float(spec.get("family_balance", 0.5)),
            coef_scale=float(spec.get("coef_scale", 1.0)),
            color_alpha=float(spec.get("color_alpha", 1.0)),
            sparse_active_fraction=float(spec.get("sparse_active_fraction", 0.1)),
            mode_dropout_prob=float(spec.get("mode_dropout_prob", 0.0)),
            family_balance_jitter=float(spec.get("family_balance_jitter", 0.0)),
        )
        gen = SyntheticGenerator(cfg=gen_cfg, basis=basis)
        n = int(spec.get("n", 4096))
        rng = np.random.default_rng(int(spec.get("seed", 0)))
        P, packed = gen.generate_batch(n, rng)
        return P, packed, grid, l_max
    if kind == "real_antenna":
        grid = _grid_from_name(str(spec.get("grid", "full")))
        l_max = int(spec.get("l_max", 15))
        cfg = RealAntennaLoaderConfig(
            root=str(spec["root"]),
            feature_subdir=str(spec.get("feature_subdir", "E_in_plane")),
            target_glob=str(spec.get("target_glob", "Results_*.txt")),
            grid=grid,
            l_max=l_max,
            shuffle_seed=int(spec.get("shuffle_seed", 42)),
            max_samples=spec.get("max_samples"),
        )
        samples = load_real_antenna(cfg)
        if not samples:
            raise FileNotFoundError(
                f"real-antenna corpus at {cfg.root!r} is empty for {name!r}"
            )
        P = np.stack([s.P for s in samples], axis=0)
        packed = np.stack([s.packed for s in samples], axis=0)
        return P, packed, grid, l_max
    raise ValueError(f"unknown dataset kind {kind!r}")


def summarise(name: str, P: np.ndarray, packed: np.ndarray, grid: GridSpec, l_max: int) -> DatasetSummary:
    K = l_max * (l_max + 2)
    blocks = np.split(packed, 4, axis=-1)
    block_mean = [float(b.mean()) for b in blocks]
    block_std = [float(b.std()) for b in blocks]
    block_min = [float(b.min()) for b in blocks]
    block_max = [float(b.max()) for b in blocks]
    block_q05 = [float(np.quantile(b, 0.05)) for b in blocks]
    block_q50 = [float(np.quantile(b, 0.50)) for b in blocks]
    block_q95 = [float(np.quantile(b, 0.95)) for b in blocks]

    re_e, im_e, re_m, im_m = blocks
    aE_sq = re_e**2 + im_e**2
    aM_sq = re_m**2 + im_m**2
    per_l = np.zeros(l_max, dtype=np.float64)
    for k, (l, _m) in enumerate(iter_modes(l_max)):
        per_l[l - 1] += float(aE_sq[:, k].mean()) + float(aM_sq[:, k].mean())

    aw = normalised_area_weights(grid)
    P_w_mean = float(((P * aw).mean()))
    P_w_std = float(np.sqrt(((P - P.mean()) ** 2 * aw).mean()))
    n_theta = P.shape[1]
    pole_band = slice(0, max(1, n_theta // 6))
    equator_band = slice(max(1, n_theta // 3), max(2, 2 * n_theta // 3))
    pole_e = float((P[:, pole_band, :] * aw[pole_band, :]).mean())
    eq_e = float((P[:, equator_band, :] * aw[equator_band, :]).mean())
    ratio = pole_e / max(eq_e, 1e-12)

    return DatasetSummary(
        name=name,
        n=P.shape[0],
        coef_block_mean=block_mean,
        coef_block_std=block_std,
        coef_block_min=block_min,
        coef_block_max=block_max,
        coef_block_q05=block_q05,
        coef_block_q50=block_q50,
        coef_block_q95=block_q95,
        per_l_energy_mean=[float(x) for x in per_l],
        P_mean=P_w_mean,
        P_std=P_w_std,
        P_min=float(P.min()),
        P_max=float(P.max()),
        pole_vs_equator_ratio=ratio,
    )


def compare_blockwise_w1(
    packed_a: np.ndarray, packed_b: np.ndarray
) -> dict[str, float]:
    """Wasserstein-1 distance per packed block (Re aE, Im aE, Re aM, Im aM)."""
    blocks_a = np.split(packed_a, 4, axis=-1)
    blocks_b = np.split(packed_b, 4, axis=-1)
    names = ("Re aE", "Im aE", "Re aM", "Im aM")
    return {
        name: float(wasserstein_distance(a.ravel(), b.ravel()))
        for name, a, b in zip(names, blocks_a, blocks_b, strict=True)
    }


def _hist_figure(
    packed_a: np.ndarray, packed_b: np.ndarray, name_a: str, name_b: str
):
    blocks_a = np.split(packed_a, 4, axis=-1)
    blocks_b = np.split(packed_b, 4, axis=-1)
    block_names = ("Re aE", "Im aE", "Re aM", "Im aM")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for i, name in enumerate(block_names):
        ax = axes[i // 2, i % 2]
        ax.hist(blocks_a[i].ravel(), bins=60, alpha=0.5, density=True, label=name_a)
        ax.hist(blocks_b[i].ravel(), bins=60, alpha=0.5, density=True, label=name_b)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _per_l_figure(
    summary_a: DatasetSummary, summary_b: DatasetSummary, l_max: int
):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ls = np.arange(1, l_max + 1)
    width = 0.4
    ax.bar(ls - width / 2, summary_a.per_l_energy_mean, width=width, label=summary_a.name)
    ax.bar(ls + width / 2, summary_b.per_l_energy_mean, width=width, label=summary_b.name)
    ax.set_xlabel("l")
    ax.set_ylabel("mean per-l energy")
    ax.set_yscale("log")
    ax.set_xticks(ls)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _power_summary_figure(
    summary_a: DatasetSummary, summary_b: DatasetSummary
):
    keys = ("P_mean", "P_std", "P_max", "pole_vs_equator_ratio")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(keys))
    width = 0.4
    vals_a = [getattr(summary_a, k) for k in keys]
    vals_b = [getattr(summary_b, k) for k in keys]
    ax.bar(x - width / 2, vals_a, width=width, label=summary_a.name)
    ax.bar(x + width / 2, vals_b, width=width, label=summary_b.name)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset distribution diagnostic.")
    parser.add_argument("--a-spec", required=True, type=str)
    parser.add_argument("--b-spec", required=True, type=str)
    parser.add_argument("--a-name", default="A", type=str)
    parser.add_argument("--b-name", default="B", type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.a_spec) as f:
        spec_a = yaml.safe_load(f)
    with open(args.b_spec) as f:
        spec_b = yaml.safe_load(f)

    P_a, pk_a, grid_a, l_max_a = _load_dataset(spec_a, args.a_name)
    P_b, pk_b, grid_b, l_max_b = _load_dataset(spec_b, args.b_name)

    if grid_a != grid_b or l_max_a != l_max_b:
        raise ValueError(
            f"grid/l_max mismatch: A=({grid_a}, L={l_max_a}) vs B=({grid_b}, L={l_max_b})"
        )

    summary_a = summarise(args.a_name, P_a, pk_a, grid_a, l_max_a)
    summary_b = summarise(args.b_name, P_b, pk_b, grid_b, l_max_b)

    distances = compare_blockwise_w1(pk_a, pk_b)
    logger.info("W1 distances per packed block:")
    for k, v in distances.items():
        logger.info("  %s = %.6f", k, v)

    (out / "stats_a.json").write_text(json.dumps(asdict(summary_a), indent=2))
    (out / "stats_b.json").write_text(json.dumps(asdict(summary_b), indent=2))
    (out / "compare.json").write_text(
        json.dumps({"wasserstein_1_per_block": distances}, indent=2)
    )

    fig_hist = _hist_figure(pk_a, pk_b, args.a_name, args.b_name)
    fig_hist.savefig(out / "hist_compare.pdf", bbox_inches="tight")
    plt.close(fig_hist)
    fig_l = _per_l_figure(summary_a, summary_b, l_max_a)
    fig_l.savefig(out / "per_l_spectrum.pdf", bbox_inches="tight")
    plt.close(fig_l)
    fig_p = _power_summary_figure(summary_a, summary_b)
    fig_p.savefig(out / "power_summary.pdf", bbox_inches="tight")
    plt.close(fig_p)

    logger.info("wrote stats and figures under %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
