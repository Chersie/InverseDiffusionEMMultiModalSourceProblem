"""Clean per-sample distribution plots (r2 + bin accuracy) for two specified runs.

Outputs (under presentation/figures/):

- r2_distribution_<run>.pdf / .png
- bin_accuracy_distribution_<run>.pdf / .png

for ``run in {best, coef_raw_flat}``. Each figure shows two panels —
val_real (left) and holdout_real (right) — with no legend, minimal chrome,
shared x-axis range, median annotation, and a single reference line.

Splits are derived from the run's own ``manifest.json`` when present, falling
back to ``S5_real_augmented_results_<run>.json`` for the headline best run.

Run from repo root::

    uv run python scripts/plot_distributions_clean.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mpinv.analysis.metrics.field_metrics import (  # noqa: E402
    per_sample_bin_accuracy_P,
    per_sample_weighted_r2_P,
)
from mpinv.core.grid import GRID_DEFAULT  # noqa: E402
from mpinv.data._basis_cache import build_basis, load_basis  # noqa: E402
from mpinv.data.real_antenna_loader import (  # noqa: E402
    RealAntennaLoaderConfig,
    load_real_antenna,
)
from mpinv.losses.differentiable_field import DifferentiableMultipoleField  # noqa: E402
from run_baseline_inprocess import _build_model  # noqa: E402

OUT_DIR = ROOT / "presentation" / "figures"


@dataclass(slots=True)
class RunSpec:
    short: str                 # "best" or "coef_raw_flat"
    pretty: str                # title-friendly
    ckpt_dir: Path
    config_json: Path          # manifest.json or S5_*results*.json
    n_train_sources: int       # how many real sources were used for training
    n_val_sources: int         # val_real source count
    n_holdout: int             # how many holdout sources to evaluate on
    checkpoint_name: str = "best"


def _runs() -> list[RunSpec]:
    base = ROOT / "experiments" / "baseline"
    return [
        RunSpec(
            short="best",
            pretty="best (mlp_3x200 + physics_power_rank + raw_plus_sh)",
            ckpt_dir=base / "figures_real_augmented_best" / "checkpoints",
            config_json=base / "S5_real_augmented_results_best.json",
            n_train_sources=80,
            n_val_sources=20,
            n_holdout=100,
            checkpoint_name="best",
        ),
        RunSpec(
            short="coef_raw_flat",
            pretty="coef_raw_flat (mlp_5x200 + coef_mse + raw_flat)",
            ckpt_dir=base / "figures_real_augmented_coef_raw_flat" / "checkpoints",
            config_json=base / "figures_real_augmented_coef_raw_flat"
                       / "checkpoints" / "manifest.json",
            n_train_sources=180,
            n_val_sources=20,
            n_holdout=100,
            checkpoint_name="best",
        ),
    ]


def _load_run(run: RunSpec):
    """Reconstruct (model, feature_extractor, l_max, scale_factor, model_name)."""
    cfg_blob = json.loads(run.config_json.read_text())
    cfg = cfg_blob.get("args", cfg_blob)  # manifest has "args"; result-json is flat
    l_max = int(cfg["l_max"])
    scale_factor = float(cfg.get("scale_factor", 1.0))
    model_name = cfg["model"]

    feat_pkl = run.ckpt_dir / "feature_extractor.pkl"
    with feat_pkl.open("rb") as f:
        feat = pickle.load(f)["feature_extractor"]
    K = l_max * (l_max + 2)
    model = _build_model(model_name, input_dim=int(feat.feature_dim),
                         output_dim=4 * K)
    state = torch.load(run.ckpt_dir / f"{run.checkpoint_name}.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, feat, l_max, scale_factor, model_name


def _predict(model, feat, decoder, P, scale_factor: float, batch_size: int = 64):
    """Run inference: (B, ntheta, nphi) P_true -> P_pred at original units."""
    s = float(scale_factor)
    sp = float(np.sqrt(s))
    z = feat.transform(P=(P * s).astype(np.float32))
    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            zb = torch.from_numpy(z[start:start + batch_size]).float()
            preds = model(zb)                           # scaled packed
            P_pred_scaled = decoder(preds / sp)         # unscaled in packed -> unscaled P
            out_chunks.append(P_pred_scaled.cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _load_real_pool(n_total: int, l_max: int, shuffle_seed: int = 42):
    """Load the first ``n_total`` real-antenna samples in shuffle-seed-42 order."""
    cfg = RealAntennaLoaderConfig(
        root=str(ROOT / "data" / "raw" / "real_antenna"),
        l_max=l_max,
        shuffle_seed=shuffle_seed,
        max_samples=n_total,
    )
    return load_real_antenna(cfg)


def _split_pool(pool, run: RunSpec) -> dict[str, np.ndarray]:
    """Return {'val_real': P, 'holdout_real': P} arrays from a pool list."""
    P = np.stack([s.P for s in pool], axis=0).astype(np.float32)
    n_train = run.n_train_sources
    n_val = run.n_val_sources
    val = P[n_train:n_train + n_val]
    holdout = P[n_train + n_val:]
    return {"val_real": val, "holdout_real": holdout}


def _plot_violin_vertical(
    metrics_per_split: dict[str, np.ndarray],
    *,
    metric_label: str,
    clip_range: tuple[float, float],
    reference_value: float | None,
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Vertical violin + strip plot of per-sample metric values per split.

    Companion to the histogram view: x = split, y = metric. No legend.
    Random/chance reference line is red, dashed; per-split median is a black bar.
    """
    splits = [(name, np.asarray(arr).reshape(-1))
              for name, arr in metrics_per_split.items()
              if np.asarray(arr).size > 0]
    n = len(splits)
    if n == 0:
        return
    palette = ("#3a86ff", "#ffbe0b", "#06d6a0", "#fb5607")
    lo, hi = clip_range

    fig, ax = plt.subplots(figsize=(6.5, 5.4))

    finite_arrays: list[np.ndarray] = []
    labels: list[str] = []
    for name, vals in splits:
        finite = vals[np.isfinite(vals)]
        finite = finite[(finite >= lo) & (finite <= hi)]
        finite_arrays.append(finite)
        labels.append(name)

    # All distributions are stacked at the EXACT same x position — both
    # violin bodies and strip dots — so the two clouds overlay perfectly
    # along a single vertical axis. Distinguishable only by colour.
    x0 = 1.0
    rng = np.random.default_rng(0)
    for i, (name, finite) in enumerate(zip(labels, finite_arrays)):
        if finite.size == 0:
            continue
        color = palette[i % len(palette)]
        # Violin body (skip if no variance — can't be drawn cleanly).
        if finite.size >= 2 and float(np.var(finite)) > 0:
            parts = ax.violinplot(
                finite, positions=[x0], showextrema=False,
                showmedians=False, widths=0.85,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.4)
                body.set_linewidth(1.0)
        # Strip dots: jittered around the SAME x0 (no per-run x-offset).
        jitter = rng.uniform(-0.10, 0.10, size=finite.size)
        ax.scatter(
            x0 + jitter, finite,
            s=16, color=color, edgecolor="black", linewidth=0.3, alpha=0.7,
        )
        # Median bar in the run's colour — no text label, no legend.
        med = float(np.median(finite))
        ax.hlines(med, x0 - 0.42, x0 + 0.42, colors=color, linewidth=2.4)

    # Random/chance reference: horizontal red dashed line.
    if reference_value is not None and lo <= reference_value <= hi:
        ax.axhline(reference_value, color="#d62728",
                   linestyle="--", linewidth=1.4)

    ax.set_xticks([x0])
    ax.set_xticklabels(["holdout_real"], fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_xlim(x0 - 0.7, x0 + 0.7)
    ax.set_ylim(lo, hi)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution(
    metrics_per_split: dict[str, np.ndarray],
    *,
    metric_label: str,
    metric_short: str,
    clip_range: tuple[float, float],
    n_bins: int,
    reference_value: float | None,
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Render a clean 1-row N-panel histogram figure with no legend."""
    splits = [(name, np.asarray(arr).reshape(-1))
              for name, arr in metrics_per_split.items()
              if np.asarray(arr).size > 0]
    n = len(splits)
    if n == 0:
        return
    palette = ("#3a86ff", "#ffbe0b", "#06d6a0", "#fb5607")

    lo, hi = clip_range
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    # Single shared axes; both distributions are overlaid with transparency.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    counts_max = 0
    for i, (name, vals) in enumerate(splits):
        finite = vals[np.isfinite(vals)]
        finite = finite[(finite >= lo) & (finite <= hi)]
        color = palette[i % len(palette)]
        if finite.size:
            counts, _, _ = ax.hist(
                finite, bins=bin_edges, color=color, alpha=0.55,
                edgecolor=color, linewidth=1.0,
            )
            counts_max = max(counts_max, int(counts.max()))
            # Median as a coloured vertical line (no text label / no legend).
            med = float(np.median(finite))
            ax.axvline(med, color=color, linestyle="-", linewidth=1.6)

    # Reference line: random/chance baseline drawn in red.
    if reference_value is not None and lo <= reference_value <= hi:
        ax.axvline(reference_value, color="#d62728",
                   linestyle="--", linewidth=1.4)

    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_xlim(lo, hi)
    if counts_max:
        ax.set_ylim(0, counts_max * 1.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    grid = GRID_DEFAULT

    # Compute per-sample metrics on holdout_real for both runs.
    r2_compare: dict[str, np.ndarray] = {}
    bin_compare: dict[str, np.ndarray] = {}
    n_bins_metric: int | None = None
    pretty_names: dict[str, str] = {}

    for run in _runs():
        print(f"\n=== run: {run.short} ===")
        model, feat, l_max, scale_factor, model_name = _load_run(run)
        try:
            basis = load_basis(grid, l_max)
        except Exception:
            basis = build_basis(grid, l_max)
        decoder = DifferentiableMultipoleField(grid=grid, l_max=l_max, basis=basis)
        decoder.eval()

        n_total = run.n_train_sources + run.n_val_sources + run.n_holdout
        pool = _load_real_pool(n_total, l_max=l_max)
        if not pool:
            print(f"!! no samples found for {run.short}; skipping")
            continue
        if len(pool) < n_total:
            print(f"warn: only {len(pool)}/{n_total} samples available for {run.short}")

        # Holdout split only.
        P_all = np.stack([s.P for s in pool], axis=0).astype(np.float32)
        P_holdout = P_all[run.n_train_sources + run.n_val_sources:]
        if P_holdout.size == 0:
            print(f"!! no holdout samples for {run.short}; skipping")
            continue
        P_pred = _predict(model, feat, decoder, P_holdout,
                          scale_factor=scale_factor)

        n_bm = 2 * l_max + 1 if l_max > 0 else 11
        if n_bins_metric is None:
            n_bins_metric = n_bm
        elif n_bins_metric != n_bm:
            print(f"warn: differing n_bins_metric across runs ({n_bins_metric} vs {n_bm}); using first")

        r2 = per_sample_weighted_r2_P(P_pred, P_holdout, grid=grid)
        ba = per_sample_bin_accuracy_P(P_pred, P_holdout, n_bins=n_bm)
        r2_compare[run.short] = np.asarray(r2, dtype=np.float64)
        bin_compare[run.short] = np.asarray(ba, dtype=np.float64)
        pretty_names[run.short] = run.pretty

        print(f"  R²        holdout_real  n={r2.size:3d}  "
              f"median={float(np.median(r2[np.isfinite(r2)])):+.3f}")
        print(f"  bin_acc   holdout_real  n={ba.size:3d}  "
              f"median={float(np.median(ba)):.3f}")

    if not r2_compare:
        print("no runs produced data; nothing to plot")
        return 1

    n_bm = n_bins_metric or 11
    chance = 1.0 / max(n_bm, 1)

    # ---- R² ----
    _plot_distribution(
        r2_compare,
        metric_label="R² (sin θ-weighted, per sample)",
        metric_short="R²",
        clip_range=(-3.0, 1.0),
        n_bins=30,
        reference_value=0.0,
        title="R² per sample — holdout_real",
        out_pdf=out_dir / "r2_distribution_holdout_compare.pdf",
        out_png=out_dir / "r2_distribution_holdout_compare.png",
    )
    _plot_violin_vertical(
        r2_compare,
        metric_label="R² (sin θ-weighted, per sample)",
        clip_range=(-3.0, 1.0),
        reference_value=0.0,
        title="R² per sample — holdout_real",
        out_pdf=out_dir / "r2_distribution_holdout_compare_violin.pdf",
        out_png=out_dir / "r2_distribution_holdout_compare_violin.png",
    )

    # ---- bin accuracy ----
    _plot_distribution(
        bin_compare,
        metric_label="bin accuracy (per sample)",
        metric_short="bin acc",
        clip_range=(0.0, 1.0),
        n_bins=30,
        reference_value=chance,
        title="Bin accuracy per sample — holdout_real",
        out_pdf=out_dir / "bin_accuracy_distribution_holdout_compare.pdf",
        out_png=out_dir / "bin_accuracy_distribution_holdout_compare.png",
    )
    _plot_violin_vertical(
        bin_compare,
        metric_label="bin accuracy (per sample)",
        clip_range=(0.0, 1.0),
        reference_value=chance,
        title="Bin accuracy per sample — holdout_real",
        out_pdf=out_dir / "bin_accuracy_distribution_holdout_compare_violin.pdf",
        out_png=out_dir / "bin_accuracy_distribution_holdout_compare_violin.png",
    )

    print(f"\nsaved figures under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
