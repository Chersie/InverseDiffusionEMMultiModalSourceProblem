"""Render a 3D power-pattern plot of the *best* model's predictions on a few holdout
real-antenna samples.

Output
------
- ``presentation/figures/holdout_3d_field.pdf``  (vector, defence-quality)
- ``presentation/figures/holdout_3d_field.png``  (raster, slide-friendly)

What the figure shows
---------------------
For each holdout sample (default: 3) we render two 3D surfaces side by side:

- left  — predicted power pattern  P_pred(theta, phi)
- right — measured  power pattern  P_true(theta, phi)

The surface is the unit sphere deformed in the radial direction by
``r = sqrt(P / max(P_pred, P_true))`` — i.e. classical "directivity-style"
3D antenna pattern. Surface colour also encodes ``P``.

Run from repo root (after ``uv sync``)::

    uv run python scripts/plot_holdout_3d_field.py

CLI flags::

    --n-samples       how many holdout samples to render (default 3)
    --first-holdout   first index in the shuffled-by-seed-42 sample list to use as
                       holdout (default 100, matching ``run_real_augmented.py``'s
                       n_train + n_val cut on the 100-sample headline run).
    --checkpoint-dir  override the default best-run checkpoint directory.
    --result-json     override the default best-run result JSON.
    --out             output basename (without extension, default
                       ``presentation/figures/holdout_3d_field``).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; figures are saved
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mpinv.core.grid import GRID_DEFAULT  # noqa: E402
from mpinv.data._basis_cache import build_basis, load_basis  # noqa: E402
from mpinv.data.real_antenna_loader import (  # noqa: E402
    RealAntennaLoaderConfig,
    load_real_antenna,
)
from mpinv.losses.differentiable_field import DifferentiableMultipoleField  # noqa: E402
from run_baseline_inprocess import _build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "experiments" / "baseline"
                    / "figures_real_augmented_best" / "checkpoints"),
        help="directory containing best.pt + feature_extractor.pkl",
    )
    p.add_argument(
        "--result-json",
        default=str(ROOT / "experiments" / "baseline"
                    / "S5_real_augmented_results_best.json"),
        help="run config / metrics JSON; we read l_max, model name, scale_factor",
    )
    p.add_argument(
        "--holdout-root",
        default=str(ROOT / "data" / "raw" / "real_antenna"),
        help="root directory of paired real-antenna samples",
    )
    p.add_argument(
        "--n-samples", default=3, type=int,
        help="how many holdout samples to render",
    )
    p.add_argument(
        "--first-holdout", default=100, type=int,
        help="first index in shuffle_seed=42 ordering treated as holdout",
    )
    p.add_argument(
        "--shuffle-seed", default=42, type=int,
        help="must match the seed used during training to make the index split honest",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "presentation" / "figures" / "holdout_3d_field"),
        help="output basename (without extension)",
    )
    p.add_argument(
        "--checkpoint-name", default="best",
        help='which checkpoint file to use (without ".pt")',
    )
    return p.parse_args()


def _load_model(args, l_max: int, scale_factor: float):
    """Reconstruct the trained model and feature extractor from disk.

    The headline best-run checkpoint dir is missing ``manifest.json``, so we
    rebuild the model from the result JSON's ``model`` field instead of relying
    on ``run_real_augmented.load_run``.
    """
    ckpt_dir = Path(args.checkpoint_dir)
    feat_pkl = ckpt_dir / "feature_extractor.pkl"
    if not feat_pkl.exists():
        raise FileNotFoundError(f"missing feature_extractor.pkl in {ckpt_dir}")
    with feat_pkl.open("rb") as f:
        feat_blob = pickle.load(f)
    feat = feat_blob["feature_extractor"]
    input_dim = int(feat.feature_dim)

    K = l_max * (l_max + 2)
    output_dim = 4 * K

    cfg = json.loads(Path(args.result_json).read_text())
    model_name = cfg.get("model", "mlp_3x200")
    model = _build_model(model_name, input_dim=input_dim, output_dim=output_dim)

    ckpt_path = ckpt_dir / f"{args.checkpoint_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} not found")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, feat, model_name


def _load_holdout_samples(args, l_max: int):
    """Load (P_true, packed_true, sample_id) for a few holdout samples."""
    cfg = RealAntennaLoaderConfig(
        root=args.holdout_root,
        l_max=l_max,
        shuffle_seed=args.shuffle_seed,
        max_samples=None,
    )
    samples = load_real_antenna(cfg)
    n_total = len(samples)
    start = max(0, args.first_holdout)
    end = min(n_total, start + args.n_samples)
    if end <= start:
        # Fallback: just take the first n_samples that exist.
        start, end = 0, min(args.n_samples, n_total)
    picks = samples[start:end]
    if not picks:
        raise RuntimeError(
            f"no real-antenna samples available under {args.holdout_root}"
        )
    return picks


def _predict_p(
    model: torch.nn.Module,
    feat,
    decoder: DifferentiableMultipoleField,
    P_batch: np.ndarray,
    *,
    scale_factor: float,
) -> np.ndarray:
    """Run feature pipeline + model + VSH decoder and return P_pred at original scale."""
    s = float(scale_factor)
    sp = float(np.sqrt(s))
    # Feature pipeline expects the scaled P (it was fit on scaled training data).
    z = feat.transform(P=(P_batch * s).astype(np.float32))
    with torch.no_grad():
        preds = model(torch.from_numpy(z).float())              # (N, 4 K) at scaled units
        # Decoder consumes scaled packed -> scaled P. We undo both scales for plotting.
        preds_unscaled = preds / sp
        P_pred = decoder(preds_unscaled).cpu().numpy()           # (N, n_theta, n_phi)
    return P_pred


def _surface_xyz(P: np.ndarray, theta_rad: np.ndarray, phi_rad: np.ndarray,
                 r_max: float):
    """Convert a P(theta, phi) array into (x, y, z) surface coordinates.

    The surface lives on the unit sphere deformed by ``r = sqrt(P / r_max)``,
    a directivity-style 3D antenna-pattern visualisation.
    """
    TH, PH = np.meshgrid(theta_rad, phi_rad, indexing="ij")
    r = np.sqrt(np.maximum(P, 0.0) / max(r_max, 1e-30))
    X = r * np.sin(TH) * np.cos(PH)
    Y = r * np.sin(TH) * np.sin(PH)
    Z = r * np.cos(TH)
    return X, Y, Z, r


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.result_json).read_text())
    l_max = int(cfg.get("l_max", 5))
    scale_factor = float(cfg.get("scale_factor", 1.0))

    model, feat, model_name = _load_model(args, l_max, scale_factor)

    grid = GRID_DEFAULT
    try:
        basis = load_basis(grid, l_max)
    except Exception:
        basis = build_basis(grid, l_max)
    decoder = DifferentiableMultipoleField(grid=grid, l_max=l_max, basis=basis)
    decoder.eval()

    picks = _load_holdout_samples(args, l_max)
    P_true_batch = np.stack([s.P for s in picks], axis=0).astype(np.float32)
    sids = [s.sample_id for s in picks]
    P_pred_batch = _predict_p(model, feat, decoder, P_true_batch,
                              scale_factor=scale_factor)

    # Spherical grid (radians).
    theta_rad = np.deg2rad(np.linspace(grid.theta_start_deg, grid.theta_end_deg, grid.n_theta))
    phi_rad = np.deg2rad(np.linspace(0.0, 360.0, grid.n_phi, endpoint=False))

    n = len(picks)
    fig = plt.figure(figsize=(11, 4.6 * n))
    cmap = plt.get_cmap("viridis")
    for i, (sid, P_t, P_p) in enumerate(zip(sids, P_true_batch, P_pred_batch)):
        # Common normalisation: same max for both surfaces of one sample.
        r_max = max(float(P_t.max()), float(P_p.max()))
        if r_max <= 0:
            r_max = 1.0
        # Predicted (left).
        ax_p = fig.add_subplot(n, 2, 2 * i + 1, projection="3d")
        Xp, Yp, Zp, rp = _surface_xyz(P_p, theta_rad, phi_rad, r_max)
        ax_p.plot_surface(
            Xp, Yp, Zp, rcount=80, ccount=120,
            facecolors=cmap(rp), shade=False, antialiased=False, linewidth=0,
        )
        ax_p.set_title(
            f"sample {sid} — predicted P\nmax(P_pred) = {P_p.max():.3g}",
            fontsize=11,
        )
        _square_3d_axes(ax_p)

        # True (right).
        ax_t = fig.add_subplot(n, 2, 2 * i + 2, projection="3d")
        Xt, Yt, Zt, rt = _surface_xyz(P_t, theta_rad, phi_rad, r_max)
        ax_t.plot_surface(
            Xt, Yt, Zt, rcount=80, ccount=120,
            facecolors=cmap(rt), shade=False, antialiased=False, linewidth=0,
        )
        ax_t.set_title(
            f"sample {sid} — true P\nmax(P_true) = {P_t.max():.3g}",
            fontsize=11,
        )
        _square_3d_axes(ax_t)

    fig.suptitle(
        f"3D power pattern  —  best run ({model_name} + physics_power_rank, raw_plus_sh)\n"
        f"holdout indices {args.first_holdout}..{args.first_holdout + n - 1}  "
        f"(shuffle_seed={args.shuffle_seed}); radius = sqrt(P / max), shared max per row",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"sample ids   : {sids}")
    print(f"saved PDF    : {pdf_path}")
    print(f"saved PNG    : {png_path}")
    return 0


def _square_3d_axes(ax) -> None:
    """Make a 3D axes look isotropic (no axis-aspect distortion)."""
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.tick_params(labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
