"""Run S1 of the baseline-experiments block.

Stage S1: 5 models x 3 losses x 6 feature pipelines x 3 seeds = 270 runs (or
a smaller subset, controlled by the ``--cells`` and ``--seeds`` arguments).

Each run is a subprocess call to ``uv run mpinv-train`` so failures of one
cell do not bring down the rest. The aggregated results are written as
``experiments/baseline/S1_results.json`` with one entry per
``(model, loss, features, seed)`` cell.

The default behaviour is **conservative**: run only the linear baseline and
the smallest MLP at the smallest feature pipeline, so that the script can be
exercised end-to-end without committing the user's machine to ~270 training
runs. Pass ``--full`` to run the full matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from itertools import product
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_MODELS = ["linear", "mlp_2x16", "mlp_2x64", "mlp_2x256", "mlp_2x512"]
ALL_LOSSES = ["coef_mse", "physics_power", "physics_power_mixed"]
ALL_FEATURES = [
    "power_pca",  # PCA@128 default
    "raw_flat",
    "subsample_stride4",
    "cv_only",  # FFT radial + SH power, skip PCA
    "pca_cv",  # PCA@128 + FFT radial + SH power
    "raw_plus_sh",  # raw_flat + SH power, skip PCA
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic_l15_baseline", type=str)
    p.add_argument("--trainer-max-epochs", default=50, type=int)
    p.add_argument(
        "--seeds",
        default="0,1,2",
        type=str,
        help="Comma-separated seed list.",
    )
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        type=str,
        help="Comma-separated model names.",
    )
    p.add_argument(
        "--losses",
        default=",".join(ALL_LOSSES),
        type=str,
        help="Comma-separated loss names.",
    )
    p.add_argument(
        "--features",
        default=",".join(ALL_FEATURES),
        type=str,
        help="Comma-separated feature pipeline names.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 model, 1 loss, 1 feature, 1 seed, 1 epoch on tiny grid.",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "experiments" / "baseline" / "S1_results.json"),
        type=str,
    )
    p.add_argument(
        "--mlflow",
        default="off",
        choices=["off", "local"],
        help="Whether to log to MLflow ('local' requires a server at 127.0.0.1:5000).",
    )
    p.add_argument(
        "--experiment-name",
        default="mpinv-baseline-S1",
        type=str,
    )
    return p.parse_args()


def cell_overrides(
    model: str, loss: str, feature: str, seed: int
) -> list[str]:
    return [
        f"model={model}",
        f"loss={loss}",
        f"features={feature}",
        f"seed={seed}",
    ]


def run_cell(
    model: str,
    loss: str,
    feature: str,
    seed: int,
    data: str,
    trainer_max_epochs: int,
    mlflow: str,
    experiment_name: str,
) -> dict:
    overrides = [
        f"data={data}",
        f"trainer.max_epochs={trainer_max_epochs}",
        f"tracking={'mlflow_local' if mlflow == 'local' else 'mlflow_off'}",
        f"experiment_name={experiment_name}",
    ] + cell_overrides(model, loss, feature, seed)

    cmd = ["uv", "run", "mpinv-train", *overrides]
    logger.info("running cell: %s / %s / %s / seed=%d", model, loss, feature, seed)
    logger.info("  cmd: %s", " ".join(cmd))

    started = _now()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
        elapsed_s = _now() - started
        ok = result.returncode == 0
        metrics = _parse_report_metrics(result.stdout) if ok else {}
        if not ok:
            logger.warning(
                "cell failed (rc=%d): %s / %s / %s / seed=%d",
                result.returncode,
                model,
                loss,
                feature,
                seed,
            )
            stderr_tail = "\n".join(result.stderr.splitlines()[-15:])
            logger.warning("stderr tail:\n%s", stderr_tail)
        return {
            "model": model,
            "loss": loss,
            "features": feature,
            "seed": seed,
            "ok": ok,
            "returncode": result.returncode,
            "elapsed_s": elapsed_s,
            "metrics": metrics,
            "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]) if not ok else "",
        }
    except subprocess.TimeoutExpired:
        logger.error("cell timed out after 30 minutes; recording as failed")
        return {
            "model": model,
            "loss": loss,
            "features": feature,
            "seed": seed,
            "ok": False,
            "returncode": -1,
            "elapsed_s": _now() - started,
            "metrics": {},
            "stderr_tail": "TIMEOUT",
        }


def _now() -> float:
    import time

    return time.time()


def _parse_report_metrics(stdout: str) -> dict[str, float]:
    """Parse the `report/<tag>/<key> = <value>` lines emitted by train.py."""
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        if "report/" not in line or " = " not in line:
            continue
        idx = line.find("report/")
        try:
            tail = line[idx:]
            key, val = tail.split(" = ")
            out[key.strip()] = float(val.strip())
        except Exception:
            continue
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.smoke:
        models = ["linear"]
        losses = ["coef_mse"]
        features = ["power_pca_small"]
        seeds = [0]
        data = "synthetic_l4_tiny"
        trainer_max_epochs = 1
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        losses = [l.strip() for l in args.losses.split(",") if l.strip()]
        features = [f.strip() for f in args.features.split(",") if f.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        data = args.data
        trainer_max_epochs = args.trainer_max_epochs

    cells = list(product(models, losses, features, seeds))
    logger.info("S1 will run %d cells (M=%d, L=%d, F=%d, S=%d)",
                len(cells), len(models), len(losses), len(features), len(seeds))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (m, l, f, s) in enumerate(cells, 1):
        logger.info("[%d/%d] cell start", i, len(cells))
        rec = run_cell(
            model=m,
            loss=l,
            feature=f,
            seed=s,
            data=data,
            trainer_max_epochs=trainer_max_epochs,
            mlflow=args.mlflow,
            experiment_name=args.experiment_name,
        )
        results.append(rec)
        # Persist every cell — survives interruptions
        out_path.write_text(json.dumps(results, indent=2))
        logger.info(
            "[%d/%d] cell %s ok=%s elapsed=%.1fs",
            i,
            len(cells),
            f"{m}/{l}/{f}/seed={s}",
            rec["ok"],
            rec["elapsed_s"],
        )

    n_ok = sum(1 for r in results if r["ok"])
    logger.info("S1 done: %d / %d cells succeeded", n_ok, len(results))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
