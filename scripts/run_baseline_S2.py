"""Run S2 of the baseline-experiments block: augmentation ablation.

Given the top-K cells (by val/coef_mse_amb_aware) from S1, run each cell with
each of 5 augmentation conditions:

- ``none`` (control)
- ``coef_phase_rotation``
- ``coef_additive_noise(sigma=0.05)``
- ``field_additive_noise(relative_sigma=0.02)``
- ``field_phi_roll``

Default 3 seeds per cell-augmentation pair; total 3 cells x 5 augs x 3 seeds = 45.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


AUGMENTATIONS = {
    "none": "+data.augmentation=null",
    "coef_phase_rotation": "+data.augmentation={name: coef_phase_rotation}",
    "coef_additive_noise": "+data.augmentation={name: coef_additive_noise, sigma: 0.05}",
    "field_additive_noise": "+data.augmentation={name: field_additive_noise, relative_sigma: 0.02}",
    "field_phi_roll": "+data.augmentation={name: field_phi_roll}",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-results", required=True, type=str,
                   help="Path to S1_results.json from run_baseline_S1.py.")
    p.add_argument("--top-k", default=3, type=int)
    p.add_argument("--seeds", default="0,1,2", type=str)
    p.add_argument("--data", default="synthetic_l15_baseline", type=str)
    p.add_argument("--trainer-max-epochs", default=50, type=int)
    p.add_argument("--mlflow", default="off", choices=["off", "local"])
    p.add_argument("--experiment-name", default="mpinv-baseline-S2", type=str)
    p.add_argument("--output",
                   default=str(REPO_ROOT / "experiments" / "baseline" / "S2_results.json"),
                   type=str)
    p.add_argument("--metric", default="report/val/coef_mse_amb_aware", type=str)
    return p.parse_args()


def pick_top_k(s1_results: list[dict], k: int, metric_key: str) -> list[dict]:
    """Pick the top-k unique (model, loss, features) triples by mean of `metric_key`."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for r in s1_results:
        if not r.get("ok"):
            continue
        v = r.get("metrics", {}).get(metric_key)
        if v is None:
            continue
        key = (r["model"], r["loss"], r["features"])
        grouped.setdefault(key, []).append(float(v))
    means = [
        (key, sum(vs) / len(vs)) for key, vs in grouped.items() if vs
    ]
    means.sort(key=lambda kv: kv[1])
    return [
        {"model": k[0], "loss": k[1], "features": k[2], "mean_metric": v}
        for k, v in means[:k]
    ]


def run_cell(
    model: str, loss: str, feature: str, seed: int, aug_name: str, aug_override: str,
    data: str, trainer_max_epochs: int, mlflow: str, experiment_name: str,
) -> dict:
    cmd = [
        "uv", "run", "mpinv-train",
        f"data={data}",
        f"trainer.max_epochs={trainer_max_epochs}",
        f"tracking={'mlflow_local' if mlflow == 'local' else 'mlflow_off'}",
        f"experiment_name={experiment_name}",
        f"model={model}",
        f"loss={loss}",
        f"features={feature}",
        f"seed={seed}",
    ]
    if aug_name != "none":
        cmd.append(aug_override)

    logger.info("S2 cell: %s / %s / %s / aug=%s / seed=%d",
                model, loss, feature, aug_name, seed)
    started = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True,
            timeout=60 * 30,
        )
        ok = result.returncode == 0
        metrics = _parse_report_metrics(result.stdout) if ok else {}
        return {
            "model": model, "loss": loss, "features": feature,
            "augmentation": aug_name, "seed": seed,
            "ok": ok, "returncode": result.returncode,
            "elapsed_s": time.time() - started,
            "metrics": metrics,
            "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]) if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model, "loss": loss, "features": feature,
            "augmentation": aug_name, "seed": seed,
            "ok": False, "returncode": -1,
            "elapsed_s": time.time() - started,
            "metrics": {}, "stderr_tail": "TIMEOUT",
        }


def _parse_report_metrics(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        if "report/" not in line or " = " not in line:
            continue
        idx = line.find("report/")
        try:
            tail = line[idx:]
            k, v = tail.split(" = ")
            out[k.strip()] = float(v.strip())
        except Exception:
            continue
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    s1 = json.loads(Path(args.s1_results).read_text())
    top = pick_top_k(s1, args.top_k, args.metric)
    if not top:
        logger.error("no valid S1 cells found at metric %s", args.metric)
        return 1
    logger.info("top %d cells from S1:", len(top))
    for t in top:
        logger.info("  %s/%s/%s -> %s = %.6f",
                    t["model"], t["loss"], t["features"], args.metric, t["mean_metric"])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(top) * len(AUGMENTATIONS) * len(seeds)
    i = 0
    for cell in top:
        for aug_name, aug_override in AUGMENTATIONS.items():
            for seed in seeds:
                i += 1
                logger.info("[%d/%d] starting", i, total)
                rec = run_cell(
                    model=cell["model"], loss=cell["loss"], feature=cell["features"],
                    seed=seed, aug_name=aug_name, aug_override=aug_override,
                    data=args.data, trainer_max_epochs=args.trainer_max_epochs,
                    mlflow=args.mlflow, experiment_name=args.experiment_name,
                )
                results.append(rec)
                out_path.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["ok"])
    logger.info("S2 done: %d / %d cells succeeded", n_ok, len(results))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
