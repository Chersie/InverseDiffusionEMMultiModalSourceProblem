"""Run S3 of the baseline-experiments block: generation-regime ablation.

Given the top-1 cell from S2 (best augmentation included), run it across the
four generation regimes:

- ``gaussian``
- ``colored alpha=1``
- ``colored alpha=2``
- ``sparse 10%``

3 seeds per regime; total 4 x 3 = 12 runs.
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


REGIMES = {
    "gaussian": {"data.generator.mode": "gaussian"},
    "colored_a1": {"data.generator.mode": "colored", "data.generator.color_alpha": 1.0},
    "colored_a2": {"data.generator.mode": "colored", "data.generator.color_alpha": 2.0},
    "sparse_p10": {
        "data.generator.mode": "sparse",
        "data.generator.sparse_active_fraction": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s2-results", required=True, type=str)
    p.add_argument("--metric", default="report/val/coef_mse_amb_aware", type=str)
    p.add_argument("--seeds", default="0,1,2", type=str)
    p.add_argument("--data", default="synthetic_l15_baseline", type=str)
    p.add_argument("--trainer-max-epochs", default=50, type=int)
    p.add_argument("--mlflow", default="off", choices=["off", "local"])
    p.add_argument("--experiment-name", default="mpinv-baseline-S3", type=str)
    p.add_argument("--output",
                   default=str(REPO_ROOT / "experiments" / "baseline" / "S3_results.json"),
                   type=str)
    return p.parse_args()


def pick_top_1(s2: list[dict], metric_key: str) -> dict:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for r in s2:
        if not r.get("ok"):
            continue
        v = r.get("metrics", {}).get(metric_key)
        if v is None:
            continue
        key = (r["model"], r["loss"], r["features"], r["augmentation"])
        grouped.setdefault(key, []).append(float(v))
    if not grouped:
        raise RuntimeError(f"no valid S2 cells at metric {metric_key}")
    best_key, best_vals = min(grouped.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    return {
        "model": best_key[0], "loss": best_key[1], "features": best_key[2],
        "augmentation": best_key[3],
        "mean_metric": sum(best_vals) / len(best_vals),
    }


def run_regime(
    cell: dict, regime_name: str, regime_overrides: dict[str, object], seed: int,
    data: str, trainer_max_epochs: int, mlflow: str, experiment_name: str,
) -> dict:
    cmd = [
        "uv", "run", "mpinv-train",
        f"data={data}",
        f"trainer.max_epochs={trainer_max_epochs}",
        f"tracking={'mlflow_local' if mlflow == 'local' else 'mlflow_off'}",
        f"experiment_name={experiment_name}",
        f"model={cell['model']}",
        f"loss={cell['loss']}",
        f"features={cell['features']}",
        f"seed={seed}",
    ]
    for k, v in regime_overrides.items():
        cmd.append(f"{k}={v}")
    if cell["augmentation"] != "none":
        aug_overrides = {
            "coef_phase_rotation": "+data.augmentation={name: coef_phase_rotation}",
            "coef_additive_noise": "+data.augmentation={name: coef_additive_noise, sigma: 0.05}",
            "field_additive_noise":
                "+data.augmentation={name: field_additive_noise, relative_sigma: 0.02}",
            "field_phi_roll": "+data.augmentation={name: field_phi_roll}",
        }
        cmd.append(aug_overrides[cell["augmentation"]])

    logger.info("S3 cell: regime=%s seed=%d", regime_name, seed)
    started = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True,
            timeout=60 * 30,
        )
        ok = result.returncode == 0
        metrics = _parse_report_metrics(result.stdout) if ok else {}
        return {
            "regime": regime_name, "seed": seed,
            **{k: cell[k] for k in ("model", "loss", "features", "augmentation")},
            "ok": ok, "returncode": result.returncode,
            "elapsed_s": time.time() - started,
            "metrics": metrics,
            "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]) if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "regime": regime_name, "seed": seed,
            **{k: cell[k] for k in ("model", "loss", "features", "augmentation")},
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
    s2 = json.loads(Path(args.s2_results).read_text())
    cell = pick_top_1(s2, args.metric)
    logger.info("S2 best cell: %s", cell)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(REGIMES) * len(seeds)
    i = 0
    for regime_name, regime_overrides in REGIMES.items():
        for seed in seeds:
            i += 1
            logger.info("[%d/%d] starting", i, total)
            rec = run_regime(
                cell=cell,
                regime_name=regime_name,
                regime_overrides=regime_overrides,
                seed=seed,
                data=args.data,
                trainer_max_epochs=args.trainer_max_epochs,
                mlflow=args.mlflow,
                experiment_name=args.experiment_name,
            )
            results.append(rec)
            out_path.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["ok"])
    logger.info("S3 done: %d / %d cells succeeded", n_ok, len(results))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
