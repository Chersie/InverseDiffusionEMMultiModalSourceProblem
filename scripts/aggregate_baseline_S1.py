"""Aggregate baseline S1/S2/S3 results into a markdown table.

Usage:

    uv run python scripts/aggregate_baseline_S1.py \\
        --results experiments/baseline/S1_results.json \\
        --output experiments/baseline/S1_table.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--key-fields",
        default="model,loss,features",
        type=str,
        help="Comma-separated record fields used to group seeds.",
    )
    p.add_argument(
        "--metrics",
        default=(
            "report/val/coef_mse,"
            "report/val/coef_r2,"
            "report/val/coef_mse_amb_aware,"
            "report/val/field_mse_w,"
            "report/val/field_nrmse_w,"
            "report/test/coef_mse,"
            "report/test/coef_mse_amb_aware,"
            "report/test/field_nrmse_w,"
            "report/holdout/coef_mse_amb_aware,"
            "report/holdout/field_nrmse_w"
        ),
        type=str,
    )
    p.add_argument(
        "--sort-by",
        default="report/val/coef_mse_amb_aware",
        type=str,
    )
    p.add_argument("--ascending", action="store_true", default=True)
    return p.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var)


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "—"
    if abs(v) >= 1e3 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.3e}"
    return f"{v:.4f}"


def main() -> int:
    args = parse_args()
    records = json.loads(Path(args.results).read_text())
    key_fields = [k.strip() for k in args.key_fields.split(",")]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if not r.get("ok"):
            continue
        key = tuple(r.get(k, "?") for k in key_fields)
        ms = r.get("metrics", {})
        for m in metrics:
            v = ms.get(m)
            if v is not None:
                grouped[key][m].append(float(v))

    rows = []
    for key, mdict in grouped.items():
        row = {k: v for k, v in zip(key_fields, key, strict=True)}
        row["n_seeds"] = max((len(vs) for vs in mdict.values()), default=0)
        for m in metrics:
            mu, sd = _mean_std(mdict.get(m, []))
            row[m] = mu
            row[m + "__sd"] = sd
        rows.append(row)

    if args.sort_by in metrics and rows:
        rows.sort(key=lambda r: (r.get(args.sort_by) is None, r.get(args.sort_by, math.inf)),
                  reverse=not args.ascending)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Aggregated results: `{Path(args.results).name}`")
    lines.append("")
    lines.append(f"Sorted by `{args.sort_by}` ({'asc' if args.ascending else 'desc'}).")
    lines.append("")
    header = key_fields + ["n"] + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        cells = [str(r[k]) for k in key_fields]
        cells.append(str(r.get("n_seeds", 0)))
        for m in metrics:
            cells.append(f"{_fmt(r[m])} ± {_fmt(r[m + '__sd'])}")
        lines.append("| " + " | ".join(cells) + " |")

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
