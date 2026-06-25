"""Pick the best cell of a given experiment step by composite metric.

Reads ``metrics.json`` files produced by ``mpinv-train`` (one per cell of a
multirun sweep), computes the composite metric

    composite = val_real/field_nrmse_w - 0.5 * val_real/spearman_rho_P

(lower is better), ranks all cells, and writes the top three plus the winner's
configuration pointer to a JSON file. The split prefix (``val_real``,
``val``, ``holdout_real``, ...) is configurable via ``--split-prefix``; the
default is ``val`` to match the keys produced by ``cli/train.py``.

Usage
-----

    uv run python scripts/select_best_step.py \\
        --metrics-glob 'multirun/2026-06-09_*/**/metrics.json' \\
        --split-prefix val \\
        --output paper/final_experiments/step1_winner.json

The output JSON has the shape::

    {
        "winner": {
            "metrics_path": "...",
            "metrics": {...},
            "composite": 0.243,
            "rank": 1
        },
        "top": [
            {"metrics_path": "...", "composite": 0.243, ...},
            ...
        ],
        "metric_definition": {
            "formula": "<split>/field_nrmse_w - 0.5 * <split>/spearman_rho_P",
            "split_prefix": "val",
            "lower_is_better": true
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("select_best_step")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--metrics-glob",
        required=True,
        help="Glob pattern matching metrics.json files. Recursive globs (**) "
             "are supported via Path.glob; double-quote the pattern in shells.",
    )
    p.add_argument(
        "--split-prefix",
        default="val",
        help="Split prefix used by the metric keys (e.g. 'val', 'val_real', "
             "'holdout', 'holdout_real'). Default: 'val'.",
    )
    p.add_argument(
        "--rho-weight",
        default=0.5,
        type=float,
        help="Weight applied to Spearman rho in the composite. "
             "composite = field_nrmse_w - rho_weight * spearman_rho_P. "
             "Default 0.5 - proposal-aligned (regression + ranking) balance.",
    )
    p.add_argument(
        "--top-k",
        default=3,
        type=int,
        help="Number of top cells to record alongside the winner. Default 3.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output JSON path (overwritten).",
    )
    return p.parse_args(argv)


def _load_metrics(path: Path) -> dict[str, float] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        logger.warning("%s has no 'metrics' dict — skipping", path)
        return None
    out: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def composite_score(
    metrics: dict[str, float],
    *,
    split_prefix: str,
    rho_weight: float,
) -> float | None:
    """Compute the composite score for a single cell.

    Returns ``None`` if either of the two required keys is missing or NaN.
    """
    nrmse_key = f"report/{split_prefix}/field_nrmse_w"
    rho_key = f"report/{split_prefix}/spearman_rho_P"
    if nrmse_key not in metrics or rho_key not in metrics:
        return None
    nrmse = metrics[nrmse_key]
    rho = metrics[rho_key]
    if not (math.isfinite(nrmse) and math.isfinite(rho)):
        return None
    return float(nrmse - rho_weight * rho)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ``Path().glob`` only accepts patterns relative to a base directory, so
    # we split the user-supplied glob into its longest non-magic prefix
    # ("anchor") and the remainder ("pattern"). For absolute prefixes the
    # anchor is the deepest existing dir; for relative prefixes we anchor at
    # CWD. Using ``Path.glob`` keeps the call ``Pathlib``-pure (PTH207 satisfied).
    raw = args.metrics_glob
    glob_chars = "*?["
    parts = Path(raw).parts
    anchor_parts: list[str] = []
    pattern_parts: list[str] = []
    seen_magic = False
    for part in parts:
        if seen_magic or any(c in part for c in glob_chars):
            seen_magic = True
            pattern_parts.append(part)
        else:
            anchor_parts.append(part)
    anchor = Path(*anchor_parts) if anchor_parts else Path()
    pattern = "/".join(pattern_parts) if pattern_parts else "*"
    if not anchor.is_absolute():
        anchor = Path.cwd() / anchor if anchor_parts else Path.cwd()
    paths = sorted(p.resolve() for p in anchor.glob(pattern))
    if not paths:
        logger.error("no files matched glob %r", args.metrics_glob)
        return 2

    cells: list[dict[str, Any]] = []
    for path in paths:
        m = _load_metrics(path)
        if m is None:
            continue
        score = composite_score(
            m, split_prefix=args.split_prefix, rho_weight=args.rho_weight
        )
        if score is None:
            logger.warning(
                "skipping %s: missing %r or %r in metrics",
                path,
                f"report/{args.split_prefix}/field_nrmse_w",
                f"report/{args.split_prefix}/spearman_rho_P",
            )
            continue
        cells.append({
            "metrics_path": str(path),
            "metrics": m,
            "composite": score,
            "field_nrmse_w": m[f"report/{args.split_prefix}/field_nrmse_w"],
            "spearman_rho_P": m[f"report/{args.split_prefix}/spearman_rho_P"],
        })

    if not cells:
        logger.error("no usable cells found (composite metric requires both "
                     "field_nrmse_w and spearman_rho_P keys)")
        return 3

    cells.sort(key=lambda c: c["composite"])
    for rank, c in enumerate(cells, start=1):
        c["rank"] = rank

    top_k = max(1, args.top_k)
    payload = {
        "winner": cells[0],
        "top": cells[:top_k],
        "n_cells": len(cells),
        "metric_definition": {
            "formula": (
                f"report/{args.split_prefix}/field_nrmse_w "
                f"- {args.rho_weight} * report/{args.split_prefix}/spearman_rho_P"
            ),
            "split_prefix": args.split_prefix,
            "rho_weight": args.rho_weight,
            "lower_is_better": True,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "ranked %d cells by composite; winner composite=%.6f written to %s",
        len(cells),
        cells[0]["composite"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
