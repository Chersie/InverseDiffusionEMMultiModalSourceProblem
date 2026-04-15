#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from models.tracking.mlflow_utils import log_basic_metrics, log_pipeline_artifacts, start_run
from src.common.paths import LIBRARY_FAST_DIR, NAIVE_DIR
from src.pipeline.decompose_fields import decompose_fields
from src.pipeline.generate_fields import calculate_fields_from_tables
from src.pipeline.generate_library import generate_fast_library
from src.pipeline.inverse_mie_fit import run_inverse_mie


def run_naive_script(script_name: str) -> None:
    from src.common.io_utils import run_python_script

    script = NAIVE_DIR / script_name
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    run_python_script(script, cwd=NAIVE_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Inverse Mie pipeline (non-breaking wrapper).")
    parser.add_argument("--skip-tables", action="store_true", help="Skip Pictures->Tables step.")
    parser.add_argument("--skip-fields", action="store_true", help="Skip Tables->Fields step.")
    parser.add_argument("--skip-decompose", action="store_true", help="Skip Fields->Results step.")
    parser.add_argument("--skip-inverse", action="store_true", help="Skip inverse Mie fit.")
    parser.add_argument(
        "--library",
        default=None,
        help="Path to multipole library (defaults to Chersie/FieldsFast0.5).",
    )
    parser.add_argument("--field-file", default="Fields.txt", help="Field filename in NaiveSolution.")
    parser.add_argument(
        "--generate-library",
        action="store_true",
        help="Generate fast library before decomposition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()

    library = Path(args.library).resolve() if args.library else LIBRARY_FAST_DIR
    field_path = NAIVE_DIR / args.field_file
    results_path = NAIVE_DIR / f"Results_{Path(args.field_file).stem}.txt"

    params = {
        "skip_tables": args.skip_tables,
        "skip_fields": args.skip_fields,
        "skip_decompose": args.skip_decompose,
        "skip_inverse": args.skip_inverse,
        "library": str(library),
        "field_file": args.field_file,
        "generate_library": args.generate_library,
    }
    with start_run("inverse_mie_pipeline", params=params):
        pictures_dir = NAIVE_DIR / "Pictures"
        tables_radiation = NAIVE_DIR / "Tables" / "RadiationPattern.txt"
        tables_axial = NAIVE_DIR / "Tables" / "AxialRatio.txt"
        have_tables = tables_radiation.exists() and tables_axial.exists()

        if args.skip_tables or not pictures_dir.exists():
            print("Step 1: skipped (no Pictures/ or --skip-tables)")
        else:
            print("Step 1: Pictures -> Tables")
            run_naive_script("1 PowerToTable.py")
            run_naive_script("1 AxialRatioToTable.py")
            have_tables = tables_radiation.exists() and tables_axial.exists()

        if args.skip_fields or not have_tables:
            print("Step 2: skipped (no Tables/ or --skip-fields)")
        else:
            print("Step 2: Tables -> Fields.txt")
            calculate_fields_from_tables()

        need_library = (not args.skip_decompose) and field_path.exists()
        if need_library and (args.generate_library or not library.exists()):
            if args.library:
                print(f"Library not found: {library}", file=sys.stderr)
                sys.exit(4)
            print("Step 0: Generate multipole library (fast)")
            library = generate_fast_library()

        if need_library and not library.exists():
            print(f"Library not found: {library}", file=sys.stderr)
            sys.exit(4)

        if args.skip_decompose or not field_path.exists():
            print("Step 3: skipped (no Fields.txt or --skip-decompose)")
        else:
            print("Step 3: Fields + Library -> Results_Fields.txt")
            results_path = decompose_fields(field_file=args.field_file, library_dir=Path(library))

        if args.skip_inverse:
            print("Step 4: skipped (--skip-inverse)")
        elif results_path.exists():
            print("Step 4: Inverse Mie fit")
            try:
                run_inverse_mie(results_path)
            except Exception:
                print("Inverse Mie step failed (non-fatal).", file=sys.stderr)
        else:
            print(f"Results file not found: {results_path}", file=sys.stderr)

        log_pipeline_artifacts([field_path, results_path, Path(library)])
        log_basic_metrics({"runtime_seconds": time.perf_counter() - run_start})

    print("Pipeline done.")


if __name__ == "__main__":
    main()
