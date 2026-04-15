#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from models.tracking.mlflow_utils import log_basic_metrics, log_pipeline_artifacts, start_run
from src.common.config import VALID_PIPELINE_TEST_STEPS
from src.common.paths import LIBRARY_FAST_DIR, LIBRARY_SLOW_DIR, NAIVE_DIR
from src.pipeline.decompose_fields import decompose_fields
from src.pipeline.generate_fields import generate_fields_latin_square
from src.pipeline.generate_library import generate_fast_library, generate_slow_library
from src.pipeline.visualize import plot_multipoles_3d, show_multipoles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipeline test steps with optional MLflow logging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: 1=library, 2=Fields.txt (Latin), 3=FieldsToMultipoles, 4a=ShowMultipoles, 4b=Plot3DMultipoles",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated list of steps to run (default: all). Example: 1,2,3,4a,4b",
    )
    parser.add_argument(
        "--slow-library",
        action="store_true",
        help="Use slow library generator (Fields0.5) instead of fast (FieldsFast0.5).",
    )
    return parser.parse_args()


def resolve_steps(steps_raw: str | None) -> list[str]:
    if steps_raw is None:
        steps = set(VALID_PIPELINE_TEST_STEPS)
    else:
        steps = {s.strip() for s in steps_raw.split(",") if s.strip()}
        invalid = steps - set(VALID_PIPELINE_TEST_STEPS)
        if invalid:
            print(f"Invalid step(s): {sorted(invalid)}. Valid: {list(VALID_PIPELINE_TEST_STEPS)}", file=sys.stderr)
            sys.exit(1)
    order = {step: idx for idx, step in enumerate(VALID_PIPELINE_TEST_STEPS)}
    return sorted(steps, key=lambda value: order[value])


def main() -> None:
    args = parse_args()
    steps = resolve_steps(args.steps)
    run_start = time.perf_counter()

    params = {"steps": ",".join(steps), "slow_library": args.slow_library}
    with start_run("synthetic_pipeline_test", params=params):
        library_dir = LIBRARY_SLOW_DIR if args.slow_library else LIBRARY_FAST_DIR

        if "1" in steps:
            print("=== 1) Generate multipole library ===")
            library_dir = generate_slow_library() if args.slow_library else generate_fast_library()

        if "2" in steps:
            print("\n=== 2) Generate Fields.txt from latin square ===")
            generate_fields_latin_square(NAIVE_DIR / "Fields.txt")

        if "3" in steps:
            print("\n=== 3) Decompose fields to multipoles ===")
            decompose_fields(field_file="Fields.txt", library_dir=Path(library_dir))

        if "4a" in steps:
            print("\n=== 4a) Show multipoles ===")
            show_multipoles()

        if "4b" in steps:
            print("\n=== 4b) Plot multipoles 3D ===")
            plot_multipoles_3d("Axial_Ratio", results_file=NAIVE_DIR / "Results_Fields.txt")

        log_pipeline_artifacts(
            [
                NAIVE_DIR / "Fields.txt",
                NAIVE_DIR / "Results_Fields.txt",
                Path(library_dir),
            ]
        )
        log_basic_metrics({"runtime_seconds": time.perf_counter() - run_start})

    print("\n=== Pipeline test done. ===")


if __name__ == "__main__":
    main()
