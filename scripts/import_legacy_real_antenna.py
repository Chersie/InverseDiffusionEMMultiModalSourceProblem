"""One-shot converter: legacy real-antenna corpora → project layout.

The legacy multipole files are written in a *different* VSH convention than
the project's basis (extra ``-(1j)^(l+1)`` global phase per mode, sign flip
on the electric family). Reusing those coefficients with the project's
forward operator produces inconsistent ``(P, packed)`` pairs — empirically
``||P_meas|| / ||P_resyn||`` ≈ 250x on the holdout corpus despite the data
being well within bandlimit. To keep the augmentation contract (consistent
``(P, packed)`` under the project's basis), this importer **discards the
legacy multipole files** and instead:

1. Loads the complex tangential field ``(E_theta, E_phi)`` from the legacy
   field text file (cols 3–6: ``|E_theta|, arg(E_theta), |E_phi|, arg(E_phi)``,
   phase in degrees).
2. Projects ``E`` onto the project's VSH basis at ``--l-max`` via
   :func:`mpinv.data.basis_decomposer.decompose_field_to_packed` to obtain
   coefficients in the project's convention.
3. Re-synthesises the *bandlimited* ``E_resyn`` and ``P_resyn = |E_resyn|^2``
   from those coefficients, so the written ``(P, packed)`` pair is exact
   under the project's basis at the chosen bandlimit.
4. Writes a 7-column field file using ``E_resyn`` and a project-convention
   ``Results_<sample_id>.txt``. Loaders downstream see strictly-consistent
   pairs and don't need to know about the legacy convention.

The high-``l`` content of the original measurement (energy outside
``l <= l_max``) is dropped at this step. The diagnostic logger reports the
mean and max relative residual ``||P_meas - P_resyn||_2 / ||P_meas||_2`` per
batch so you can decide whether the chosen ``l_max`` captures enough.

Two source layouts are supported via ``--layout``.

``in_plane`` (default, the actual holdout)
------------------------------------------
Source: e.g. ``/Users/chersie/Desktop/diplom_dump/E+multip``::

    <src>/E_in_plane/<id>.txt                       # 8-col legacy field file
    <src>/Multipoles_in_plane/Results_<id>.txt      # NOT USED (wrong basis)

``legacy_rotation``
-------------------
Source: e.g. ``~/Desktop/diplom/data/external/Rotation``::

    <src>/Fields/<rotation>/<id>.txt
    <src>/Multipoles/<rotation>/<id>.txt            # NOT USED (wrong basis)

Sample IDs are prefixed with the rotation label (e.g. ``XY_0_1010``) because
the same numeric ``<id>`` exists in both rotations.

Target layout (both source layouts collapse to this)::

    <dst>/E_in_plane/<sample_id>.txt                # 7-col, P_resyn
    <dst>/Results_<sample_id>.txt                   # project-convention coefs

Usage::

    uv run python scripts/import_legacy_real_antenna.py
    uv run python scripts/import_legacy_real_antenna.py --l-max 5 --clear-target
    uv run python scripts/import_legacy_real_antenna.py --dry-run

Idempotent: re-running overwrites target files but does not touch the source.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from mpinv.core.grid import GridSpec
from mpinv.core.packing import iter_modes, unpack_coefficients
from mpinv.data._basis_cache import VSHBasis, build_basis, load_basis
from mpinv.data.basis_decomposer import decompose_field_to_packed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--layout",
        default="in_plane",
        choices=["in_plane", "legacy_rotation"],
        help="Source layout selector. ``in_plane`` for the actual holdout "
        "(E_in_plane/ + Multipoles_in_plane/Results_<id>.txt), "
        "``legacy_rotation`` for Fields/<rot>/ + Multipoles/<rot>/<id>.txt.",
    )
    p.add_argument(
        "--legacy-root",
        default=str(
            Path("/Users/chersie/Desktop/diplom_dump/E+multip").expanduser()
        ),
        type=str,
        help="Root containing the source corpus. See module docstring for the "
        "expected sub-layout per ``--layout`` value.",
    )
    p.add_argument(
        "--target-root",
        default="data/raw/real_antenna",
        type=str,
        help="Project-side target root; expected by the loader.",
    )
    p.add_argument(
        "--rotations",
        default="Rotation_XY,Rotation_YZ",
        type=str,
        help="(legacy_rotation only) Comma-separated rotation directories.",
    )
    p.add_argument("--n-theta", default=179, type=int)
    p.add_argument("--n-phi", default=360, type=int)
    p.add_argument(
        "--l-max",
        default=5,
        type=int,
        help="Bandlimit at which to project the measured E onto the project's basis.",
    )
    p.add_argument(
        "--clear-target",
        action="store_true",
        help="Remove target-root before importing (clean replacement).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk and validate without writing anything.",
    )
    return p.parse_args()


def _parse_legacy_field_file(
    src: Path, *, grid: GridSpec, n_expected_rows: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a legacy (7- or 8-col) field file; return ``(E_theta, E_phi)`` complex.

    Returns ``None`` on parse / shape failure.
    """
    try:
        data = np.loadtxt(src, dtype=np.float64)
    except Exception as exc:
        logger.warning("  loadtxt failed for %s: %s", src.name, exc)
        return None
    if data.ndim != 2 or data.shape[0] != n_expected_rows:
        logger.warning("  %s: shape %s != expected (%d, 7|8)", src.name,
                       data.shape, n_expected_rows)
        return None
    if data.shape[1] not in (7, 8):
        logger.warning("  %s: %d cols, expected 7 or 8", src.name, data.shape[1])
        return None
    rows = data.reshape(grid.n_phi, grid.n_theta, data.shape[1])
    abs_th = rows[..., 3].astype(np.float64).T
    arg_th = np.deg2rad(rows[..., 4]).astype(np.float64).T
    abs_ph = rows[..., 5].astype(np.float64).T
    arg_ph = np.deg2rad(rows[..., 6]).astype(np.float64).T
    E_theta = (abs_th * np.exp(1j * arg_th)).astype(np.complex64)
    E_phi = (abs_ph * np.exp(1j * arg_ph)).astype(np.complex64)
    return E_theta, E_phi


def _write_field_file_from_E(
    path: Path, E_theta: np.ndarray, E_phi: np.ndarray, *, grid: GridSpec
) -> None:
    """Write a 7-column field file in the project's expected layout.

    Row order is ``(phi, theta)`` outer×inner, matching what the project's
    loader (:func:`mpinv.data.real_antenna_loader._parse_feature_file`)
    expects when it reshapes into ``(n_phi, n_theta, 7)``.
    """
    theta_axis = np.rad2deg(grid.theta_axis())   # (n_theta,)
    phi_axis = np.rad2deg(grid.phi_axis())       # (n_phi,)
    theta_grid = np.broadcast_to(theta_axis[None, :], (grid.n_phi, grid.n_theta))
    phi_grid = np.broadcast_to(phi_axis[:, None], (grid.n_phi, grid.n_theta))
    P = (np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2).astype(np.float64)
    P_phi_theta = P.T  # (n_phi, n_theta)
    abs_th = np.abs(E_theta).astype(np.float64).T
    arg_th = np.rad2deg(np.angle(E_theta)).astype(np.float64).T
    abs_ph = np.abs(E_phi).astype(np.float64).T
    arg_ph = np.rad2deg(np.angle(E_phi)).astype(np.float64).T
    rows = np.stack(
        (
            theta_grid.ravel(),
            phi_grid.ravel(),
            P_phi_theta.ravel(),
            abs_th.ravel(),
            arg_th.ravel(),
            abs_ph.ravel(),
            arg_ph.ravel(),
        ),
        axis=1,
    )
    np.savetxt(path, rows, fmt="%.6e")


def _write_results_file(path: Path, packed: np.ndarray, *, l_max: int) -> None:
    """Write a project-convention ``Results_<id>.txt`` (Type l m Re Im)."""
    a_e, a_m = unpack_coefficients(packed[None])
    a_e = a_e[0]
    a_m = a_m[0]
    with path.open("w") as f:
        for k, (l, m) in enumerate(iter_modes(l_max)):
            f.write(f"E {l} {m} {a_e[k].real: .18e} {a_e[k].imag: .18e}\n")
            f.write(f"M {l} {m} {a_m[k].real: .18e} {a_m[k].imag: .18e}\n")


def _decompose_and_write(
    src_field: Path,
    dst_field: Path,
    dst_results: Path,
    *,
    grid: GridSpec,
    basis: VSHBasis,
    l_max: int,
) -> tuple[bool, str, dict[str, float]]:
    """End-to-end: parse legacy field → decompose → write project-convention pair.

    Returns ``(ok, message, diag)`` where ``diag`` carries per-sample residual
    diagnostics (relative ``P`` and ``E`` reconstruction errors at the chosen
    ``l_max``).
    """
    parsed = _parse_legacy_field_file(
        src_field, grid=grid, n_expected_rows=grid.n_pixels
    )
    if parsed is None:
        return False, "parse failed", {}
    E_theta, E_phi = parsed
    packed = decompose_field_to_packed(
        E_theta, E_phi, basis=basis, grid=grid
    )
    a_e, a_m = unpack_coefficients(packed[None])
    E_e_resyn = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m_resyn = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E_resyn = (E_e_resyn + E_m_resyn)[0]
    E_theta_resyn = E_resyn[0].astype(np.complex64)
    E_phi_resyn = E_resyn[1].astype(np.complex64)
    P_meas = (np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2).astype(np.float64)
    P_resyn = (np.abs(E_theta_resyn) ** 2 + np.abs(E_phi_resyn) ** 2).astype(
        np.float64
    )
    p_norm = float(np.linalg.norm(P_meas))
    p_resid = float(np.linalg.norm(P_meas - P_resyn))
    p_rel = p_resid / max(p_norm, 1e-30)
    e_norm = float(np.linalg.norm(np.stack([E_theta, E_phi])))
    e_resid = float(np.linalg.norm(
        np.stack([E_theta - E_theta_resyn, E_phi - E_phi_resyn])
    ))
    e_rel = e_resid / max(e_norm, 1e-30)
    diag = {
        "p_norm": p_norm,
        "p_rel_rmse": p_rel,
        "e_norm": e_norm,
        "e_rel_rmse": e_rel,
    }
    _write_field_file_from_E(dst_field, E_theta_resyn, E_phi_resyn, grid=grid)
    _write_results_file(dst_results, packed, l_max=l_max)
    return True, "ok", diag


def _import_in_plane(
    args: argparse.Namespace,
    src_root: Path,
    dst_root: Path,
    dst_features: Path,
    *,
    grid: GridSpec,
    basis: VSHBasis,
) -> tuple[int, list[tuple[str, str]], list[dict[str, float]]]:
    """Walk the ``E_in_plane/`` legacy layout.

    The original ``Multipoles_in_plane/Results_<id>.txt`` files are *not*
    consulted; we re-derive ``packed`` from the measured ``E`` via the
    project's basis (see module docstring).
    """
    fields_dir = src_root / "E_in_plane"
    if not fields_dir.is_dir():
        logger.error("expected %s under %s", fields_dir, src_root)
        return 0, [], []
    field_files = sorted(fields_dir.glob("*.txt"))
    logger.info("in_plane corpus: %d field files found", len(field_files))
    converted = 0
    skipped: list[tuple[str, str]] = []
    diags: list[dict[str, float]] = []
    for i, fp in enumerate(field_files):
        sid = fp.stem
        target_feature = dst_features / f"{sid}.txt"
        target_results = dst_root / f"Results_{sid}.txt"
        if args.dry_run:
            logger.info("DRY RUN: would write %s and %s",
                        target_feature, target_results)
            converted += 1
            continue
        ok, msg, diag = _decompose_and_write(
            fp, target_feature, target_results,
            grid=grid, basis=basis, l_max=args.l_max,
        )
        if not ok:
            skipped.append((sid, msg))
            continue
        diags.append(diag)
        converted += 1
        if (i + 1) % 50 == 0:
            logger.info("  ... %d / %d converted", i + 1, len(field_files))
    return converted, skipped, diags


def _import_legacy_rotation(
    args: argparse.Namespace,
    src_root: Path,
    dst_root: Path,
    dst_features: Path,
    *,
    grid: GridSpec,
    basis: VSHBasis,
) -> tuple[int, list[tuple[str, str]], list[dict[str, float]]]:
    """Walk the ``Fields/<rot>/`` legacy rotation layout."""
    rotations = [r.strip() for r in args.rotations.split(",") if r.strip()]
    converted = 0
    skipped: list[tuple[str, str]] = []
    diags: list[dict[str, float]] = []
    for rot in rotations:
        prefix = rot.replace("Rotation_", "")
        fields_dir = src_root / "Fields" / rot
        if not fields_dir.is_dir():
            logger.warning("skip %s: %s missing", rot, fields_dir)
            continue
        field_files = sorted(fields_dir.glob("*.txt"))
        logger.info("%s: %d field files found", rot, len(field_files))
        for i, fp in enumerate(field_files):
            sid = fp.stem
            new_sid = f"{prefix}_{sid}"
            target_feature = dst_features / f"{new_sid}.txt"
            target_results = dst_root / f"Results_{new_sid}.txt"
            if args.dry_run:
                logger.info("DRY RUN: would write %s and %s",
                            target_feature, target_results)
                converted += 1
                continue
            ok, msg, diag = _decompose_and_write(
                fp, target_feature, target_results,
                grid=grid, basis=basis, l_max=args.l_max,
            )
            if not ok:
                skipped.append((new_sid, msg))
                continue
            diags.append(diag)
            converted += 1
            if (i + 1) % 50 == 0:
                logger.info("  ... %d / %d converted", i + 1, len(field_files))
    return converted, skipped, diags


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    src_root = Path(args.legacy_root)
    dst_root = Path(args.target_root)
    dst_features = dst_root / "E_in_plane"

    if not src_root.is_dir():
        logger.error("legacy root not found: %s", src_root)
        return 2

    if args.clear_target and not args.dry_run:
        if dst_root.exists():
            logger.info("clearing target root %s", dst_root)
            shutil.rmtree(dst_root)
    if not args.dry_run:
        dst_features.mkdir(parents=True, exist_ok=True)
        dst_root.mkdir(parents=True, exist_ok=True)

    grid = GridSpec(
        n_phi=args.n_phi, n_theta=args.n_theta,
        theta_start_deg=1.0, theta_end_deg=179.0,
    )
    if not args.dry_run:
        try:
            basis = load_basis(grid, args.l_max)
        except Exception:
            basis = build_basis(grid, args.l_max)
    else:
        basis = None  # type: ignore[assignment]

    if args.layout == "in_plane":
        converted, skipped, diags = _import_in_plane(
            args, src_root, dst_root, dst_features, grid=grid, basis=basis,
        )
        layout_note = (
            "Source layout: ``E_in_plane/<id>.txt`` (legacy multipole files "
            "ignored; coefficients re-derived in the project's basis)."
        )
    elif args.layout == "legacy_rotation":
        converted, skipped, diags = _import_legacy_rotation(
            args, src_root, dst_root, dst_features, grid=grid, basis=basis,
        )
        rotations_used = ", ".join(
            r.strip() for r in args.rotations.split(",") if r.strip()
        )
        layout_note = (
            f"Source layout: legacy rotation, rotations included: "
            f"{rotations_used} (legacy multipole files ignored; coefficients "
            f"re-derived in the project's basis)."
        )
    else:
        logger.error("unknown layout %s", args.layout)
        return 2

    logger.info(
        "converted %d sample pairs (skipped %d, l_max=%d)",
        converted, len(skipped), args.l_max,
    )
    if skipped:
        for sid, msg in skipped[:10]:
            logger.warning("  skipped %s: %s", sid, msg)
        if len(skipped) > 10:
            logger.warning("  ... %d more skipped", len(skipped) - 10)

    p_rels = np.array([d["p_rel_rmse"] for d in diags]) if diags else np.array([])
    e_rels = np.array([d["e_rel_rmse"] for d in diags]) if diags else np.array([])
    if p_rels.size:
        logger.info(
            "L=%d projection residual on real corpus: "
            "P_rel mean=%.3f median=%.3f max=%.3f; "
            "E_rel mean=%.3f median=%.3f max=%.3f",
            args.l_max,
            float(p_rels.mean()), float(np.median(p_rels)), float(p_rels.max()),
            float(e_rels.mean()), float(np.median(e_rels)), float(e_rels.max()),
        )

    if not args.dry_run:
        readme = dst_root / "README.md"
        residual_block = ""
        if p_rels.size:
            residual_block = (
                f"\n## Projection residual at L={args.l_max}\n\n"
                f"- ``||P_meas - P_resyn||_2 / ||P_meas||_2``: "
                f"mean={float(p_rels.mean()):.4f}, "
                f"median={float(np.median(p_rels)):.4f}, "
                f"max={float(p_rels.max()):.4f}.\n"
                f"- ``||E - E_resyn||_2 / ||E||_2``: "
                f"mean={float(e_rels.mean()):.4f}, "
                f"median={float(np.median(e_rels)):.4f}, "
                f"max={float(e_rels.max()):.4f}.\n\n"
                f"Above L={args.l_max}, the high-`l` content of the original "
                f"measurement is discarded by construction.\n"
            )
        readme.write_text(
            "# Real-antenna corpus (imported)\n\n"
            f"Imported on conversion from ``{src_root}``.\n\n"
            f"- Layout: ``{args.layout}``\n"
            f"- {layout_note}\n"
            f"- Total sample pairs: {converted}\n"
            f"- Bandlimit (project basis): L = {args.l_max}\n"
            f"- Target format: `E_in_plane/<sample_id>.txt` (7-column far-field, "
            f"P_resyn = |E_resyn|^2) paired with `Results_<sample_id>.txt` "
            f"(project-convention coefficients).\n"
            f"- Grid: n_theta={args.n_theta}, n_phi={args.n_phi}.\n"
            f"- Generator: `scripts/import_legacy_real_antenna.py "
            f"--layout {args.layout} --l-max {args.l_max}`.\n"
            f"{residual_block}"
        )
        logger.info("wrote %s", readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
