"""Library version of the real-augmented data pipeline used by S5/staged.

This module is the single source of truth for the *load → split → augment*
flow that ``scripts/run_real_augmented.py`` and ``scripts/run_staged_real_augmented.py``
have historically implemented inline. The script entry points re-export the
helpers below for backwards compatibility; the Hydra builder
:func:`build_real_augmented_pipeline` lets ``mpinv-train`` consume the same
pipeline through ``configs/data/real_augmented_l5.yaml``.

Pipeline (mirrors the canonical S5 protocol described in
``research/baseline-experiments/manifest.md`` R7):

1. List paired real-antenna files and split sample ids deterministically into
   train / val / holdout (all three sample-id-disjoint; no augmented copy of a
   train source ever appears in val or holdout).
2. Load the train + val sources, truncate-and-resynthesise their power patterns
   from the L=l_max packed coefficients so the (P, packed) pair lives exactly
   on the bandlimit-l_max manifold.
3. Optionally rescale P and packed coefficients by ``scale_factor`` and
   ``sqrt(scale_factor)`` respectively (lifts O(1e-6) real-antenna magnitudes
   into a numerically friendlier regime; preserves the |E|^2 contract).
4. Augment the train sources to ``n_augmented`` total samples by composing
   ``field_phi_roll -> coef_mode_dropout -> field_additive_noise``; the
   memory-heavy dropout step is processed in chunks.
5. Optionally generate a synthetic test split (``colored alpha=2``) as a
   distribution reference and a holdout-real split (sample-id disjoint).

The returned dict is API-compatible with :func:`mpinv.cli._builders.build_data_pipeline`
(``grid``, ``l_max``, ``n_train``, ``n_val``, ``P_train``, ``packed_train``,
``P_val``, ``packed_val``, ``batch_size``, ``num_workers``, ``basis``) plus
optional ``P_test``, ``packed_test``, ``P_holdout``, ``packed_holdout``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mpinv.core.grid import GridSpec
from mpinv.core.packing import unpack_coefficients
from mpinv.data._basis_cache import build_basis, load_basis
from mpinv.data.augment import (
    CoefModeDropoutConfig,
    FieldAdditiveNoiseConfig,
    FieldPhiRollConfig,
    apply_augmentation,
)
from mpinv.data.real_antenna_loader import (
    RealAntennaLoaderConfig,
    list_real_antenna_samples,
    load_real_antenna,
)
from mpinv.data.synthetic_generator import (
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)

logger = logging.getLogger(__name__)


def truncate_and_resynthesise(
    P_orig: np.ndarray,
    packed: np.ndarray,
    *,
    basis,
) -> np.ndarray:
    """Re-synthesise P from (already truncated) packed coefficients.

    Returns a fresh ``P`` with the same shape as ``P_orig`` but consistent with
    the L=l_max sub-block of ``packed``. Drops any l > l_max content from the
    original measurement; honest fidelity / runtime trade-off.
    """
    a_e, a_m = unpack_coefficients(packed)
    E_e = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E = E_e + E_m
    P = (E.real**2 + E.imag**2).sum(axis=1).astype(np.float32)
    if P.shape != P_orig.shape:
        raise RuntimeError(
            f"re-synthesis shape mismatch: got {P.shape}, expected {P_orig.shape}"
        )
    return P


def load_real(
    *,
    holdout_root: str,
    feature_subdir: str,
    grid: GridSpec,
    l_max: int,
    shuffle_seed: int,
    n_source: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load up to ``n_source`` real-antenna samples and stack them.

    Returns ``(P, packed, sample_ids)`` of shapes ``(N, n_theta, n_phi)``,
    ``(N, 4 K)``, and a list of length ``N``.
    """
    cfg = RealAntennaLoaderConfig(
        root=holdout_root,
        feature_subdir=feature_subdir,
        grid=grid,
        l_max=l_max,
        shuffle_seed=shuffle_seed,
        max_samples=n_source,
    )
    samples = load_real_antenna(cfg)
    if not samples:
        raise FileNotFoundError(
            f"no paired real-antenna samples found under {holdout_root!r}; "
            f"expected files like {holdout_root}/{feature_subdir}/<id>.txt and "
            f"{holdout_root}/Results_<id>.txt"
        )
    P = np.stack([s.P for s in samples], axis=0)
    packed = np.stack([s.packed for s in samples], axis=0)
    sids = [s.sample_id for s in samples]
    logger.info("loaded %d real-antenna samples (l_max=%d)", len(samples), l_max)
    return P, packed, sids


def load_smoke(
    *,
    grid: GridSpec,
    l_max: int,
    basis,
    n_source: int,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stand-in for the real-antenna corpus; substitutes synthetic colored alpha=2 samples.

    Used when ``data/raw/real_antenna`` is empty or unavailable. Code-path
    check only — results from smoke mode are not meaningful.
    """
    cfg = SyntheticGeneratorConfig(grid=grid, l_max=l_max, mode="colored", color_alpha=2.0)
    gen = SyntheticGenerator(cfg=cfg, basis=basis)
    rng = np.random.default_rng(shuffle_seed)
    P, packed = gen.generate_batch(n_source, rng)
    sids = [f"smoke_{i:03d}" for i in range(n_source)]
    logger.warning(
        "SMOKE TEST: substituted %d synthetic colored alpha=2 samples for the holdout",
        n_source,
    )
    return P, packed, sids


def build_augmented(
    P_src: np.ndarray,
    packed_src: np.ndarray,
    *,
    n_augmented: int,
    dropout_prob: float,
    field_sigma: float,
    l_max: int,
    basis,
    rng: np.random.Generator,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ``field_phi_roll -> coef_mode_dropout -> field_additive_noise`` to amplify
    the source pool to ``n_augmented`` augmented samples.

    The coefficient-space dropout step is processed in chunks to cap peak memory
    (~``chunk_size`` MB per family on the canonical 360 x 179 grid).
    """
    n_src = P_src.shape[0]
    if n_src == 0:
        raise ValueError("empty source corpus")
    src_idx = rng.integers(0, n_src, size=n_augmented)
    P_aug = P_src[src_idx].copy()
    pk_aug = packed_src[src_idx].copy()

    P_aug, pk_aug = apply_augmentation(
        P_aug, pk_aug, cfg=FieldPhiRollConfig(), rng=rng, l_max=l_max
    )

    if dropout_prob > 0:
        P_chunks: list[np.ndarray] = []
        pk_chunks: list[np.ndarray] = []
        for start in range(0, n_augmented, chunk_size):
            stop = min(start + chunk_size, n_augmented)
            Pc, pkc = apply_augmentation(
                P_aug[start:stop],
                pk_aug[start:stop],
                cfg=CoefModeDropoutConfig(dropout_prob=dropout_prob),
                rng=rng,
                basis=basis,
            )
            P_chunks.append(Pc)
            pk_chunks.append(pkc)
        P_aug = np.concatenate(P_chunks, axis=0)
        pk_aug = np.concatenate(pk_chunks, axis=0)

    P_aug, pk_aug = apply_augmentation(
        P_aug,
        pk_aug,
        cfg=FieldAdditiveNoiseConfig(relative_sigma=field_sigma),
        rng=rng,
    )
    return P_aug, pk_aug


def peek_split_ids(
    *,
    holdout_root: str,
    feature_subdir: str,
    grid: GridSpec,
    l_max: int,
    shuffle_seed: int,
    n_source: int,
    n_train_sources: int,
    n_holdout_samples: int,
    holdout_shuffle_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(train_sids, val_sids, holdout_sids)`` from the loader's
    deterministic shuffle *without* loading any field data.

    Splits are sample-id disjoint: train+val come from the first ``n_source``
    files (after the main shuffle), holdout from the remaining pool re-shuffled
    by ``holdout_shuffle_seed`` and capped at ``n_holdout_samples``.
    """
    cfg = RealAntennaLoaderConfig(
        root=holdout_root,
        feature_subdir=feature_subdir,
        grid=grid,
        l_max=l_max,
        shuffle_seed=shuffle_seed,
        max_samples=None,
    )
    pairs = list_real_antenna_samples(cfg)
    sids = [sid for _, _, sid in pairs]
    train_val = sids[:n_source]
    train_sids = train_val[:n_train_sources]
    val_sids = train_val[n_train_sources:]
    holdout_sids: list[str] = []
    if n_holdout_samples > 0:
        pool = sids[n_source:]
        if pool:
            rng = np.random.default_rng(holdout_shuffle_seed)
            order = rng.permutation(len(pool))
            holdout_sids = [pool[int(i)] for i in order[:n_holdout_samples]]
    return train_sids, val_sids, holdout_sids


def _load_subset_by_ids(
    *,
    holdout_root: str,
    feature_subdir: str,
    grid: GridSpec,
    l_max: int,
    sids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load only the listed sample ids. Order matches the input list."""
    if not sids:
        n_theta = grid.n_theta
        n_phi = grid.n_phi
        K = l_max * (l_max + 2)
        return (
            np.empty((0, n_theta, n_phi), dtype=np.float32),
            np.empty((0, 4 * K), dtype=np.float32),
        )
    cfg = RealAntennaLoaderConfig(
        root=holdout_root,
        feature_subdir=feature_subdir,
        grid=grid,
        l_max=l_max,
        shuffle_seed=0,
        max_samples=None,
    )
    samples = load_real_antenna(cfg)
    by_id = {s.sample_id: s for s in samples}
    chosen = [by_id[sid] for sid in sids if sid in by_id]
    P = np.stack([s.P for s in chosen], axis=0)
    packed = np.stack([s.packed for s in chosen], axis=0)
    return P, packed


def build_real_augmented_pipeline(
    *,
    grid: dict,
    l_max: int,
    holdout_root: str = "data/raw/real_antenna",
    feature_subdir: str = "E_in_plane",
    n_source: int = 200,
    n_train_sources: int = 180,
    n_augmented: int = 10000,
    n_holdout_samples: int = 100,
    shuffle_seed: int = 42,
    holdout_shuffle_seed: int = 314159,
    aug_seed: int = 4242,
    dropout_prob: float = 0.1,
    field_sigma: float = 1e-8,
    scale_factor: float = 1.0,
    aug_chunk_size: int = 500,
    include_synthetic_test: bool = False,
    n_synthetic_test: int = 512,
    seed_synthetic_test: int = 9012,
    include_dummy_probe: bool = True,
    dummy_amplitude: float = 1.0,
    smoke_test: bool = False,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Hydra-friendly builder for the real-augmented L=l_max pipeline.

    Returns a dict matching :func:`mpinv.cli._builders.build_data_pipeline`'s
    layout, plus optional ``P_test``, ``packed_test``, ``P_holdout``,
    ``packed_holdout`` keys when those splits are populated. When
    ``include_dummy_probe=True`` the dict also carries ``P_dummy`` (4 K
    decoded probe fields), ``packed_dummy`` (4 K x 4 K identity scaled by
    ``dummy_amplitude``) and ``dummy_active_indices`` for the
    single-non-zero-coefficient evaluation split.

    The returned arrays are numpy on CPU; the trainer moves them to the
    configured device.
    """
    grid_obj = GridSpec(
        n_phi=grid["n_phi"],
        n_theta=grid["n_theta"],
        theta_start_deg=grid.get("theta_start_deg", 1.0),
        theta_end_deg=grid.get("theta_end_deg", 179.0),
    )
    try:
        basis = load_basis(grid_obj, l_max)
    except Exception:
        basis = build_basis(grid_obj, l_max)

    # Decide split ids before loading any field data so the loader call below
    # only touches the sources we actually need.
    if smoke_test:
        train_sids: list[str] = []
        val_sids: list[str] = []
        holdout_sids: list[str] = []
    else:
        train_sids, val_sids, holdout_sids = peek_split_ids(
            holdout_root=holdout_root,
            feature_subdir=feature_subdir,
            grid=grid_obj,
            l_max=l_max,
            shuffle_seed=shuffle_seed,
            n_source=n_source,
            n_train_sources=n_train_sources,
            n_holdout_samples=n_holdout_samples,
            holdout_shuffle_seed=holdout_shuffle_seed,
        )

    if smoke_test:
        P_all, packed_all, all_sids = load_smoke(
            grid=grid_obj,
            l_max=l_max,
            basis=basis,
            n_source=n_source,
            shuffle_seed=shuffle_seed,
        )
    else:
        P_all, packed_all, all_sids = load_real(
            holdout_root=holdout_root,
            feature_subdir=feature_subdir,
            grid=grid_obj,
            l_max=l_max,
            shuffle_seed=shuffle_seed,
            n_source=n_source,
        )

    P_all = truncate_and_resynthesise(P_all, packed_all, basis=basis)

    if scale_factor != 1.0:
        P_all = (P_all * scale_factor).astype(np.float32)
        packed_all = (packed_all * np.sqrt(scale_factor)).astype(np.float32)

    if smoke_test:
        # In smoke mode, all samples are synthetic; deterministic split from
        # the shuffled order produced by load_smoke.
        n_train_eff = min(n_train_sources, P_all.shape[0])
        P_train_src = P_all[:n_train_eff]
        packed_train_src = packed_all[:n_train_eff]
        P_val = P_all[n_train_eff:].copy()
        packed_val = packed_all[n_train_eff:].copy()
        P_holdout = np.empty((0, *P_all.shape[1:]), dtype=np.float32)
        packed_holdout = np.empty((0, packed_all.shape[1]), dtype=np.float32)
    else:
        idx_by_sid = {sid: i for i, sid in enumerate(all_sids)}
        train_idx = [idx_by_sid[sid] for sid in train_sids if sid in idx_by_sid]
        val_idx = [idx_by_sid[sid] for sid in val_sids if sid in idx_by_sid]
        P_train_src = P_all[train_idx]
        packed_train_src = packed_all[train_idx]
        P_val = P_all[val_idx].copy()
        packed_val = packed_all[val_idx].copy()
        if holdout_sids:
            P_holdout, packed_holdout = _load_subset_by_ids(
                holdout_root=holdout_root,
                feature_subdir=feature_subdir,
                grid=grid_obj,
                l_max=l_max,
                sids=holdout_sids,
            )
            P_holdout = truncate_and_resynthesise(P_holdout, packed_holdout, basis=basis)
            if scale_factor != 1.0:
                P_holdout = (P_holdout * scale_factor).astype(np.float32)
                packed_holdout = (packed_holdout * np.sqrt(scale_factor)).astype(np.float32)
        else:
            P_holdout = np.empty((0, *P_all.shape[1:]), dtype=np.float32)
            packed_holdout = np.empty((0, packed_all.shape[1]), dtype=np.float32)

    rng_aug = np.random.default_rng(aug_seed)
    P_train, packed_train = build_augmented(
        P_train_src,
        packed_train_src,
        n_augmented=n_augmented,
        dropout_prob=dropout_prob,
        field_sigma=field_sigma,
        l_max=l_max,
        basis=basis,
        rng=rng_aug,
        chunk_size=aug_chunk_size,
    )

    out: dict[str, Any] = {
        "grid": grid_obj,
        "l_max": l_max,
        "n_train": int(P_train.shape[0]),
        "n_val": int(P_val.shape[0]),
        "P_train": P_train,
        "packed_train": packed_train,
        "P_val": P_val,
        "packed_val": packed_val,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "basis": basis,
        "augmentation": None,
    }

    if int(P_holdout.shape[0]) > 0:
        out["P_holdout"] = P_holdout
        out["packed_holdout"] = packed_holdout
        out["n_holdout"] = int(P_holdout.shape[0])

    if include_synthetic_test:
        cfg_t = SyntheticGeneratorConfig(
            grid=grid_obj, l_max=l_max, mode="colored", color_alpha=2.0
        )
        gen = SyntheticGenerator(cfg=cfg_t, basis=basis)
        rng = np.random.default_rng(seed_synthetic_test)
        P_t, packed_t = gen.generate_batch(n_synthetic_test, rng)
        if scale_factor != 1.0:
            P_t = (P_t * scale_factor).astype(np.float32)
            packed_t = (packed_t * np.sqrt(scale_factor)).astype(np.float32)
        out["n_test"] = int(P_t.shape[0])
        out["P_test"] = P_t
        out["packed_test"] = packed_t

    if include_dummy_probe:
        from mpinv.data.dummy_probe import build_single_mode_probe

        # Dummy probe: 4 K samples, each a one-hot in packed space. Amplitude is
        # in *post-scale* packed units, matching the rest of this pipeline (so
        # the dummy split lives on the same scale as train/val/holdout when
        # scale_factor != 1.0). No additional rescaling is applied below.
        P_dum, packed_dum, active = build_single_mode_probe(
            basis, l_max, amplitude=dummy_amplitude
        )
        out["n_dummy"] = int(P_dum.shape[0])
        out["P_dummy"] = P_dum
        out["packed_dummy"] = packed_dum
        out["dummy_active_indices"] = active

    return out


__all__ = [
    "build_augmented",
    "build_real_augmented_pipeline",
    "load_real",
    "load_smoke",
    "peek_split_ids",
    "truncate_and_resynthesise",
]
