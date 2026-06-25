"""Data layer: synthetic generation, real-antenna loader, memmap dataset, augmentation."""

from mpinv.data._basis_cache import VSHBasis, build_basis, load_basis
from mpinv.data.augment import (
    AugmentationConfig,
    CoefAdditiveNoiseConfig,
    CoefPhaseRotationConfig,
    FieldAdditiveNoiseConfig,
    FieldPhiRollConfig,
    apply_augmentation,
    build_augmentation,
)
from mpinv.data.memmap_dataset import (
    MemmapDataset,
    MemmapShard,
    list_shards,
    shard_token,
    write_shard,
)
from mpinv.data.real_antenna_loader import (
    RealAntennaLoaderConfig,
    RealAntennaSample,
    iter_real_antenna,
    list_real_antenna_samples,
    load_real_antenna,
)
from mpinv.data.splits import SplitConfig, build_splits
from mpinv.data.synthetic_generator import SyntheticGenerator, SyntheticGeneratorConfig

__all__ = [
    "AugmentationConfig",
    "CoefAdditiveNoiseConfig",
    "CoefPhaseRotationConfig",
    "FieldAdditiveNoiseConfig",
    "FieldPhiRollConfig",
    "MemmapDataset",
    "MemmapShard",
    "RealAntennaLoaderConfig",
    "RealAntennaSample",
    "SplitConfig",
    "SyntheticGenerator",
    "SyntheticGeneratorConfig",
    "VSHBasis",
    "apply_augmentation",
    "build_augmentation",
    "build_basis",
    "build_splits",
    "iter_real_antenna",
    "list_real_antenna_samples",
    "list_shards",
    "load_basis",
    "load_real_antenna",
    "shard_token",
    "write_shard",
]
