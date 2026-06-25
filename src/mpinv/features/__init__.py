"""Feature extractors and feature pipelines."""

from mpinv.features.composite import CompositeFeaturesConfig, CompositePipeline
from mpinv.features.fft_radial import FFTRadial, FFTRadialConfig
from mpinv.features.hog import HOGConfig, HOGExtractor
from mpinv.features.modes import InputMode, select_channels
from mpinv.features.normalisers import (
    Normaliser,
    PassthroughScaler,
    StandardScaler,
    build_normaliser,
)
from mpinv.features.pca import IncrementalPCAStream, RandomizedPCA
from mpinv.features.power_pipeline import PowerPCAPipeline, PowerPCAPipelineConfig
from mpinv.features.raw_flat import RawFlattenPipeline, RawFlattenPipelineConfig
from mpinv.features.registry import FEATURE_EXTRACTORS, register_feature
from mpinv.features.sh_power import SHPower, SHPowerConfig
from mpinv.features.subsample import SubsampleGridPipeline, SubsampleGridPipelineConfig

__all__ = [
    "FEATURE_EXTRACTORS",
    "CompositeFeaturesConfig",
    "CompositePipeline",
    "FFTRadial",
    "FFTRadialConfig",
    "HOGConfig",
    "HOGExtractor",
    "IncrementalPCAStream",
    "InputMode",
    "Normaliser",
    "PassthroughScaler",
    "PowerPCAPipeline",
    "PowerPCAPipelineConfig",
    "RandomizedPCA",
    "RawFlattenPipeline",
    "RawFlattenPipelineConfig",
    "SHPower",
    "SHPowerConfig",
    "StandardScaler",
    "SubsampleGridPipeline",
    "SubsampleGridPipelineConfig",
    "build_normaliser",
    "register_feature",
    "select_channels",
]
