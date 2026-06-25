"""Core primitives: tensor-shape contracts, packing, grid, area weights, seeds, types."""

from mpinv.core.area_weights import area_weights, normalised_area_weights
from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import (
    K_MODES,
    L_MAX,
    PACKED_DIM,
    flat_index,
    iter_modes,
    pack_coefficients,
    pack_to_sht_grid,
    unpack_coefficients,
    unpack_from_sht_grid,
)
from mpinv.core.seeds import RNGContext, set_global_seed
from mpinv.core.shapes import (
    assert_field_complex,
    assert_packed_coeffs,
    assert_power_pattern,
)

__all__ = [
    "GRID_DEFAULT",
    "K_MODES",
    "L_MAX",
    "PACKED_DIM",
    "GridSpec",
    "RNGContext",
    "area_weights",
    "assert_field_complex",
    "assert_packed_coeffs",
    "assert_power_pattern",
    "flat_index",
    "iter_modes",
    "normalised_area_weights",
    "pack_coefficients",
    "pack_to_sht_grid",
    "set_global_seed",
    "unpack_coefficients",
    "unpack_from_sht_grid",
]
