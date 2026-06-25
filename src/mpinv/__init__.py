"""mpinv: phaseless multipole-coefficient inversion."""

__version__ = "0.1.0"

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import K_MODES, L_MAX, PACKED_DIM, pack_coefficients, unpack_coefficients

__all__ = [
    "GRID_DEFAULT",
    "K_MODES",
    "L_MAX",
    "PACKED_DIM",
    "GridSpec",
    "__version__",
    "pack_coefficients",
    "unpack_coefficients",
]
