"""Vector spherical-harmonic basis precomputation and caching.

The VSH basis tensor :math:`\\Psi^X_{lm}(\\theta, \\varphi)` is a fixed function of the
angular grid and the truncation order. We compute it once with NumPy/SciPy and persist
it to disk as a single ``.npz`` file, then load it as a torch tensor for use in both
the synthetic data generator and the differentiable forward operator.

Convention follows Jackson, *Classical Electrodynamics*, 3rd ed., Chapter 9
(presentation/ch1_full.md §1.1):

- Scalar SH ``Y_l^m`` with the Condon-Shortley phase, normalised so the spherical inner
  product yields :math:`\\delta_{ll'}\\delta_{mm'}`. We use ``scipy.special.sph_harm_y``
  (or ``sph_harm`` on older SciPy) which already includes Condon-Shortley.
- Magnetic family (TE) :math:`\\boldsymbol\\Psi^M_{lm} =
  \\frac{1}{\\sqrt{l(l+1)}}\\, \\mathbf L Y_l^m` where
  :math:`\\mathbf L = -i\\,\\mathbf r\\times\\nabla` is the angular-momentum operator
  on the unit sphere. In tangential ``(theta, phi)`` components this gives
  :math:`(\\Psi^M)_\\theta = \\frac{1}{\\sqrt{l(l+1)}\\,\\sin\\theta}\\,\\partial_\\varphi Y_l^m`,
  :math:`(\\Psi^M)_\\varphi = -\\frac{1}{\\sqrt{l(l+1)}}\\,\\partial_\\theta Y_l^m`.
- Electric family (TM) :math:`\\boldsymbol\\Psi^E_{lm} = \\hat{\\mathbf r} \\times
  \\boldsymbol\\Psi^M_{lm}`, i.e. a 90° rotation in the tangent plane:
  :math:`(\\Psi^E)_\\theta = -(\\Psi^M)_\\varphi`,
  :math:`(\\Psi^E)_\\varphi = (\\Psi^M)_\\theta`.

Output tensor layout (NumPy):

    basis : (n_modes_K, n_families=2, n_components=2, n_theta, n_phi) complex64

with ``family`` in (E, M) and ``component`` in (theta, phi). The mode index runs over
the canonical ``iter_modes(L_MAX)`` order.

This is the *single source of truth* for the forward operator. Any inconsistency
between this basis and torch-harmonics is detected in
:mod:`tests.unit.test_differentiable_field` (Phase C).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX, iter_modes


def _scalar_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Scalar spherical harmonic ``Y_l^m(theta, phi)``.

    Uses ``scipy.special.sph_harm_y`` (SciPy >= 1.15) when available, else falls back
    to the older ``scipy.special.sph_harm`` (deprecated argument order swap).
    """
    try:
        from scipy.special import sph_harm_y  # type: ignore[attr-defined]

        return sph_harm_y(int(l), int(m), theta, phi)
    except ImportError:
        from scipy.special import sph_harm  # type: ignore[attr-defined]

        return sph_harm(int(m), int(l), phi, theta)


def _dtheta_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Polar derivative ``∂_theta Y_l^m`` via the raising-operator identity::

        ∂_theta Y_l^m = m cot(theta) Y_l^m
                      + sqrt((l - m)(l + m + 1)) e^{-i phi} Y_{l, m+1}

    (Jackson 3.69 / standard QM raising operator on S^2.) For ``m = l`` the second
    term is zero because ``Y_{l, l+1} = 0``. For ``m = -l`` the second term is
    non-zero (m+1 = -l+1 is still a valid order).

    Returns zero arrays for ``l = 0`` and ``|m| > l``.
    """
    if l == 0:
        return np.zeros_like(theta, dtype=np.complex128)
    if abs(m) > l:
        return np.zeros_like(theta, dtype=np.complex128)
    cot_theta = np.cos(theta) / np.sin(theta)
    term1 = m * cot_theta * _scalar_sph_harm(l, m, theta, phi)
    # Y_{l, m+1} vanishes only when m + 1 > l (i.e. m == l).
    if m + 1 > l:
        return term1
    coef = np.sqrt((l - m) * (l + m + 1))
    term2 = coef * np.exp(-1j * phi) * _scalar_sph_harm(l, m + 1, theta, phi)
    return term1 + term2


@dataclass(frozen=True, slots=True)
class VSHBasis:
    """Precomputed VSH basis tensor and its provenance metadata.

    Attributes
    ----------
    basis : np.ndarray
        Complex array of shape ``(K, 2, 2, n_theta, n_phi)`` with the (mode, family,
        component, theta, phi) ordering described in the module docstring.
    grid : GridSpec
        The grid this basis was computed on.
    l_max : int
        Truncation order. ``K = l_max (l_max + 2)``.
    digest : str
        Short hex digest of the parameters; used for cache invalidation.
    """

    basis: np.ndarray
    grid: GridSpec
    l_max: int
    digest: str

    @property
    def n_modes(self) -> int:
        return self.basis.shape[0]

    def to_torch(
        self,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Convert to a contiguous torch tensor of shape ``(K, 2, 2, n_theta, n_phi)``."""
        return torch.as_tensor(np.ascontiguousarray(self.basis), dtype=dtype, device=device)


def _digest(grid: GridSpec, l_max: int) -> str:
    h = hashlib.sha256()
    h.update(repr((grid, l_max)).encode())
    return h.hexdigest()[:16]


def build_basis(grid: GridSpec = GRID_DEFAULT, l_max: int = L_MAX) -> VSHBasis:
    """Compute the VSH basis tensor on ``grid``.

    Memory: at L = 15 on the 360x179 grid this allocates roughly 250 MB (255 modes x
    2 families x 2 components x 64,440 pixels x 8 bytes per complex64).
    """
    theta_1d = grid.theta_axis()
    phi_1d = grid.phi_axis()
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")  # (n_theta, n_phi)
    sin_theta = np.sin(theta)

    K = l_max * (l_max + 2)
    out = np.zeros((K, 2, 2, grid.n_theta, grid.n_phi), dtype=np.complex64)

    for k, (l, m) in enumerate(iter_modes(l_max)):
        Y = _scalar_sph_harm(l, m, theta, phi)
        dY_dth = _dtheta_sph_harm(l, m, theta, phi)
        dY_dph = 1j * m * Y  # ∂_phi Y_l^m = i m Y_l^m
        norm = 1.0 / np.sqrt(l * (l + 1))

        # Magnetic (TE / toroidal): Psi^M = (1/sqrt(l(l+1))) L Y, with L = -i r x ∇.
        # In components on S^2:
        #   (Psi^M)_theta =  (1/(sqrt(l(l+1)) sin theta)) ∂_phi Y
        #   (Psi^M)_phi   = -(1/sqrt(l(l+1))) ∂_theta Y
        psi_m_theta = norm * dY_dph / sin_theta
        psi_m_phi = -norm * dY_dth

        # Electric (TM): Psi^E = r_hat x Psi^M, i.e. 90° rotation in the tangent plane:
        #   (Psi^E)_theta = -(Psi^M)_phi
        #   (Psi^E)_phi   = +(Psi^M)_theta
        psi_e_theta = -psi_m_phi
        psi_e_phi = psi_m_theta

        out[k, 0, 0] = psi_e_theta
        out[k, 0, 1] = psi_e_phi
        out[k, 1, 0] = psi_m_theta
        out[k, 1, 1] = psi_m_phi

    return VSHBasis(basis=out, grid=grid, l_max=l_max, digest=_digest(grid, l_max))


def _cache_path(cache_dir: Path, grid: GridSpec, l_max: int) -> Path:
    return cache_dir / f"vsh_basis_L{l_max}_n{grid.n_theta}x{grid.n_phi}_{_digest(grid, l_max)}.npz"


def load_basis(
    grid: GridSpec = GRID_DEFAULT,
    l_max: int = L_MAX,
    cache_dir: Path | str = "data/cache",
) -> VSHBasis:
    """Load the VSH basis from disk, computing and caching it on first use."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, grid, l_max)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            basis = f["basis"]
        return VSHBasis(basis=basis, grid=grid, l_max=l_max, digest=_digest(grid, l_max))
    obj = build_basis(grid, l_max)
    np.savez(path, basis=obj.basis)
    return obj
