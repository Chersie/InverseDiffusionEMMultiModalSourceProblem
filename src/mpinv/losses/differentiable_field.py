"""Differentiable VSH decoder: packed coefficients -> complex field -> power pattern.

Implementation strategy
-----------------------
We do *not* use ``torch_harmonics.InverseRealVectorSHT`` for the production path.
Reasons:

1. The library returns a *real-valued* tangential field (its target is real spatial
   signals like wind), but our problem requires a *complex* field whose modulus
   squared gives the power pattern (presentation/ch1_full.md section 1.4).
2. The legacy framework's adapter on top of ``torch-harmonics`` had documented
   l-band and m-index bugs (see PHYSICS_LAYER_AUDIT_RESULTS.md in the legacy repo).
3. The public-API cost: the inverse vector real SHT pins ``nlat`` to a Clenshaw-
   Curtiss grid that includes the poles; our 1-degree grid excludes them.

Instead we precompute the complex VSH basis tensor on the project grid via
:mod:`mpinv.data._basis_cache`, store it as a torch buffer, and run a single
einsum to get the complex field. This matches the synthetic data generator's
forward operator *exactly*, so a perfect inverse model achieves zero loss.

Memory: the basis tensor is roughly 250 MB at L = 15 on the (179, 360) grid in
``complex64``; this is the dominant memory cost of the layer.

Forward contract
----------------
- input  ``packed`` : tensor of shape ``(B, 4 K)``, real (``float32`` or ``float16`` /
  ``bfloat16`` under autocast). ``4 K = 1020`` for ``L = 15``.
- output ``P``       : tensor of shape ``(B, n_theta, n_phi)``, real, non-negative
  (sum of squared-magnitudes of two complex tangential channels).

Optionally :meth:`field` returns the intermediate complex field of shape
``(B, 2, n_theta, n_phi)`` for diagnostic plots and reference checks.
"""

from __future__ import annotations

from typing import overload

import torch
from torch import nn

from mpinv.core.grid import GRID_DEFAULT, GridSpec
from mpinv.core.packing import L_MAX
from mpinv.core.shapes import assert_packed_coeffs, assert_power_pattern
from mpinv.data._basis_cache import VSHBasis, build_basis, load_basis


class DifferentiableMultipoleField(nn.Module):
    """Synthesise a complex tangential field and its power pattern from packed coefficients.

    The basis is loaded from disk once (or computed on the fly for tests) and held as
    two buffers ``basis_real`` and ``basis_imag`` of shape ``(K, 2 family, 2 component,
    n_theta, n_phi)``. Splitting into real and imaginary parts lets us use real
    einsums and then re-assemble, which composes cleanly with autocast and avoids
    the ``aten::view_as_complex`` quirks that come up in some autograd paths.
    """

    basis_real: torch.Tensor  # (K, 2 family, 2 comp, n_theta, n_phi)
    basis_imag: torch.Tensor

    def __init__(
        self,
        grid: GridSpec = GRID_DEFAULT,
        l_max: int = L_MAX,
        basis: VSHBasis | None = None,
        cache_dir: str = "data/cache",
    ):
        super().__init__()
        self.grid = grid
        self.l_max = l_max
        self.K = l_max * (l_max + 2)
        if basis is None:
            try:
                basis = load_basis(grid, l_max, cache_dir=cache_dir)
            except Exception:
                basis = build_basis(grid, l_max)
        if basis.l_max != l_max or basis.grid != grid:
            raise ValueError(
                f"basis provenance mismatch: basis (l_max={basis.l_max}, grid={basis.grid}) "
                f"!= layer (l_max={l_max}, grid={grid})"
            )
        re = torch.as_tensor(basis.basis.real.copy(), dtype=torch.float32)
        im = torch.as_tensor(basis.basis.imag.copy(), dtype=torch.float32)
        self.register_buffer("basis_real", re, persistent=False)
        self.register_buffer("basis_imag", im, persistent=False)

    def _split_packed(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(re_aE, im_aE, re_aM, im_aM)``, each of shape ``(B, K)``."""
        if packed.shape[-1] != 4 * self.K:
            raise ValueError(
                f"packed trailing dim must be 4 K = {4 * self.K}, got {packed.shape[-1]}"
            )
        re_aE, im_aE, re_aM, im_aM = torch.chunk(packed, 4, dim=-1)
        return re_aE, im_aE, re_aM, im_aM

    @overload
    def forward(self, packed: torch.Tensor, return_field: bool = False) -> torch.Tensor: ...

    @overload
    def forward(
        self, packed: torch.Tensor, return_field: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(  # type: ignore[override]
        self, packed: torch.Tensor, return_field: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the power pattern (and optionally the complex field).

        Parameters
        ----------
        packed : torch.Tensor
            Real packed coefficients of shape ``(B, 4 K)``.
        return_field : bool, default False
            If True, also return the complex field tensor ``(B, 2, n_theta, n_phi)``.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
        """
        assert_packed_coeffs(packed, name="packed")

        re_aE, im_aE, re_aM, im_aM = self._split_packed(packed)
        # basis split per family: (K, 2 comp, n_theta, n_phi) for each of E, M
        bE_re = self.basis_real[:, 0]
        bE_im = self.basis_imag[:, 0]
        bM_re = self.basis_real[:, 1]
        bM_im = self.basis_imag[:, 1]

        # Real and imaginary parts of E_e = sum_k a^E_k Psi^E_k (complex coef * complex basis):
        # (Re a + i Im a)(Re b + i Im b) = (Re a Re b - Im a Im b) + i(Re a Im b + Im a Re b)
        E_e_re = torch.einsum("nk,kctp->nctp", re_aE, bE_re) - torch.einsum(
            "nk,kctp->nctp", im_aE, bE_im
        )
        E_e_im = torch.einsum("nk,kctp->nctp", re_aE, bE_im) + torch.einsum(
            "nk,kctp->nctp", im_aE, bE_re
        )
        E_m_re = torch.einsum("nk,kctp->nctp", re_aM, bM_re) - torch.einsum(
            "nk,kctp->nctp", im_aM, bM_im
        )
        E_m_im = torch.einsum("nk,kctp->nctp", re_aM, bM_im) + torch.einsum(
            "nk,kctp->nctp", im_aM, bM_re
        )

        E_re = E_e_re + E_m_re
        E_im = E_e_im + E_m_im
        # P = |E_theta|^2 + |E_phi|^2 = sum_c (Re E_c)^2 + (Im E_c)^2 over both components
        P = (E_re.pow(2) + E_im.pow(2)).sum(dim=1)

        assert_power_pattern(P, grid=self.grid, name="P")
        if return_field:
            field = torch.complex(E_re, E_im)  # (B, 2 comp, n_theta, n_phi)
            return P, field
        return P
