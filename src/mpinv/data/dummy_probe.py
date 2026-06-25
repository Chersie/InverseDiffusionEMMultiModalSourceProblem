"""Single-mode probe generator for the ``dummy`` evaluation split.

Produces a deterministic batch of ``4 K`` samples, each carrying exactly one
non-zero packed coefficient (one-hot in packed space). Decoding through the
VSH basis yields a power pattern that reflects the field response of that
single mode, so a correct model is expected to recover near-zero predictions
everywhere except at the active packed slot (modulo the §1.7 reflected-conjugate
ambiguity).

The packed layout is fixed in :mod:`mpinv.core.packing`:

    [Re(a^E) | Im(a^E) | Re(a^M) | Im(a^M)]   length 4 K = 4 L (L + 2)

So the 4K = 140 samples for L = 5 split four ways: 35 "Re a^E" probes, 35
"Im a^E", 35 "Re a^M", 35 "Im a^M". Each subset cycles through the canonical
``l = 1..L``, ``m = -l..+l`` order. ``active_indices = list(range(4 K))``.

This is the data-side complement of
:func:`mpinv.analysis.plots.dummy_probe.build_dummy_probe_figure`, which
expects exactly this ``(pred_packed, active_indices)`` shape.
"""

from __future__ import annotations

import numpy as np

from mpinv.core.packing import unpack_coefficients
from mpinv.data._basis_cache import VSHBasis


def build_single_mode_probe(
    basis: VSHBasis,
    l_max: int,
    *,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build a one-hot probe batch in packed space and decode it through ``basis``.

    Parameters
    ----------
    basis : VSHBasis
        Precomputed VSH basis tensor of shape ``(K, 2, 2, n_theta, n_phi)``.
        Must be on the same grid + ``l_max`` as the rest of the pipeline.
    l_max : int
        Truncation order. The probe has ``4 K = 4 l_max (l_max + 2)`` samples.
    amplitude : float
        Magnitude assigned to the single active packed slot in each sample.
        Default ``1.0`` (unit amplitude in packed space). When the project's
        real-augmented pipeline uses ``scale_factor=1e6`` the packed values
        of real samples land near O(1), so unit amplitude is in-distribution.

    Returns
    -------
    P_dummy : np.ndarray
        Real array of shape ``(4 K, n_theta, n_phi)``; row ``k`` is the field
        produced by setting only ``packed_dummy[k, k] = amplitude``.
    packed_dummy : np.ndarray
        Real array of shape ``(4 K, 4 K)``; ``amplitude * I`` truncated to
        ``float32``.
    active_indices : list[int]
        ``list(range(4 K))`` — the active slot for each row, suitable for
        :func:`mpinv.analysis.plots.dummy_probe.build_dummy_probe_figure`.
    """
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1; got {l_max}")
    K = l_max * (l_max + 2)
    if basis.l_max != l_max:
        raise ValueError(
            f"basis.l_max={basis.l_max} disagrees with requested l_max={l_max}"
        )
    if basis.n_modes != K:
        raise ValueError(
            f"basis.n_modes={basis.n_modes} disagrees with K={K} for l_max={l_max}"
        )

    n_samples = 4 * K
    packed_dummy = (np.eye(n_samples, dtype=np.float32) * float(amplitude))

    # Decode through the same einsum pipeline as truncate_and_resynthesise so the
    # P_dummy exactly matches what the differentiable VSH decoder will produce
    # at evaluation time.
    a_e, a_m = unpack_coefficients(packed_dummy)
    E_e = np.einsum("nk,kctp->nctp", a_e, basis.basis[:, 0])
    E_m = np.einsum("nk,kctp->nctp", a_m, basis.basis[:, 1])
    E = E_e + E_m
    P_dummy = (E.real**2 + E.imag**2).sum(axis=1).astype(np.float32)

    active_indices = list(range(n_samples))
    return P_dummy, packed_dummy, active_indices


__all__ = ["build_single_mode_probe"]
