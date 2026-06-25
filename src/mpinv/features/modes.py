"""Input mode selection: which channels of the angular pattern feed the model."""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class InputMode(StrEnum):
    """Which channels of the angular pattern to feed the model.

    Members
    -------
    POWER : a single real channel ``P = |E_theta|^2 + |E_phi|^2``.
    MAGNITUDE : two real channels ``(|E_theta|, |E_phi|)``; phase is discarded.
    COMPLEX : four real channels ``(Re E_theta, Im E_theta, Re E_phi, Im E_phi)``.
    """

    POWER = "power"
    MAGNITUDE = "magnitude"
    COMPLEX = "complex"


def select_channels(E: np.ndarray | None, P: np.ndarray | None, mode: InputMode) -> np.ndarray:
    """Project the data into the channel layout requested by ``mode``.

    Parameters
    ----------
    E : np.ndarray or None
        Complex field of shape ``(B, 2, n_theta, n_phi)``. Required for
        ``MAGNITUDE`` and ``COMPLEX``; may be ``None`` for ``POWER``.
    P : np.ndarray or None
        Power pattern of shape ``(B, n_theta, n_phi)``. Required for ``POWER``;
        may be ``None`` for the other modes.
    mode : InputMode

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(B, C, n_theta, n_phi)`` where ``C`` is 1, 2, or 4.
    """
    mode = InputMode(mode)
    if mode is InputMode.POWER:
        if P is None:
            if E is None:
                raise ValueError("POWER mode needs P or E")
            P = (E.real**2 + E.imag**2).sum(axis=1)
        return P[:, None].astype(np.float32, copy=False)
    if E is None:
        raise ValueError(f"mode {mode} needs the complex field E")
    if mode is InputMode.MAGNITUDE:
        m_theta = np.abs(E[:, 0])
        m_phi = np.abs(E[:, 1])
        return np.stack((m_theta, m_phi), axis=1).astype(np.float32, copy=False)
    if mode is InputMode.COMPLEX:
        out = np.empty((E.shape[0], 4, E.shape[2], E.shape[3]), dtype=np.float32)
        out[:, 0] = E[:, 0].real
        out[:, 1] = E[:, 0].imag
        out[:, 2] = E[:, 1].real
        out[:, 3] = E[:, 1].imag
        return out
    raise ValueError(f"unknown input mode: {mode}")
