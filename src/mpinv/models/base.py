"""Base config for trainable models. Every model exposes ``input_dim``, ``output_dim``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BaseModelConfig:
    """Common knobs for every regressor in the framework.

    Attributes
    ----------
    input_dim : int
        Width of the feature vector at the model's input.
    output_dim : int
        Width of the packed coefficient vector at the model's output.
        For the project default this is ``PACKED_DIM = 4 K = 1020`` (L = 15).
    """

    input_dim: int
    output_dim: int
