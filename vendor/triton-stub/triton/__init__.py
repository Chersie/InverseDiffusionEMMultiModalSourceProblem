"""No-op triton stub.

This package satisfies ``import triton`` and the ``@triton.jit`` decorator on
platforms without CUDA. Functions decorated with ``@triton.jit`` are not callable;
they merely return a sentinel that raises if invoked. The torch-harmonics 0.6.5
``_disco_convolution`` module imports triton at load time, but the kernels are
only reachable through ``DiscreteContinuousConv*`` classes that this framework
does not use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import language

__version__ = "0.0.0+stub"


class _StubKernel:
    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "triton kernel called via stub; this build of torch-harmonics has no GPU "
            "implementation on macOS arm64. Use the SHT API only."
        )

    def __getitem__(self, _grid: Any) -> "_StubKernel":
        return self


def jit(fn: Callable[..., Any] | None = None, **_kwargs: Any) -> Any:
    """No-op replacement for ``triton.jit``."""
    if fn is None:
        return jit
    return _StubKernel(fn)


def autotune(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    def _wrap(fn: Callable[..., Any]) -> Any:
        return _StubKernel(fn)

    return _wrap


def heuristics(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    def _wrap(fn: Callable[..., Any]) -> Any:
        return _StubKernel(fn)

    return _wrap


class Config:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs


__all__ = ["Config", "__version__", "autotune", "heuristics", "jit", "language"]
