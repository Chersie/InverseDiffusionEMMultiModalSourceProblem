"""No-op ``triton.language`` stub. Provides only the names referenced as type
annotations at decorator/module-load time inside torch-harmonics 0.6.5's
``_disco_convolution`` (``constexpr``, ``program_id``, ``arange``, ``load``,
``store``, ``cdiv``, ``where``, ``minimum``, ``maximum``, etc.).
"""

from __future__ import annotations

from typing import Any


class _Anything:
    """Sentinel class used in place of real triton types in annotations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, _name: str) -> "_Anything":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> "_Anything":
        return self


class dtype(_Anything):  # noqa: N801
    """Stub for ``triton.language.dtype`` — torch._dynamo references it at import time."""


constexpr = _Anything
tensor = _Anything
pointer_type = _Anything
block_type = _Anything
function_type = _Anything
void = _Anything()


def _stub(*_args: Any, **_kwargs: Any) -> _Anything:
    return _Anything()


program_id = _stub
arange = _stub
load = _stub
store = _stub
cdiv = _stub
where = _stub
minimum = _stub
maximum = _stub
zeros = _stub
zeros_like = _stub
full = _stub
exp = _stub
log = _stub
sqrt = _stub
abs = _stub  # noqa: A001
sum = _stub  # noqa: A001
max = _stub  # noqa: A001
min = _stub  # noqa: A001
sin = _stub
cos = _stub
sigmoid = _stub
softmax = _stub
multiple_of = _stub
debug_barrier = _stub
atomic_add = _stub

float32 = _Anything()
float64 = _Anything()
int32 = _Anything()
int64 = _Anything()
uint32 = _Anything()
uint64 = _Anything()
bfloat16 = _Anything()
float16 = _Anything()
