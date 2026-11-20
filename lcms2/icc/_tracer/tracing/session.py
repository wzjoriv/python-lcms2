"""
Tracing a function on a placeholder, with NumPy temporarily patched.
"""

from __future__ import annotations

import os
import threading
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from .array import TracedArray, _check_float_dtype
from .functions import _stack
from .graph import Node, TraceAbort, per_channel

_TRACE_LOCK = threading.Lock()


_RANDOM_FUNCTIONS = (
    "rand",
    "randn",
    "random",
    "random_sample",
    "ranf",
    "sample",
    "randint",
    "uniform",
    "normal",
    "standard_normal",
    "choice",
    "shuffle",
    "permutation",
    "poisson",
    "binomial",
    "exponential",
    "beta",
    "gamma",
)


_PYTHON_PER_ELEMENT = ("vectorize", "frompyfunc")


def _forbidden_python_loop(name: str) -> Callable[..., Any]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise TraceAbort(f"np.{name} calls a Python function for every element")

    return forbidden


def _forbidden_random(name: str) -> Callable[..., Any]:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise TraceAbort(f"np.random.{name}: random numbers make the transform non-deterministic")

    return forbidden


class _PatchArrayConstructors:
    """Make ``np.asarray``/``np.array`` pass traced values through while tracing."""

    def __enter__(self) -> _PatchArrayConstructors:
        _TRACE_LOCK.acquire()
        self._asarray, self._array = np.asarray, np.array
        orig_asarray, orig_array = self._asarray, self._array

        def asarray(a: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(a, TracedArray):
                _check_float_dtype(kwargs.get("dtype", args[0] if args else None), "np.asarray")
                return a
            return orig_asarray(a, *args, **kwargs)

        def array(a: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(a, TracedArray):
                _check_float_dtype(kwargs.get("dtype", args[0] if args else None), "np.array")
                return a if kwargs.get("copy") is False else TracedArray(a.node)
            if isinstance(a, (list, tuple)) and any(isinstance(v, TracedArray) for v in a):
                return _stack(a)
            return orig_array(a, *args, **kwargs)

        np.asarray, np.array = asarray, array
        self._random = {}
        for name in _RANDOM_FUNCTIONS:
            original = getattr(np.random, name, None)
            if original is not None:
                self._random[name] = original
                setattr(np.random, name, _forbidden_random(name))
        for name in _PYTHON_PER_ELEMENT:
            self._random[f"np.{name}"] = getattr(np, name)
            setattr(np, name, _forbidden_python_loop(name))
        return self

    def __exit__(self, *exc: object) -> bool:
        np.asarray, np.array = self._asarray, self._array
        for name, original in self._random.items():
            if name.startswith("np."):
                setattr(np, name[3:], original)
            else:
                setattr(np.random, name, original)
        _TRACE_LOCK.release()
        return False


def _numpy_frame(error: BaseException) -> str:
    """`` (inside numpy's <function>)`` for the first NumPy function an exception passed through."""
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if f"{os.sep}numpy{os.sep}" in frame.f_code.co_filename:
            return f" (inside numpy's {getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)})"
    return ""


def trace_function(fn: Callable[[Any], Any], n_channels: int) -> tuple[Node, int]:
    """Trace ``fn`` on an `n_channels` placeholder.

    Returns
    -------
    tuple
        ``(root, width)``: the optimized output node and its channel count.

    Raises
    ------
    TraceAbort
        If `fn` uses anything the tracer cannot record.
    """
    x = TracedArray.input(n_channels)
    try:
        with _PatchArrayConstructors():
            y = fn(x)
    except TraceAbort:
        raise
    except Exception as e:  # noqa: BLE001
        raise TraceAbort(f"exception while tracing: {e!r}{_numpy_frame(e)}") from e
    if isinstance(y, (list, tuple)):
        y = _stack(y)
    if not isinstance(y, TracedArray):
        raise TraceAbort("traced function did not return an array expression")
    from ..optimize import optimize_node

    return optimize_node(per_channel(y.node)), y.n_channels
