"""
`TracedArray`, the placeholder whose operations are recorded as a graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .graph import (
    _SUPPORTED_UFUNCS,
    Node,
    TraceAbort,
    const_node,
    select_node,
    stack_node,
)


def _operand(x: Operand) -> Node:
    if isinstance(x, TracedArray):
        if x.moved:
            raise TraceAbort(
                "a channel-first view (np.moveaxis / .T) can only be unpacked, not used in arithmetic"
            )
        return x.node
    if isinstance(x, Node):
        return x
    if isinstance(x, (bool, int, float, np.number, np.bool_)) or isinstance(x, np.ndarray):
        return const_node(x)
    if isinstance(x, (list, tuple)):
        return const_node(np.asarray(x, dtype=np.float64))
    raise TraceAbort(f"unsupported operand type {type(x).__name__}")


def _is_scalar_const(node: Node) -> bool:
    return node.op == "const" and node.meta["value"].ndim == 0


def elementwise(op: str, *operands: Operand, meta: dict[str, Any] | None = None) -> TracedArray:
    """Apply an elementwise operation with numpy-style channel broadcasting."""
    nodes = [_operand(o) for o in operands]
    widths = [n.width or 1 for n in nodes]
    width = max(widths)
    if any(w not in (1, width) for w in widths):
        raise TraceAbort(f"cannot broadcast channel widths {widths} in {op!r}")
    vectorizable = all(w == width or _is_scalar_const(n) for n, w in zip(nodes, widths, strict=False))
    if width == 1 or vectorizable:
        return TracedArray(Node(op, tuple(nodes), meta, width=width))
    channels = []
    for i in range(width):
        parts = [select_node(n, i) if w == width else n for n, w in zip(nodes, widths, strict=False)]
        channels.append(Node(op, tuple(parts), meta, width=1))
    return TracedArray(stack_node(channels))


def matmul(node: Node, matrix: ArrayLike) -> TracedArray:
    """``node @ matrix`` for a constant ``(width, k)`` matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    if matrix.ndim != 2:
        raise TraceAbort(f"matmul with a constant of shape {matrix.shape}")
    width = node.width or 1
    if matrix.shape[0] != width:
        if width == 1:
            node = stack_node([node])
        else:
            raise TraceAbort(f"matmul of {width} channels with a {matrix.shape} matrix")
    if node.op not in ("stack", "input"):
        node = stack_node([select_node(node, i) for i in range(width)])
    out = TracedArray(Node("matmul", (node,), {"matrix": matrix}, width=matrix.shape[1]))
    return out.channel(0) if squeeze else out


class TracedArray:
    """Placeholder array whose last axis holds symbolic channels."""

    __array_priority__ = 1000

    def __init__(self, node: Node, moved: bool = False):
        self.node = node
        #: True for a channel-first view (``np.moveaxis(x, -1, 0)``, ``x.T``) that may be unpacked
        self.moved = moved

    @property
    def n_channels(self) -> int:
        return self.node.width or 1

    @property
    def shape(self) -> tuple[int]:
        return (self.n_channels,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return self.n_channels

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float64)

    @classmethod
    def input(cls, n_channels: int) -> TracedArray:
        return cls(Node("input", (), {"n_channels": n_channels}, width=n_channels))

    def channel(self, i: int) -> TracedArray:
        n = self.n_channels
        if not -n <= i < n:
            raise TraceAbort(f"channel index {i} out of range for {n} channels")
        return TracedArray(select_node(self.node, i % n))

    def _channels(self, indices: Iterable[int]) -> TracedArray:
        picked = [select_node(self.node, i % self.n_channels) for i in indices]
        if len(picked) == 1:
            return TracedArray(picked[0])
        return TracedArray(stack_node(picked))

    def _key_indices(self, item: Any) -> list[int]:
        key = item[-1] if isinstance(item, tuple) and item else item
        if isinstance(item, tuple):
            rest = item[:-1]
            if any(not (k is Ellipsis or k == slice(None)) for k in rest):
                raise TraceAbort(f"unsupported indexing pattern {item!r}")
        n = self.n_channels
        if key is Ellipsis:
            return list(range(n))
        if isinstance(key, (int, np.integer)):
            if not -n <= int(key) < n:
                raise TraceAbort(f"channel index {key} out of range for {n} channels")
            return [int(key) % n]
        if isinstance(key, slice):
            return list(range(*key.indices(n)))
        if isinstance(key, (list, np.ndarray)) and np.asarray(key).dtype.kind in "iu":
            return [int(k) % n for k in np.asarray(key).ravel()]
        raise TraceAbort(f"unsupported indexing pattern {item!r}")

    def __getitem__(self, item: Any) -> TracedArray:
        if self.moved:
            if isinstance(item, (int, np.integer)):
                return self.channel(int(item))
            raise TraceAbort("a channel-first view can only be indexed by channel number")
        _reject_mask_key(item)
        key = item[-1] if isinstance(item, tuple) and item else item
        if key is None:
            if isinstance(item, tuple) and len(item) > 1 and item[-2] is not Ellipsis:
                return self[item[:-1]][..., None]
            if self.n_channels != 1:
                raise TraceAbort("x[..., None] is only traceable on a 1-channel value")
            return self
        if isinstance(key, (int, np.integer)):
            return self.channel(int(key))
        indices = self._key_indices(item)
        if key is Ellipsis:
            return self
        return self._channels(indices)

    def __setitem__(self, item: Any, value: Operand) -> None:
        _reject_mask_key(item)
        indices = self._key_indices(item)
        value_node = _operand(value)
        width = value_node.width or 1
        if width not in (1, len(indices)):
            raise TraceAbort(f"cannot assign {width} channels to {len(indices)}")
        channels = [select_node(self.node, i) for i in range(self.n_channels)]
        for j, i in enumerate(indices):
            channels[i] = value_node if width == 1 else select_node(value_node, j)
        self.node = stack_node(channels) if len(channels) > 1 else channels[0]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Operand, **kwargs: Any) -> TracedArray:
        if method != "__call__":
            raise TraceAbort(f"unsupported ufunc method {method}")
        if kwargs.get("out") is not None:
            raise TraceAbort("ufunc out= is not traceable")
        if ufunc is np.matmul:
            return _matmul_any(*inputs)
        name = _SUPPORTED_UFUNCS.get(ufunc)
        if name is None:
            raise TraceAbort(f"unsupported ufunc {ufunc.__name__}")
        return elementwise(name, *inputs)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        from .functions import FUNCTIONS

        handler = FUNCTIONS.get(func)
        if handler is None:
            raise TraceAbort(f"unsupported numpy function {getattr(func, '__name__', func)}")
        return handler(*args, **kwargs)

    def __matmul__(self, other: Operand) -> TracedArray:
        return _matmul_any(self, other)

    def __rmatmul__(self, other: Operand) -> TracedArray:
        return _matmul_any(other, self)

    def dot(self, other: Operand) -> TracedArray:
        return _matmul_any(self, other)

    def __add__(self, other: Operand) -> TracedArray:
        return elementwise("add", self, other)

    def __radd__(self, other: Operand) -> TracedArray:
        return elementwise("add", other, self)

    def __sub__(self, other: Operand) -> TracedArray:
        return elementwise("sub", self, other)

    def __rsub__(self, other: Operand) -> TracedArray:
        return elementwise("sub", other, self)

    def __mul__(self, other: Operand) -> TracedArray:
        return _multiply(self, other)

    def __rmul__(self, other: Operand) -> TracedArray:
        return _multiply(other, self)

    def __truediv__(self, other: Operand) -> TracedArray:
        return elementwise("div", self, other)

    def __rtruediv__(self, other: Operand) -> TracedArray:
        return elementwise("div", other, self)

    def __pow__(self, other: Operand) -> TracedArray:
        return elementwise("pow", self, other)

    def __rpow__(self, other: Operand) -> TracedArray:
        return elementwise("pow", other, self)

    def __mod__(self, other: Operand) -> TracedArray:
        return elementwise("mod", self, other)

    def __rmod__(self, other: Operand) -> TracedArray:
        return elementwise("mod", other, self)

    def __floordiv__(self, other: Operand) -> TracedArray:
        return elementwise("floor_div", self, other)

    def __rfloordiv__(self, other: Operand) -> TracedArray:
        return elementwise("floor_div", other, self)

    def __neg__(self) -> TracedArray:
        return elementwise("neg", self)

    def __pos__(self) -> TracedArray:
        return self

    def __abs__(self) -> TracedArray:
        return elementwise("abs", self)

    def __eq__(self, other: Operand) -> TracedArray:
        return elementwise("eq", self, other)

    def __ne__(self, other: Operand) -> TracedArray:
        return elementwise("ne", self, other)

    def __lt__(self, other: Operand) -> TracedArray:
        return elementwise("lt", self, other)

    def __le__(self, other: Operand) -> TracedArray:
        return elementwise("le", self, other)

    def __gt__(self, other: Operand) -> TracedArray:
        return elementwise("gt", self, other)

    def __ge__(self, other: Operand) -> TracedArray:
        return elementwise("ge", self, other)

    def __and__(self, other: Operand) -> TracedArray:
        return elementwise("logical_and", self, other)

    def __rand__(self, other: Operand) -> TracedArray:
        return elementwise("logical_and", other, self)

    def __or__(self, other: Operand) -> TracedArray:
        return elementwise("logical_or", self, other)

    def __ror__(self, other: Operand) -> TracedArray:
        return elementwise("logical_or", other, self)

    def __invert__(self) -> TracedArray:
        return elementwise("logical_not", self)

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        raise TraceAbort("branching on a traced value; use np.where instead of if/while")

    def __float__(self) -> float:
        raise TraceAbort(
            "float() on a traced value: math.* functions are not traceable, "
            "use np.sqrt, np.exp, np.log, ... instead"
        )

    def __int__(self) -> int:
        raise TraceAbort("int() on a traced value")

    def __index__(self) -> int:
        raise TraceAbort("a traced value was used as an index or size")

    def __len__(self) -> int:
        raise TraceAbort("len() on a traced value; use x.shape[-1] or iterate over channels")

    def __iter__(self) -> Iterator[TracedArray]:
        if not self.moved:
            raise TraceAbort(
                "iterating over a traced array (for v in x, r, g, b = x) walks over pixels, not "
                "channels; index channels with x[..., i] or unpack np.moveaxis(x, -1, 0)"
            )
        for i in range(self.n_channels):
            yield self.channel(i)

    def item(self, *args: Any) -> NoReturn:
        raise TraceAbort(".item() on a traced value converts it to a Python scalar")

    def tolist(self) -> NoReturn:
        raise TraceAbort(".tolist() on a traced value converts it to Python lists")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise TraceAbort(f".{name} on a traced value is not traceable")

    def clip(self, a_min: Operand | None = None, a_max: Operand | None = None) -> TracedArray:
        from .functions import _clip

        return _clip(self, a_min, a_max)

    def astype(self, dtype: DTypeLike, *args: Any, **kwargs: Any) -> TracedArray:
        _check_float_dtype(dtype, "astype")
        return TracedArray(self.node)

    def reshape(self, *shape: Any, **kwargs: Any) -> TracedArray:
        return TracedArray(self.node)

    def copy(self, *args: Any, **kwargs: Any) -> TracedArray:
        return TracedArray(self.node)

    def squeeze(self, *args: Any, **kwargs: Any) -> TracedArray:
        return TracedArray(self.node)

    def conj(self) -> TracedArray:
        return TracedArray(self.node)

    @property
    def T(self) -> TracedArray:
        return TracedArray(self.node, moved=True)

    def sqrt(self) -> TracedArray:
        return elementwise("sqrt", self)

    def exp(self) -> TracedArray:
        return elementwise("exp", self)

    def log(self) -> TracedArray:
        return elementwise("log", self)

    def round(self, decimals: int = 0) -> TracedArray:
        if decimals != 0:
            raise TraceAbort("round() with decimals != 0 is not traceable")
        return elementwise("round", self)

    def sum(self, axis: int | None = None, keepdims: bool = False, **kwargs: Any) -> TracedArray:
        from .functions import _sum

        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, keepdims: bool = False, **kwargs: Any) -> TracedArray:
        from .functions import _mean

        return _mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False, **kwargs: Any) -> TracedArray:
        from .functions import _reduce

        return _reduce(self, "reduce_max", axis)

    def min(self, axis: int | None = None, keepdims: bool = False, **kwargs: Any) -> TracedArray:
        from .functions import _reduce

        return _reduce(self, "reduce_min", axis)

    def argmax(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TraceAbort("argmax() is not traceable")

    def argmin(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TraceAbort("argmin() is not traceable")


Operand = TracedArray | Node | float | np.ndarray | Sequence[float]


def _reject_mask_key(item: Any) -> None:
    keys = item if isinstance(item, tuple) else (item,)
    if any(isinstance(k, TracedArray) for k in keys):
        raise TraceAbort(
            "boolean-mask indexing (x[x < 0] = ...) selects pixels by their values; "
            "use np.where, np.maximum or np.clip instead"
        )


def _check_float_dtype(dtype: DTypeLike, where: str) -> None:
    if dtype is None:
        return
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return
    if kind != "f":
        raise TraceAbort(f"{where}({np.dtype(dtype).name}) quantizes or changes the value type")


def _multiply(a: Operand, b: Operand) -> TracedArray:
    for traced, other in ((a, b), (b, a)):
        if isinstance(traced, TracedArray) and traced.n_channels == 1:
            if not isinstance(other, TracedArray):
                const = _operand(other)
                if const.op == "const" and const.meta["value"].ndim == 1:
                    return matmul(traced.node, const.meta["value"].reshape(1, -1))
    return elementwise("mul", a, b)


def _matmul_any(a: Operand, b: Operand) -> TracedArray:
    if isinstance(a, TracedArray) and isinstance(b, TracedArray):
        raise TraceAbort("matmul/dot between two traced values is bilinear, not linear")
    if isinstance(a, TracedArray):
        return matmul(a.node, b)
    if isinstance(b, TracedArray):
        m = np.asarray(a, dtype=np.float64)
        return matmul(b.node, m.T if m.ndim == 2 else m)
    raise TraceAbort("matmul without a traced operand")
