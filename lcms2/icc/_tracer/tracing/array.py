"""`TracedArray`, the placeholder whose operations are recorded as a graph."""

from __future__ import annotations

import numpy as np

from .graph import (
    _SUPPORTED_UFUNCS,
    Node,
    TraceAbort,
    const_node,
    select_node,
    stack_node,
)


def _operand(x):
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


def elementwise(op: str, *operands, meta=None) -> TracedArray:
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


def matmul(node: Node, matrix) -> TracedArray:
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
    def shape(self):
        return (self.n_channels,)

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return self.n_channels

    @property
    def dtype(self):
        return np.dtype(np.float64)

    @classmethod
    def input(cls, n_channels: int) -> TracedArray:
        return cls(Node("input", (), {"n_channels": n_channels}, width=n_channels))

    def channel(self, i: int) -> TracedArray:
        n = self.n_channels
        if not -n <= i < n:
            raise TraceAbort(f"channel index {i} out of range for {n} channels")
        return TracedArray(select_node(self.node, i % n))

    def _channels(self, indices) -> TracedArray:
        picked = [select_node(self.node, i % self.n_channels) for i in indices]
        if len(picked) == 1:
            return TracedArray(picked[0])
        return TracedArray(stack_node(picked))

    def _key_indices(self, item) -> list:
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

    def __getitem__(self, item):
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

    def __setitem__(self, item, value):
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    def __array_function__(self, func, types, args, kwargs):
        from .functions import FUNCTIONS

        handler = FUNCTIONS.get(func)
        if handler is None:
            raise TraceAbort(f"unsupported numpy function {getattr(func, '__name__', func)}")
        return handler(*args, **kwargs)

    def __matmul__(self, other):
        return _matmul_any(self, other)

    def __rmatmul__(self, other):
        return _matmul_any(other, self)

    def dot(self, other):
        return _matmul_any(self, other)

    def __add__(self, other):
        return elementwise("add", self, other)

    def __radd__(self, other):
        return elementwise("add", other, self)

    def __sub__(self, other):
        return elementwise("sub", self, other)

    def __rsub__(self, other):
        return elementwise("sub", other, self)

    def __mul__(self, other):
        return _multiply(self, other)

    def __rmul__(self, other):
        return _multiply(other, self)

    def __truediv__(self, other):
        return elementwise("div", self, other)

    def __rtruediv__(self, other):
        return elementwise("div", other, self)

    def __pow__(self, other):
        return elementwise("pow", self, other)

    def __rpow__(self, other):
        return elementwise("pow", other, self)

    def __mod__(self, other):
        return elementwise("mod", self, other)

    def __rmod__(self, other):
        return elementwise("mod", other, self)

    def __floordiv__(self, other):
        return elementwise("floor_div", self, other)

    def __rfloordiv__(self, other):
        return elementwise("floor_div", other, self)

    def __neg__(self):
        return elementwise("neg", self)

    def __pos__(self):
        return self

    def __abs__(self):
        return elementwise("abs", self)

    def __eq__(self, other):
        return elementwise("eq", self, other)

    def __ne__(self, other):
        return elementwise("ne", self, other)

    def __lt__(self, other):
        return elementwise("lt", self, other)

    def __le__(self, other):
        return elementwise("le", self, other)

    def __gt__(self, other):
        return elementwise("gt", self, other)

    def __ge__(self, other):
        return elementwise("ge", self, other)

    def __and__(self, other):
        return elementwise("logical_and", self, other)

    def __rand__(self, other):
        return elementwise("logical_and", other, self)

    def __or__(self, other):
        return elementwise("logical_or", self, other)

    def __ror__(self, other):
        return elementwise("logical_or", other, self)

    def __invert__(self):
        return elementwise("logical_not", self)

    def __hash__(self):
        return id(self)

    def __bool__(self):
        raise TraceAbort("branching on a traced value; use np.where instead of if/while")

    def __float__(self):
        raise TraceAbort(
            "float() on a traced value: math.* functions are not traceable, "
            "use np.sqrt, np.exp, np.log, ... instead"
        )

    def __int__(self):
        raise TraceAbort("int() on a traced value")

    def __index__(self):
        raise TraceAbort("a traced value was used as an index or size")

    def __len__(self):
        raise TraceAbort("len() on a traced value; use x.shape[-1] or iterate over channels")

    def __iter__(self):
        if not self.moved:
            raise TraceAbort(
                "iterating over a traced array (for v in x, r, g, b = x) walks over pixels, not "
                "channels; index channels with x[..., i] or unpack np.moveaxis(x, -1, 0)"
            )
        for i in range(self.n_channels):
            yield self.channel(i)

    def item(self, *args):
        raise TraceAbort(".item() on a traced value converts it to a Python scalar")

    def tolist(self):
        raise TraceAbort(".tolist() on a traced value converts it to Python lists")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise TraceAbort(f".{name} on a traced value is not traceable")

    def clip(self, a_min=None, a_max=None):
        from .functions import _clip

        return _clip(self, a_min, a_max)

    def astype(self, dtype, *args, **kwargs):
        _check_float_dtype(dtype, "astype")
        return TracedArray(self.node)

    def reshape(self, *shape, **kwargs):
        return TracedArray(self.node)

    def copy(self, *args, **kwargs):
        return TracedArray(self.node)

    def squeeze(self, *args, **kwargs):
        return TracedArray(self.node)

    def conj(self):
        return TracedArray(self.node)

    @property
    def T(self):
        return TracedArray(self.node, moved=True)

    def sqrt(self):
        return elementwise("sqrt", self)

    def exp(self):
        return elementwise("exp", self)

    def log(self):
        return elementwise("log", self)

    def round(self, decimals=0):
        if decimals != 0:
            raise TraceAbort("round() with decimals != 0 is not traceable")
        return elementwise("round", self)

    def sum(self, axis=None, keepdims=False, **kwargs):
        from .functions import _sum

        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False, **kwargs):
        from .functions import _mean

        return _mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False, **kwargs):
        from .functions import _reduce

        return _reduce(self, "reduce_max", axis)

    def min(self, axis=None, keepdims=False, **kwargs):
        from .functions import _reduce

        return _reduce(self, "reduce_min", axis)

    def argmax(self, *args, **kwargs):
        raise TraceAbort("argmax() is not traceable")

    def argmin(self, *args, **kwargs):
        raise TraceAbort("argmin() is not traceable")


def _reject_mask_key(item):
    keys = item if isinstance(item, tuple) else (item,)
    if any(isinstance(k, TracedArray) for k in keys):
        raise TraceAbort(
            "boolean-mask indexing (x[x < 0] = ...) selects pixels by their values; "
            "use np.where, np.maximum or np.clip instead"
        )


def _check_float_dtype(dtype, where):
    if dtype is None:
        return
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return
    if kind != "f":
        raise TraceAbort(f"{where}({np.dtype(dtype).name}) quantizes or changes the value type")


def _multiply(a, b):
    for traced, other in ((a, b), (b, a)):
        if isinstance(traced, TracedArray) and traced.n_channels == 1:
            if not isinstance(other, TracedArray):
                const = _operand(other)
                if const.op == "const" and const.meta["value"].ndim == 1:
                    return matmul(traced.node, const.meta["value"].reshape(1, -1))
    return elementwise("mul", a, b)


def _matmul_any(a, b):
    if isinstance(a, TracedArray) and isinstance(b, TracedArray):
        raise TraceAbort("matmul/dot between two traced values is bilinear, not linear")
    if isinstance(a, TracedArray):
        return matmul(a.node, b)
    if isinstance(b, TracedArray):
        m = np.asarray(a, dtype=np.float64)
        return matmul(b.node, m.T if m.ndim == 2 else m)
    raise TraceAbort("matmul without a traced operand")
