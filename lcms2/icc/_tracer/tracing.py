"""Record numpy expressions on a placeholder array as a node graph."""

from __future__ import annotations

import os
import threading
import traceback

import numpy as np


class Node:
    """One operation of a traced expression graph.

    Attributes
    ----------
    op : str
        Operation name.
    args : tuple of Node
        Operand nodes.
    meta : dict
        Operation parameters (a constant value, a channel index, a matrix, ...).
    width : int or None
        Number of channels the node carries, when known.
    """

    __slots__ = ("op", "args", "meta", "width")

    def __init__(self, op, args, meta=None, width=None):
        self.op = op
        self.args = args
        self.meta = meta or {}
        self.width = width

    def __repr__(self):
        return f"Node({self.op}, width={self.width}, meta={self.meta})"


class TraceAbort(Exception):
    """Raised when a function uses something that cannot be traced."""


_SUPPORTED_UFUNCS = {
    np.add: "add",
    np.subtract: "sub",
    np.multiply: "mul",
    np.true_divide: "div",
    np.power: "pow",
    np.float_power: "pow",
    np.negative: "neg",
    np.positive: "pos",
    np.absolute: "abs",
    np.fabs: "abs",
    np.mod: "mod",
    np.remainder: "mod",
    np.fmod: "mod",
    np.floor_divide: "floor_div",
    np.reciprocal: "reciprocal",
    np.square: "square",
    np.sign: "sign",
    np.sqrt: "sqrt",
    np.cbrt: "cbrt",
    np.exp: "exp",
    np.exp2: "exp2",
    np.expm1: "expm1",
    np.log: "log",
    np.log2: "log2",
    np.log10: "log10",
    np.log1p: "log1p",
    np.floor: "floor",
    np.ceil: "ceil",
    np.rint: "round",
    np.trunc: "trunc",
    np.sin: "sin",
    np.cos: "cos",
    np.tan: "tan",
    np.arcsin: "arcsin",
    np.arccos: "arccos",
    np.arctan: "arctan",
    np.arctan2: "arctan2",
    np.hypot: "hypot",
    np.sinh: "sinh",
    np.cosh: "cosh",
    np.tanh: "tanh",
    np.arcsinh: "arcsinh",
    np.arccosh: "arccosh",
    np.arctanh: "arctanh",
    np.deg2rad: "deg2rad",
    np.radians: "deg2rad",
    np.rad2deg: "rad2deg",
    np.degrees: "rad2deg",
    np.maximum: "maximum",
    np.minimum: "minimum",
    np.fmax: "fmax",
    np.fmin: "fmin",
    np.copysign: "copysign",
    np.equal: "eq",
    np.not_equal: "ne",
    np.less: "lt",
    np.less_equal: "le",
    np.greater: "gt",
    np.greater_equal: "ge",
    np.logical_and: "logical_and",
    np.logical_or: "logical_or",
    np.logical_not: "logical_not",
    np.isnan: "isnan",
    np.isfinite: "isfinite",
    np.isinf: "isinf",
}


UNARY_FUNCS = {
    "neg": np.negative,
    "pos": np.positive,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "square": np.square,
    "sign": np.sign,
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "reciprocal": np.reciprocal,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "trunc": np.trunc,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "deg2rad": np.deg2rad,
    "rad2deg": np.rad2deg,
    "isnan": np.isnan,
    "isfinite": np.isfinite,
    "isinf": np.isinf,
    "logical_not": np.logical_not,
}

BINARY_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "fmax": np.fmax,
    "fmin": np.fmin,
    "copysign": np.copysign,
    "mod": np.mod,
    "floor_div": np.floor_divide,
    "arctan2": np.arctan2,
    "hypot": np.hypot,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
}

COMPARISON_FUNCS = {
    "le": np.less_equal,
    "lt": np.less,
    "ge": np.greater_equal,
    "gt": np.greater,
    "eq": np.equal,
    "ne": np.not_equal,
}


def const_node(value) -> Node:
    """A constant node; a 1-D value with more than one entry is per-channel."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1:
        if any(d != 1 for d in arr.shape[:-1]):
            raise TraceAbort(f"constant of shape {arr.shape} is not a per-channel vector")
        arr = arr.reshape(-1)
    if arr.ndim == 1 and arr.size == 1:
        arr = arr.reshape(())
    return Node("const", (), {"value": arr}, width=arr.size if arr.ndim == 1 else 1)


ELEMENTWISE_OPS = frozenset(_SUPPORTED_UFUNCS.values()) | {"clip", "where", "interp"}


def select_node(node: Node, index: int) -> Node:
    """Channel `index` of `node`, pushed through elementwise operations."""
    if node.op == "stack":
        return node.args[index]
    if node.op == "const":
        value = node.meta["value"]
        return const_node(value[index] if value.ndim == 1 else value)
    if node.op == "input":
        return Node("select", (node,), {"index": index}, width=1)
    if (node.width or 1) == 1:
        return node
    if node.op in ELEMENTWISE_OPS:
        args = tuple(
            select_node(a, index) if (a.width or 1) == node.width else a for a in node.args
        )
        return Node(node.op, args, node.meta, width=1)
    return Node("select", (node,), {"index": index}, width=1)


def per_channel(node: Node) -> Node:
    """Rewrite a multi-channel elementwise node as a stack of per-channel nodes."""
    if (node.width or 1) > 1 and node.op in ELEMENTWISE_OPS:
        return stack_node([select_node(node, i) for i in range(node.width)])
    return node


def stack_node(nodes) -> Node:
    """Concatenate 1-channel nodes along the channel axis."""
    return Node("stack", tuple(nodes), width=len(nodes))


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
        handler = _FUNCTIONS.get(func)
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
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False, **kwargs):
        return _mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False, **kwargs):
        return _reduce(self, "reduce_max", axis)

    def min(self, axis=None, keepdims=False, **kwargs):
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


def _as_traced(x) -> TracedArray:
    return x if isinstance(x, TracedArray) else TracedArray(_operand(x))


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


def _check_channel_axis(axis):
    if axis is None:
        raise TraceAbort(
            "a reduction over all axes (no axis= given) mixes different pixels; "
            "reduce over the channels with axis=-1"
        )
    if axis != -1:
        raise TraceAbort(f"reduction over axis={axis!r} is not a channel-axis reduction")


def _sum(x, axis=None, keepdims=False, **kwargs):
    x = _as_traced(x)
    _check_channel_axis(axis)
    return matmul(x.node, np.ones((x.n_channels, 1))).channel(0)


def _mean(x, axis=None, keepdims=False, **kwargs):
    x = _as_traced(x)
    _check_channel_axis(axis)
    return matmul(x.node, np.full((x.n_channels, 1), 1.0 / x.n_channels)).channel(0)


def _reduce(x, op, axis=None, *args, **kwargs):
    x = _as_traced(x)
    _check_channel_axis(axis)
    if x.n_channels == 1:
        return x
    return TracedArray(Node(op, (x.node,), {"width": x.n_channels}, width=1))


def _clip(x, a_min=None, a_max=None, **kwargs):
    a_min = kwargs.pop("min", a_min)
    a_max = kwargs.pop("max", a_max)
    lo = -np.inf if a_min is None else a_min
    hi = np.inf if a_max is None else a_max
    return elementwise("clip", x, lo, hi)


def _where(cond, a=None, b=None):
    if a is None or b is None:
        raise TraceAbort("np.where with one argument is not traceable")
    return elementwise("where", cond, a, b)


def _select(condlist, choicelist, default=0.0):
    result = default
    for cond, choice in zip(reversed(list(condlist)), reversed(list(choicelist)), strict=False):
        result = elementwise("where", cond, choice, result)
    return _as_traced(result)


def _stack(arrays, axis=-1, **kwargs):
    arrays = list(arrays)
    nodes = []
    for a in arrays:
        node = _operand(a)
        if (node.width or 1) != 1:
            raise TraceAbort("np.stack of multi-channel traced values is not traceable")
        nodes.append(node)
    return TracedArray(stack_node(nodes))


def _concatenate(arrays, axis=-1, **kwargs):
    if axis not in (-1, None):
        raise TraceAbort(f"np.concatenate along axis={axis!r} is not a channel concat")
    nodes = []
    for a in arrays:
        node = _operand(a)
        nodes.extend(select_node(node, i) for i in range(node.width or 1))
    return TracedArray(stack_node(nodes))


def _split(x, sections, axis=-1):
    x = _as_traced(x)
    if axis not in (-1, x.n_channels - 1):
        raise TraceAbort(f"np.split along axis={axis!r} is not a channel split")
    n = x.n_channels
    if isinstance(sections, int):
        if n % sections:
            raise TraceAbort("np.split into unequal channel groups")
        bounds = list(range(0, n + 1, n // sections))
    else:
        bounds = [0] + [int(s) for s in sections] + [n]
    return [x._channels(range(a, b)) for a, b in zip(bounds[:-1], bounds[1:], strict=False)]


def _unstack(x, axis=-1):
    x = _as_traced(x)
    return tuple(x.channel(i) for i in range(x.n_channels))


def _identity_function(x, *args, **kwargs):
    return _as_traced(x)


def _copy_function(x, *args, **kwargs):
    return TracedArray(_operand(x))


def _like(fill):
    def handler(x, *args, **kwargs):
        value = fill if fill is not None else (args[0] if args else kwargs["fill_value"])
        width = _operand(x).width or 1
        return TracedArray(const_node(np.full(width, float(value)) if width > 1 else float(value)))

    return handler


def _full(shape, fill_value, *args, **kwargs):
    return TracedArray(const_node(float(fill_value)))


def _norm(x, ord=None, axis=None, keepdims=False):
    if ord not in (None, 2):
        raise TraceAbort("only the 2-norm over the channel axis is traceable")
    _check_channel_axis(axis)
    x = _as_traced(x)
    return elementwise("sqrt", _sum(elementwise("square", x), axis=-1))


def _column_stack(arrays):
    return _concatenate(arrays)


def _polyval(p, x):
    if isinstance(p, TracedArray):
        raise TraceAbort("np.polyval with traced coefficients is not traceable")
    coeffs = np.asarray(p, dtype=np.float64).ravel()
    result = _as_traced(x) * 0.0 + coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result


def _interp(x, xp, fp, left=None, right=None, period=None):
    if isinstance(xp, TracedArray) or isinstance(fp, TracedArray):
        raise TraceAbort("np.interp with traced breakpoints is not traceable")
    if period is not None or left is not None or right is not None:
        raise TraceAbort("np.interp with left/right/period is not traceable")
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    return elementwise("interp", x, meta={"xp": xp, "fp": fp})


def _repeat(a, repeats, axis=None):
    if axis not in (-1,) or not isinstance(repeats, (int, np.integer)):
        raise TraceAbort("np.repeat is only traceable with an integer count along axis=-1")
    x = _as_traced(a)
    return TracedArray(stack_node([select_node(x.node, i) for i in range(x.n_channels) for _ in range(int(repeats))]))


def _tile(a, reps):
    count = reps if isinstance(reps, (int, np.integer)) else (reps[-1] if len(reps) == 1 else None)
    if count is None:
        raise TraceAbort("np.tile is only traceable along the channel axis")
    x = _as_traced(a)
    return TracedArray(stack_node([select_node(x.node, i) for _ in range(int(count)) for i in range(x.n_channels)]))


def _is_channel_axis_pair(a, b) -> bool:
    return {a, b} == {-1, 0}


def _moveaxis(x, source, destination):
    if not _is_channel_axis_pair(source, destination):
        raise TraceAbort(f"np.moveaxis({source}, {destination}) is not a channel-axis move")
    return TracedArray(_as_traced(x).node, moved=True)


def _swapaxes(x, axis1, axis2):
    if not _is_channel_axis_pair(axis1, axis2):
        raise TraceAbort(f"np.swapaxes({axis1}, {axis2}) is not a channel-axis swap")
    return TracedArray(_as_traced(x).node, moved=True)


def _transpose(x, axes=None):
    if axes is not None:
        raise TraceAbort("np.transpose with explicit axes is not traceable")
    return TracedArray(_as_traced(x).node, moved=True)


def _tensordot(a, b, axes=2):
    traced_first = isinstance(a, TracedArray)
    matrix = np.asarray(b if traced_first else a, dtype=np.float64)
    if matrix.ndim != 2:
        raise TraceAbort("np.tensordot is only traceable with a constant matrix")
    if isinstance(axes, int):
        pair = ([-1], [0]) if (axes == 1 and traced_first) else ([-1], [1]) if axes == 1 else None
    else:
        pair = (list(np.atleast_1d(axes[0])), list(np.atleast_1d(axes[1])))
    if pair is None or len(pair[0]) != 1 or len(pair[1]) != 1:
        raise TraceAbort(f"np.tensordot with axes={axes!r} is not a channel contraction")
    t_axis, m_axis = (int(pair[0][0]), int(pair[1][0])) if traced_first else (int(pair[1][0]), int(pair[0][0]))
    if t_axis != -1 or m_axis not in (0, 1, -1, -2):
        raise TraceAbort(f"np.tensordot with axes={axes!r} is not a channel contraction")
    x = _as_traced(a if traced_first else b)
    # the traced channels contract with the matrix axis m_axis; the other matrix axis is the output
    return matmul(x.node, matrix if m_axis in (0, -2) else matrix.T)


def _average(x, axis=None, weights=None, **kwargs):
    x = _as_traced(x)
    _check_channel_axis(axis)
    if weights is None:
        return _mean(x, axis=-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != x.n_channels or w.sum() == 0:
        raise TraceAbort("np.average weights do not match the channels")
    return matmul(x.node, (w / w.sum()).reshape(-1, 1)).channel(0)


def _dot(a, b, out=None):
    return _matmul_any(a, b)


def _einsum(subscripts, *operands, **kwargs):
    if len(operands) != 2:
        raise TraceAbort("np.einsum is only traceable for two operands")
    spec = subscripts.replace(" ", "")
    inputs, _, output = spec.partition("->")
    left, right = inputs.split(",")
    a, b = operands
    traced_first = isinstance(a, TracedArray)
    t_sub, m_sub = (left, right) if traced_first else (right, left)
    matrix = np.asarray(b if traced_first else a, dtype=np.float64)
    t_idx, out_idx = t_sub.replace("...", ""), output.replace("...", "")
    if len(t_idx) != 1 or len(out_idx) != 1 or len(m_sub) != 2 or matrix.ndim != 2:
        raise TraceAbort(f"unsupported einsum {subscripts!r}")
    if m_sub == t_idx + out_idx:
        return matmul(_as_traced(a if traced_first else b).node, matrix)
    if m_sub == out_idx + t_idx:
        return matmul(_as_traced(a if traced_first else b).node, matrix.T)
    raise TraceAbort(f"unsupported einsum {subscripts!r}")


_FUNCTIONS = {
    np.stack: _stack,
    np.concatenate: _concatenate,
    np.clip: _clip,
    np.where: _where,
    np.select: _select,
    np.dot: _dot,
    np.matmul: _dot,
    np.einsum: _einsum,
    np.sum: _sum,
    np.mean: _mean,
    np.max: lambda x, axis=None, **kw: _reduce(x, "reduce_max", axis),
    np.amax: lambda x, axis=None, **kw: _reduce(x, "reduce_max", axis),
    np.min: lambda x, axis=None, **kw: _reduce(x, "reduce_min", axis),
    np.amin: lambda x, axis=None, **kw: _reduce(x, "reduce_min", axis),
    np.split: _split,
    np.array_split: _split,
    np.dsplit: lambda x, sections: _split(x, sections),
    np.interp: _interp,
    np.round: lambda x, decimals=0, out=None: _as_traced(x).round(decimals),
    np.around: lambda x, decimals=0, out=None: _as_traced(x).round(decimals),
    np.ones_like: _like(1.0),
    np.zeros_like: _like(0.0),
    np.full_like: _like(None),
    np.full: _full,
    np.empty_like: _like(0.0),
    np.linalg.norm: _norm,
    np.column_stack: _column_stack,
    np.dstack: _column_stack,
    np.hstack: _column_stack,
    np.polyval: _polyval,
    np.copy: _copy_function,
    np.broadcast_to: _copy_function,
    np.atleast_1d: _copy_function,
    np.expand_dims: _copy_function,
    np.squeeze: _copy_function,
    np.transpose: _transpose,
    np.moveaxis: _moveaxis,
    np.swapaxes: _swapaxes,
    np.tensordot: _tensordot,
    np.repeat: _repeat,
    np.tile: _tile,
    np.average: _average,
    np.asarray: _identity_function,
    np.asanyarray: _identity_function,
    np.ascontiguousarray: _copy_function,
    np.real: _copy_function,
    np.nan_to_num: _copy_function,
}
if hasattr(np, "unstack"):
    _FUNCTIONS[np.unstack] = _unstack


def stack_channels(traced_list) -> TracedArray:
    """Stack 1-channel traced values into one multi-channel value."""
    return _stack(traced_list)


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


def _forbidden_python_loop(name):
    def forbidden(*args, **kwargs):
        raise TraceAbort(f"np.{name} calls a Python function for every element")

    return forbidden


def _forbidden_random(name):
    def forbidden(*args, **kwargs):
        raise TraceAbort(f"np.random.{name}: random numbers make the transform non-deterministic")

    return forbidden


class _PatchArrayConstructors:
    """Make ``np.asarray``/``np.array`` pass traced values through while tracing."""

    def __enter__(self):
        _TRACE_LOCK.acquire()
        self._asarray, self._array = np.asarray, np.array
        orig_asarray, orig_array = self._asarray, self._array

        def asarray(a, *args, **kwargs):
            if isinstance(a, TracedArray):
                _check_float_dtype(kwargs.get("dtype", args[0] if args else None), "np.asarray")
                return a
            return orig_asarray(a, *args, **kwargs)

        def array(a, *args, **kwargs):
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

    def __exit__(self, *exc):
        np.asarray, np.array = self._asarray, self._array
        for name, original in self._random.items():
            if name.startswith("np."):
                setattr(np, name[3:], original)
            else:
                setattr(np.random, name, original)
        _TRACE_LOCK.release()
        return False


def _numpy_frame(error) -> str:
    """`` (inside numpy's <function>)`` for the first NumPy function an exception passed through."""
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if f"{os.sep}numpy{os.sep}" in frame.f_code.co_filename:
            return f" (inside numpy's {getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)})"
    return ""


def trace_function(fn, n_channels: int):
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
    from .optimize import optimize_node

    return optimize_node(per_channel(y.node)), y.n_channels
