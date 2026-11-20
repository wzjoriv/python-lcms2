"""NumPy functions recorded on traced arrays, registered in `FUNCTIONS`."""

from __future__ import annotations

import numpy as np

from .array import TracedArray, _matmul_any, _operand, elementwise, matmul
from .graph import Node, TraceAbort, const_node, select_node, stack_node


def _as_traced(x) -> TracedArray:
    return x if isinstance(x, TracedArray) else TracedArray(_operand(x))


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


def _norm(x, ord=None, axis=None, keepdims=False):
    if ord not in (None, 2):
        raise TraceAbort("only the 2-norm over the channel axis is traceable")
    _check_channel_axis(axis)
    x = _as_traced(x)
    return elementwise("sqrt", _sum(elementwise("square", x), axis=-1))


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
    t_axis, m_axis = (
        (int(pair[0][0]), int(pair[1][0])) if traced_first else (int(pair[1][0]), int(pair[0][0]))
    )
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
    return TracedArray(
        stack_node([select_node(x.node, i) for i in range(x.n_channels) for _ in range(int(repeats))])
    )


def _tile(a, reps):
    count = reps if isinstance(reps, (int, np.integer)) else (reps[-1] if len(reps) == 1 else None)
    if count is None:
        raise TraceAbort("np.tile is only traceable along the channel axis")
    x = _as_traced(a)
    return TracedArray(
        stack_node([select_node(x.node, i) for _ in range(int(count)) for i in range(x.n_channels)])
    )


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


def stack_channels(traced_list) -> TracedArray:
    """Stack 1-channel traced values into one multi-channel value."""
    return _stack(traced_list)


FUNCTIONS = {
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
    FUNCTIONS[np.unstack] = _unstack
