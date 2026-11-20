"""Nodes of a traced expression graph and the operation tables."""

from __future__ import annotations

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
        args = tuple(select_node(a, index) if (a.width or 1) == node.width else a for a in node.args)
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
