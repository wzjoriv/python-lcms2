"""
Constant folding, subexpression sharing and algebraic simplification of traced graphs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .tracing import (
    BINARY_FUNCS,
    COMPARISON_FUNCS,
    UNARY_FUNCS,
    Node,
    TraceAbort,
    const_node,
    select_node,
)


def _eval_const(node: Node) -> np.ndarray:
    """Evaluate a graph whose leaves are all constants."""
    op = node.op
    if op == "const":
        return np.asarray(node.meta["value"], dtype=np.float64)
    if op == "select":
        (parent,) = node.args
        val = _eval_const(parent)
        idx = node.meta["index"]
        return np.asarray(val)[..., idx] if np.ndim(val) else np.asarray(val)
    if op == "stack":
        return np.stack([_eval_const(a) for a in node.args], axis=-1)
    if op == "matmul":
        (pre,) = node.args
        return _eval_const(pre) @ node.meta["matrix"]
    if op == "clip":
        x, lo, hi = node.args
        return np.clip(_eval_const(x), _eval_const(lo), _eval_const(hi))
    if op == "where":
        c, a, b = node.args
        return np.where(_eval_const(c), _eval_const(a), _eval_const(b))
    if op == "interp":
        (a,) = node.args
        return np.interp(_eval_const(a), node.meta["xp"], node.meta["fp"])
    if op in ("reduce_max", "reduce_min"):
        (a,) = node.args
        v = _eval_const(a)
        return v.max(axis=-1) if op == "reduce_max" else v.min(axis=-1)
    if op == "pow":
        a, b = node.args
        return np.power(_eval_const(a), _eval_const(b))
    if op in UNARY_FUNCS:
        (a,) = node.args
        return UNARY_FUNCS[op](_eval_const(a))
    if op in BINARY_FUNCS:
        a, b = node.args
        return BINARY_FUNCS[op](_eval_const(a), _eval_const(b))
    if op in COMPARISON_FUNCS:
        a, b = node.args
        return COMPARISON_FUNCS[op](_eval_const(a), _eval_const(b))
    raise TraceAbort(f"optimize: cannot constant-fold op {op!r}")


def _make_const(value: ArrayLike) -> Node:
    return const_node(value)


def _const_equals(node: Node, value: float) -> bool:
    if node.op != "const":
        return False
    v = np.asarray(node.meta["value"])
    return v.size == 1 and float(np.asarray(v).reshape(-1)[0]) == value


def _simplify(node: Node) -> Node:
    op = node.op
    args = node.args
    if op == "add":
        a, b = args
        if _const_equals(b, 0.0):
            return a
        if _const_equals(a, 0.0):
            return b
    elif op == "sub":
        a, b = args
        if _const_equals(b, 0.0):
            return a
    elif op == "mul":
        a, b = args
        if _const_equals(b, 1.0):
            return a
        if _const_equals(a, 1.0):
            return b
        if (_const_equals(a, 0.0) and (b.width or 1) == 1) or (_const_equals(b, 0.0) and (a.width or 1) == 1):
            return _make_const(0.0)
    elif op == "div":
        a, b = args
        if _const_equals(b, 1.0):
            return a
    elif op == "neg":
        (a,) = args
        if a.op == "neg":
            return a.args[0]
    elif op == "pos":
        (a,) = args
        return a
    elif op == "select":
        (a,) = args
        if a.op in ("stack", "const") or ((a.width or 1) == 1 and a.op != "input"):
            return select_node(a, node.meta["index"])
    elif op == "where":
        cond, a, b = args
        if cond.op == "const":
            chosen = np.asarray(cond.meta["value"])
            if chosen.size == 1:
                return a if bool(chosen.reshape(-1)[0]) else b
    return node


def _meta_key(meta: dict[str, Any]) -> tuple:
    items = []
    for k in sorted(meta):
        v = meta[k]
        if isinstance(v, np.ndarray):
            items.append((k, "ndarray", v.shape, v.tobytes()))
        else:
            items.append((k, v))
    return tuple(items)


def _struct_key(op: str, args: Iterable[Node], meta: dict[str, Any]) -> tuple:
    return (op, tuple(id(a) for a in args), _meta_key(meta))


def optimize_node(root: Node) -> Node:
    """Return an equivalent, simplified graph for `root`."""
    memo: dict[int, Node] = {}
    cache: dict = {}

    def cse(new_node: Node) -> Node:
        key = _struct_key(new_node.op, new_node.args, new_node.meta)
        existing = cache.get(key)
        if existing is not None:
            return existing
        cache[key] = new_node
        return new_node

    def visit(node: Node) -> Node:
        cached = memo.get(id(node))
        if cached is not None:
            return cached

        if node.op in ("input", "const"):
            result = cse(node) if node.op == "const" else node
            memo[id(node)] = result
            return result

        new_args = tuple(visit(a) for a in node.args)

        if new_args and all(a.op == "const" for a in new_args):
            try:
                value = _eval_const(Node(node.op, new_args, node.meta, node.width))
            except Exception:
                value = None
            if value is not None:
                folded = cse(_make_const(value))
                memo[id(node)] = folded
                return folded

        rebuilt = node if new_args == node.args else Node(node.op, new_args, node.meta, node.width)
        simplified = _simplify(rebuilt)
        if simplified is not rebuilt:
            result = cse(simplified)
            memo[id(node)] = result
            return result

        result = cse(rebuilt)
        memo[id(node)] = result
        return result

    return visit(root)
