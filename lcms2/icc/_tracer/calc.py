"""
Compile traced graphs into ICC.2 calculator operations.
"""

from __future__ import annotations

import numpy as np

from .chain import _node_width
from .tracing import ELEMENTWISE_OPS, Node, TraceAbort

_NULLARY_SEL = (0, 0)

_CALC_UNARY = {
    "neg": ["neg "],
    "abs": ["abs "],
    "sqrt": ["sqrt"],
    "cbrt": ["cbrt"],
    "square": ["sq  "],
    "sign": ["sign"],
    "floor": ["flor"],
    "ceil": ["ceil"],
    "round": ["rond"],
    "trunc": ["trnc"],
    "exp": ["exp "],
    "log": ["ln  "],
    "log10": ["log "],
    "sin": ["sin "],
    "cos": ["cos "],
    "tan": ["tan "],
    "arcsin": ["asin"],
    "arccos": ["acos"],
    "arctan": ["atan"],
    "logical_not": ["not "],
}

_CALC_BINARY = {
    "add": "add ",
    "sub": "sub ",
    "mul": "mul ",
    "div": "div ",
    "pow": "pow ",
    "maximum": "max ",
    "minimum": "min ",
    "fmax": "max ",
    "fmin": "min ",
    "lt": "lt  ",
    "le": "le  ",
    "gt": "gt  ",
    "ge": "ge  ",
    "eq": "eq  ",
    "ne": "neq ",
    "logical_and": "and ",
    "logical_or": "or  ",
}


def _op(sig):
    return (sig, _NULLARY_SEL)


def _data(value):
    value = float(value)
    if np.isnan(value):
        return _op("NaN ")
    if np.isinf(value):
        return _op("+INF" if value > 0 else "-INF")
    return ("data", value)


_CALC_UNARY_SEQUENCES = {
    "pos": [],
    "log2": [_op("ln  "), _data(1.0 / np.log(2.0)), _op("mul ")],
    "log1p": [_data(1.0), _op("add "), _op("ln  ")],
    "expm1": [_op("exp "), _data(1.0), _op("sub ")],
    "deg2rad": [_data(np.pi / 180.0), _op("mul ")],
    "rad2deg": [_data(180.0 / np.pi), _op("mul ")],
    "sinh": [
        ("copy", (0, 0)),
        _op("exp "),
        ("flip", (0, 0)),
        _op("neg "),
        _op("exp "),
        _op("sub "),
        _data(0.5),
        _op("mul "),
    ],
    "cosh": [
        ("copy", (0, 0)),
        _op("exp "),
        ("flip", (0, 0)),
        _op("neg "),
        _op("exp "),
        _op("add "),
        _data(0.5),
        _op("mul "),
    ],
    "tanh": [
        _data(2.0),
        _op("mul "),
        _op("exp "),
        ("copy", (0, 0)),
        _data(1.0),
        _op("sub "),
        ("flip", (0, 0)),
        _data(1.0),
        _op("add "),
        _op("div "),
    ],
    "arcsinh": [
        ("copy", (0, 0)),
        _op("sq  "),
        _data(1.0),
        _op("add "),
        _op("sqrt"),
        _op("add "),
        _op("ln  "),
    ],
    "arccosh": [
        ("copy", (0, 0)),
        _op("sq  "),
        _data(1.0),
        _op("sub "),
        _op("sqrt"),
        _op("add "),
        _op("ln  "),
    ],
    "arctanh": [
        ("copy", (0, 0)),
        _data(1.0),
        _op("add "),
        ("flip", (0, 0)),
        _op("neg "),
        _data(1.0),
        _op("add "),
        _op("div "),
        _op("ln  "),
        _data(0.5),
        _op("mul "),
    ],
    "isnan": [("copy", (0, 0)), _op("neq ")],
    "isinf": [_op("abs "), _op("+INF"), _op("eq  ")],
    "isfinite": [
        ("copy", (0, 0)),
        ("copy", (0, 0)),
        _op("neq "),
        ("flip", (0, 0)),
        _op("abs "),
        _op("+INF"),
        _op("eq  "),
        _op("or  "),
        _op("not "),
    ],
}

_CALC_BINARY_SEQUENCES = {
    "mod": [
        ("copy", (1, 0)),
        _op("div "),
        _op("flor"),
        _op("mul "),
        _op("sub "),
    ],
    "floor_div": [_op("div "), _op("flor")],
    "hypot": [_op("sq  "), ("flip", (0, 0)), _op("sq  "), _op("add "), _op("sqrt")],
    "copysign": [
        _data(0.0),
        _op("lt  "),
        _data(2.0),
        _op("mul "),
        _data(1.0),
        ("flip", (0, 0)),
        _op("sub "),
        ("flip", (0, 0)),
        _op("abs "),
        _op("mul "),
    ],
}


class _CalcCompiler:
    """Compile traced graphs into ICC.2 calculator operations."""

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.expanded: dict = {}
        self.slots: dict = {}
        self.n_temps = 0

    def channels(self, node: Node, width: int) -> list:
        key = (id(node), width)
        if key not in self.expanded:
            self.expanded[key] = [self.resolve(c) for c in _channels_of(node, width, self.channels)]
        return self.expanded[key]

    def resolve(self, node: Node) -> Node:
        while True:
            if node.op == "select" and node.args[0].op != "input":
                parent = node.args[0]
                width = _node_width(parent)
                if width is None or node.meta["index"] >= width:
                    raise TraceAbort("calc-compile: select from a node of unknown width")
                node = self.channels(parent, width)[node.meta["index"]]
            elif node.op == "matmul" and node.meta["matrix"].shape[1] == 1:
                # a one-channel matmul (a sum or a dot product) used as a scalar
                node = self.channels(node, 1)[0]
            else:
                break
        if node.op == "const" and np.asarray(node.meta["value"]).size != 1:
            raise TraceAbort("calc-compile: vector constant in a scalar position")
        return node

    def children(self, node: Node) -> list:
        if node.op in ("select", "const"):
            return []
        if node.op in ("reduce_max", "reduce_min"):
            return self.channels(node.args[0], node.meta["width"])
        return [self.resolve(a) for a in node.args]

    def _count(self, roots) -> dict:
        counts: dict = {}
        seen: set = set()
        order: list = []

        def visit(node):
            counts[id(node)] = counts.get(id(node), 0) + 1
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in self.children(node):
                visit(child)
            order.append(node)

        for root in roots:
            visit(root)
        return counts, order

    def expr(self, node: Node) -> list:
        slot = self.slots.get(id(node))
        if slot is not None:
            return [("tget", (slot, 0))]
        op = node.op
        if op == "const":
            return [_data(np.asarray(node.meta["value"]).reshape(-1)[0])]
        if op == "select":
            return [("in  ", (node.meta["index"], 0))]
        kids = self.children(node)
        if op in ("reduce_max", "reduce_min"):
            sig = "max " if op == "reduce_max" else "min "
            return [o for k in kids for o in self.expr(k)] + [(sig, (len(kids) - 2, 0))]
        if op == "clip":
            x, lo, hi = kids
            return self.expr(x) + self.expr(lo) + [_op("max ")] + self.expr(hi) + [_op("min ")]
        if op == "where":
            cond, a, b = kids
            then_ops, else_ops = self.expr(a), self.expr(b)
            return self.expr(cond) + [("if  ", len(then_ops)), ("else", len(else_ops))] + then_ops + else_ops
        if op == "interp":
            return self._interp(kids[0], node.meta["xp"], node.meta["fp"])
        if op == "arctan2":
            y, x = kids
            return self.expr(x) + self.expr(y) + [_op("atn2")]
        if op in _CALC_UNARY:
            return self.expr(kids[0]) + [_op(s) for s in _CALC_UNARY[op]]
        if op in _CALC_UNARY_SEQUENCES:
            return self.expr(kids[0]) + list(_CALC_UNARY_SEQUENCES[op])
        if op == "reciprocal":
            return [_data(1.0)] + self.expr(kids[0]) + [_op("div ")]
        if op == "exp2":
            return [_data(2.0)] + self.expr(kids[0]) + [_op("pow ")]
        if op in _CALC_BINARY:
            a, b = kids
            return self.expr(a) + self.expr(b) + [_op(_CALC_BINARY[op])]
        if op in _CALC_BINARY_SEQUENCES:
            a, b = kids
            return self.expr(a) + self.expr(b) + list(_CALC_BINARY_SEQUENCES[op])
        raise TraceAbort(f"calc-compile: no calculator form for {op!r}")

    def _interp(self, x: Node, xp, fp) -> list:
        slot = self._new_slot()
        ops = self.expr(x) + [("tput", (slot, 0)), _data(fp[0])]
        for i in range(len(xp) - 1):
            width = xp[i + 1] - xp[i]
            if width <= 0:
                continue
            slope = (fp[i + 1] - fp[i]) / width
            ops += [
                ("tget", (slot, 0)),
                _data(xp[i]),
                _op("sub "),
                _data(0.0),
                _op("max "),
                _data(width),
                _op("min "),
                _data(slope),
                _op("mul "),
                _op("add "),
            ]
        return ops

    def _new_slot(self) -> int:
        slot = self.n_temps
        self.n_temps += 1
        if self.n_temps > 65535:
            raise TraceAbort("calc-compile: too many temporaries")
        return slot

    def program(self, roots) -> list:
        counts, order = self._count(roots)
        ops: list = []
        for node in order:
            if counts[id(node)] > 1 and node.op not in ("const", "select"):
                body = self.expr(node)
                slot = self._new_slot()
                ops += body + [("tput", (slot, 0))]
                self.slots[id(node)] = slot
        for i, root in enumerate(roots):
            ops += self.expr(root) + [("out ", (i, 0))]
        return ops


def _channels_of(node: Node, width: int, expand=None):
    """Per-channel nodes of a `width`-channel node, expanding operands with `expand`."""
    expand = expand or _channels_of
    if node.op == "stack":
        if len(node.args) != width:
            raise TraceAbort("reduce over a stack of the wrong width")
        return list(node.args)
    if node.op == "const":
        value = np.asarray(node.meta["value"], dtype=np.float64).reshape(-1)
        if value.size == 1:
            return [node] * width
        if value.size != width:
            raise TraceAbort("constant of the wrong width")
        return [Node("const", (), {"value": np.asarray(v)}, width=1) for v in value]
    if node.op == "input" and node.meta.get("n_channels") == width:
        return [Node("select", (node,), {"index": i}, width=1) for i in range(width)]
    if node.op == "matmul":
        (pre,) = node.args
        matrix = node.meta["matrix"]
        in_w, out_w = matrix.shape
        if out_w != width:
            raise TraceAbort("matmul output width doesn't match expected width")
        pre_chans = expand(pre, in_w)
        result = []
        for j in range(out_w):
            terms = []
            for i in range(in_w):
                coeff = float(matrix[i, j])
                if coeff == 0.0:
                    continue
                if coeff == 1.0:
                    terms.append(pre_chans[i])
                else:
                    const = Node("const", (), {"value": np.asarray(coeff)}, width=1)
                    terms.append(Node("mul", (pre_chans[i], const), width=1))
            acc = terms[0] if terms else Node("const", (), {"value": np.asarray(0.0)}, width=1)
            for t in terms[1:]:
                acc = Node("add", (acc, t), width=1)
            result.append(acc)
        return result
    if (node.width or 1) == 1 and width == 1:
        return [node]
    if node.op in ELEMENTWISE_OPS:
        expanded = [expand(a, width) if (a.width or 1) == width else [a] * width for a in node.args]
        return [Node(node.op, tuple(e[i] for e in expanded), node.meta, width=1) for i in range(width)]
    raise TraceAbort(f"cannot expand op {node.op!r} into {width} channels for calc compilation")


def compile_calc_program(root: Node, n_in: int, n_out: int) -> list:
    """Compile a traced graph into ICC.2 calculator operations.

    Returns
    -------
    list of tuple
        ``(sig, arg)`` operations for `mpe.encode_calc_element`.

    Raises
    ------
    TraceAbort
        If the graph uses an operation the calculator cannot express.
    """
    compiler = _CalcCompiler(n_in)
    roots = compiler.channels(root, n_out)
    return compiler.program(roots)
