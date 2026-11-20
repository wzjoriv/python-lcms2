"""Decompose traced graphs into chains of matrix and per-channel curve stages."""

from __future__ import annotations

import numpy as np

from .tracing import (
    BINARY_FUNCS,
    COMPARISON_FUNCS,
    ELEMENTWISE_OPS,
    UNARY_FUNCS,
    Node,
    TraceAbort,
)


def _peel_one_matmul(node: Node):
    """Split ``X @ M (+ offset)`` into ``(X, M, offset)``, or None."""
    n = node
    offset = None
    if n.op == "add":
        left, right = n.args
        if right.op == "const":
            n, off_node = left, right
        elif left.op == "const":
            n, off_node = right, left
        else:
            return None
        offset = off_node.meta["value"]
    if n.op != "matmul":
        return None
    (pre,) = n.args
    matrix = n.meta["matrix"]
    if offset is None:
        offset = np.zeros(matrix.shape[-1])
    offset = np.broadcast_to(offset, (matrix.shape[-1],)).astype(np.float64).copy()
    return pre, matrix, offset


def _peel_matrix(node: Node):
    """Split consecutive ``@ M (+ offset)`` layers into ``(X, M_total, offset_total)``, or None."""
    layers = []
    cur = node
    while True:
        peeled = _peel_one_matmul(cur)
        if peeled is None:
            break
        pre, matrix, offset = peeled
        layers.append((matrix, offset))
        cur = pre
    if not layers:
        return None
    layers.reverse()
    total_matrix, total_offset = layers[0]
    for matrix, offset in layers[1:]:
        total_offset = total_offset @ matrix + offset
        total_matrix = total_matrix @ matrix
    return cur, total_matrix, total_offset


def _linear_terms(node: Node):
    """Express `node` as ``const + sum(coeff * select(...))``.



    Returns


    -------


    tuple or None


        ``(const, {(id(parent), index): (parent, coeff)})``, or None if `node`


        is not affine in channel selections.


    """
    if node.op == "const":
        value = np.asarray(node.meta["value"])
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0]), {}
    if node.op == "select":
        shared = node.args[0]
        idx = node.meta["index"]
        return 0.0, {(id(shared), idx): (shared, 1.0)}
    if node.op == "neg":
        (a,) = node.args
        sub = _linear_terms(a)
        if sub is None:
            return None
        c, terms = sub
        return -c, {k: (s, -co) for k, (s, co) in terms.items()}
    if node.op in ("add", "sub"):
        a, b = node.args
        ta, tb = _linear_terms(a), _linear_terms(b)
        if ta is None or tb is None:
            return None
        ca, tsa = ta
        cb, tsb = tb
        sign = 1.0 if node.op == "add" else -1.0
        merged = dict(tsa)
        for k, (s, co) in tsb.items():
            if k in merged:
                s0, co0 = merged[k]
                merged[k] = (s0, co0 + sign * co)
            else:
                merged[k] = (s, sign * co)
        return ca + sign * cb, merged
    if node.op == "mul":
        a, b = node.args
        if a.op == "const":
            const_node, other = a, b
        elif b.op == "const":
            const_node, other = b, a
        else:
            return None
        if np.asarray(const_node.meta["value"]).size != 1:
            return None
        const_val = float(np.asarray(const_node.meta["value"]).reshape(-1)[0])
        sub = _linear_terms(other)
        if sub is None:
            return None
        c, terms = sub
        return c * const_val, {k: (s, co * const_val) for k, (s, co) in terms.items()}
    if node.op == "div":
        a, b = node.args
        if b.op != "const" or np.asarray(b.meta["value"]).size != 1:
            return None
        const_val = float(np.asarray(b.meta["value"]).reshape(-1)[0])
        sub = _linear_terms(a)
        if sub is None:
            return None
        c, terms = sub
        return c / const_val, {k: (s, co / const_val) for k, (s, co) in terms.items()}
    return None


def _peel_implicit_linear_stack(node: Node, expected_ch: int):
    """Split a stack of affine channel combinations into ``(parent, M, offset)``, or None."""
    if node.op != "stack" or len(node.args) != expected_ch:
        return None
    parsed = []
    shared_ids = set()
    for child in node.args:
        lt = _linear_terms(child)
        if lt is None:
            return None
        c, terms = lt
        parsed.append((c, terms))
        shared_ids.update(id(s) for s, _ in terms.values())
    if len(shared_ids) > 1:
        return None
    if not shared_ids:
        return None
    shared_obj = None
    for _, terms in parsed:
        for s, _ in terms.values():
            shared_obj = s
            break
        if shared_obj is not None:
            break
    width = _node_width(shared_obj)
    if width is None:
        return None
    matrix = np.zeros((width, expected_ch))
    offset = np.zeros(expected_ch)
    for j, (c, terms) in enumerate(parsed):
        offset[j] = c
        for (_, idx), (_, coeff) in terms.items():
            if idx >= width:
                return None
            matrix[idx, j] += coeff
    return shared_obj, matrix, offset


def _peel_curveset(node: Node):
    """Split ``stack(curve_i(select(parent, idx_i)))`` into ``(parent, curves, idxs)``, or None."""
    if node.op != "stack":
        return None
    shared = None
    curve_fns = []
    idxs = []
    for child in node.args:
        fn, upstream = _extract_single_channel_curve(child)
        if fn is None:
            return None
        idx = upstream.meta.get("index")
        if idx is None:
            return None
        parent = upstream.args[0]
        if shared is None:
            shared = parent
        elif parent is not shared:
            return None
        curve_fns.append(fn)
        idxs.append(idx)
    return shared, curve_fns, idxs


def _node_width(node: Node):
    """Channel count of `node`, or None if unknown."""
    if node.width is not None:
        return node.width
    if node.op == "input":
        return node.meta["n_channels"]
    if node.op == "matmul":
        return node.meta["matrix"].shape[-1]
    if node.op == "stack":
        return len(node.args)
    if node.op == "select":
        return 1
    if node.op == "clip":
        return _node_width(node.args[0])
    return None


def _extract_single_channel_curve(node: Node):
    """``(curve, select_node)`` if `node` reads one channel selection, else ``(None, None)``."""
    select_node = _find_unique_select(node)
    if select_node is None:
        return None, None
    try:
        fn = _compile_scalar_expr(node, select_node)
    except TraceAbort:
        return None, None
    return fn, select_node


def _find_unique_select(node: Node):
    """The only 'select' node reachable from `node`, or None."""
    found = []

    def walk(n: Node):
        if n.op == "select":
            found.append(n)
            return
        if n.op == "const":
            return
        for a in n.args:
            if isinstance(a, Node):
                walk(a)

    walk(node)
    uniq = {id(f): f for f in found}
    if len(uniq) != 1:
        return None
    return next(iter(uniq.values()))


def _compile_scalar_expr(node: Node, select_node: Node):
    """Compile the scalar graph at `node` into ``f(x)`` with `select_node` as ``x``."""
    consts: list = []
    lines: list = []
    names: dict = {}

    def const(value) -> str:
        consts.append(value)
        return f"c[{len(consts) - 1}]"

    def emit(n: Node) -> str:
        if n is select_node:
            return "x"
        cached = names.get(id(n))
        if cached is not None:
            return cached
        op, args = n.op, n.args
        if op == "const":
            value = np.asarray(n.meta["value"], dtype=np.float64)
            if value.size != 1:
                raise TraceAbort("vector constant in a scalar curve")
            return const(float(value.reshape(-1)[0]))
        if op == "clip":
            expr = f"np.clip({emit(args[0])}, {emit(args[1])}, {emit(args[2])})"
        elif op == "pow":
            base, exponent = args
            if (
                exponent.op == "const"
                and float(np.asarray(exponent.meta["value"]).reshape(-1)[0]).is_integer()
            ):
                e = int(np.asarray(exponent.meta["value"]).reshape(-1)[0])
                expr = f"np.power({emit(base)}, {e})"
            else:
                expr = f"np.power(np.clip({emit(base)}, 0, None), {emit(exponent)})"
        elif op in BINARY_FUNCS or op in COMPARISON_FUNCS:
            fn = BINARY_FUNCS.get(op) or COMPARISON_FUNCS[op]
            expr = f"{const(fn)}({emit(args[0])}, {emit(args[1])})"
        elif op in UNARY_FUNCS:
            expr = f"{const(UNARY_FUNCS[op])}({emit(args[0])})"
        elif op == "where":
            expr = f"np.where({emit(args[0])}, {emit(args[1])}, {emit(args[2])})"
        elif op == "interp":
            expr = f"np.interp({emit(args[0])}, {const(n.meta['xp'])}, {const(n.meta['fp'])})"
        else:
            raise TraceAbort(f"op {op} is not a pure elementwise scalar op")
        name = f"t{len(lines)}"
        lines.append(f"    {name} = {expr}")
        names[id(n)] = name
        return name

    result = emit(node)
    source = "def f(x):\n" + "\n".join(lines + [f"    return np.broadcast_to({result}, np.shape(x))"])
    scope: dict = {"np": np, "c": consts}
    exec(compile(source, "<curve>", "exec"), scope)
    return scope["f"]


def _peel_vectorized_curveset(node: Node, expected_ch: int):
    """Split ``f(leaf)`` applied to every channel into ``(leaf, [f] * expected_ch)``, or None."""

    def find_leaf(n: Node):
        if n.op == "const":
            return None
        if n.op not in ELEMENTWISE_OPS:
            return n
        leaves = []
        for a in n.args:
            if isinstance(a, Node):
                leaf = find_leaf(a)
                if leaf is not None:
                    leaves.append(leaf)
        uniq = {id(leaf_): leaf_ for leaf_ in leaves}
        if len(uniq) > 1:
            raise TraceAbort("more than one upstream node feeds this elementwise chain")
        if not uniq:
            return None
        return next(iter(uniq.values()))

    try:
        leaf = find_leaf(node)
    except TraceAbort:
        return None
    if leaf is None or leaf is node or _node_width(leaf) != expected_ch:
        return None
    try:
        fn = _compile_scalar_expr(node, leaf)
    except TraceAbort:
        return None
    return leaf, [fn] * expected_ch


def decompose_chain(root: Node, n_in: int, n_out: int, max_stages: int = 6):
    """Decompose a traced graph into matrix and curve-set stages.



    Returns


    -------


    list or None


        Stages in application order, each ``('matrix', M, offset)`` with


        ``y = x @ M + offset`` or ``('curveset', [f_0, ...])``; None if the


        graph is not such a chain from the `n_in`-channel input.


    """
    stages = []
    cur = root
    if cur.op == "clip":
        cur = cur.args[0]
    expected_ch = n_out

    for _ in range(max_stages):
        peeled = _peel_matrix(cur)
        if peeled is None:
            peeled = _peel_implicit_linear_stack(cur, expected_ch)
        if peeled is not None:
            pre, matrix, offset = peeled
            if matrix.shape[-1] != expected_ch:
                return None
            if stages and stages[-1][0] == "matrix":
                _, outer, outer_off = stages.pop()
                offset = offset @ outer + outer_off
                matrix = matrix @ outer
            stages.append(("matrix", matrix, offset))
            cur = pre
            expected_ch = matrix.shape[0]
            continue
        peeled_c = _peel_curveset(cur)
        if peeled_c is not None:
            shared, curve_fns, idxs = peeled_c
            if len(curve_fns) != expected_ch:
                return None
            if idxs != list(range(len(idxs))):
                width = _node_width(shared)
                if width is None or any(idx >= width for idx in idxs):
                    return None
                perm = np.zeros((width, len(idxs)))
                for j, idx in enumerate(idxs):
                    perm[idx, j] = 1.0
                stages.append(("curveset", curve_fns))
                stages.append(("matrix", perm, np.zeros(len(idxs))))
                cur = shared
                expected_ch = width
                continue
            stages.append(("curveset", curve_fns))
            cur = shared
            continue
        peeled_v = _peel_vectorized_curveset(cur, expected_ch)
        if peeled_v is not None:
            shared, curve_fns = peeled_v
            if len(curve_fns) != expected_ch:
                return None
            stages.append(("curveset", curve_fns))
            cur = shared
            continue
        break

    if cur.op == "input" and cur.meta.get("n_channels") == n_in and expected_ch == n_in:
        stages.reverse()
        return stages

    return None
