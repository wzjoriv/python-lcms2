"""Matrix/curve chain shapes of a space's conversions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._tracer.chain import decompose_chain
from . import curves as cv
from .verify import agrees, call

if TYPE_CHECKING:
    from .context import Context


def identity(t):
    """Identity curve."""
    return np.asarray(t, dtype=np.float64)


def chain_as_curves_matrix(chain, n: int):
    """``(curves, matrix, offset)`` equivalent to a forward chain, or None."""
    if chain is None:
        return None
    stages = list(chain)
    lead = None
    if len(stages) >= 2 and stages[0][0] == "matrix" and stages[1][0] == "curveset":
        lead = stages.pop(0)
    if len(stages) == 1 and stages[0][0] == "matrix":
        stages = [("curveset", [identity] * n)] + stages
    if len(stages) == 1 and stages[0][0] == "curveset" and len(stages[0][1]) == 3:
        stages = stages + [("matrix", np.eye(3), np.zeros(3))]
    if len(stages) != 2 or stages[0][0] != "curveset" or stages[1][0] != "matrix":
        return None
    curves = list(stages[0][1])
    _, matrix, offset = stages[1]
    if lead is None:
        return (curves, matrix, offset) if len(curves) == n else None
    _, pre, pre_off = lead
    pre_off = np.broadcast_to(np.asarray(pre_off, dtype=np.float64).reshape(-1), (pre.shape[1],))
    if pre.shape != (n, len(curves)):
        return None
    sources = []
    for j in range(pre.shape[1]):
        nz = np.flatnonzero(np.abs(pre[:, j]) > 0)
        if nz.size != 1:
            return None
        sources.append(int(nz[0]))
    if sorted(sources) != list(range(n)):
        return None
    new_curves = [None] * n
    new_matrix = np.zeros((n, matrix.shape[1]))
    for j, i in enumerate(sources):
        a, b, f = pre[i, j], pre_off[j], curves[j]
        new_curves[i] = lambda t, f=f, a=a, b=b: f(np.asarray(t, dtype=np.float64) * a + b)
        new_matrix[i] = matrix[j]
    return new_curves, new_matrix, offset


def chain_as_matrix_curves(chain, n: int):
    """``(matrix, offset, curves)`` equivalent to a backward chain, or None."""
    if chain is None:
        return None
    stages = list(chain)
    if len(stages) == 1 and stages[0][0] == "curveset" and n == 3:
        stages = [("matrix", np.eye(3), np.zeros(3))] + stages
    if len(stages) == 1 and stages[0][0] == "matrix":
        stages = stages + [("curveset", [identity] * n)]
    if len(stages) == 3 and stages[2][0] == "matrix":
        _, post, post_off = stages[2]
        if post.shape != (n, n) or not np.allclose(post, np.diag(np.diag(post))):
            return None
        scale = np.diag(post)
        shift = np.broadcast_to(np.asarray(post_off, dtype=np.float64).reshape(-1), (n,))
        curves = [
            lambda u, f=f, a=a, b=b: f(u) * a + b for f, a, b in zip(stages[1][1], scale, shift, strict=False)
        ]
        stages = [stages[0], ("curveset", curves)]
    if len(stages) != 2 or stages[0][0] != "matrix" or stages[1][0] != "curveset":
        return None
    _, matrix, offset = stages[0]
    curves = list(stages[1][1])
    if matrix.shape != (3, n) or len(curves) != n:
        return None
    return matrix, np.asarray(offset, dtype=np.float64).reshape(-1), curves


def stage_ranges(chain, samples: np.ndarray):
    """Input ``(lo, hi)`` per channel of every curveset stage in `chain` over `samples`."""
    cur = samples
    out = []
    for stage in chain:
        if stage[0] == "matrix":
            cur = cur @ stage[1] + stage[2]
            out.append(None)
        else:
            ranges = []
            for i, fn in enumerate(stage[1]):
                col = cur[:, i][np.isfinite(cur[:, i])]
                if col.size == 0:
                    return None
                lo, hi = float(col.min()), float(col.max())
                pad = 0.2 * (hi - lo) if hi > lo else 1.0
                trimmed = cv.finite_range(fn, lo - pad, hi + pad, lo, hi)
                if trimmed is None:
                    return None
                ranges.append(trimmed)
            cur = np.stack([cv.evaluate(fn, cur[:, i]) for i, fn in enumerate(stage[1])], axis=-1)
            out.append(ranges)
    return out


def eval_chain(chain, x: np.ndarray) -> np.ndarray:
    """Apply a matrix/curve chain to ``(k, n)`` samples."""
    cur = x
    for stage in chain:
        if stage[0] == "matrix":
            cur = cur @ stage[1] + stage[2]
        else:
            cur = np.stack([cv.evaluate(fn, cur[:, i]) for i, fn in enumerate(stage[1])], axis=-1)
    return cur


def chains(ctx: Context):
    """Verified forward and backward matrix/curve chains of the space, each possibly None."""
    fwd_chain = decompose_chain(ctx.fwd_root, ctx.n, 3) if ctx.fwd_root is not None else None
    if fwd_chain is not None and not agrees(
        ctx.xyz_samples, eval_chain(fwd_chain, ctx.device_samples)
    ):
        fwd_chain = None
    bwd_chain = decompose_chain(ctx.bwd_root, 3, ctx.n) if ctx.bwd_root is not None else None
    if bwd_chain is not None and not ctx.backward_agrees(
        call(ctx.bwd, ctx.xyz_finite), eval_chain(bwd_chain, ctx.xyz_finite)
    ):
        bwd_chain = None
    fwd_matrix = None
    if fwd_chain is None:
        found = ctx.forward_curves_matrix()
        if found is not None:
            curves, fwd_matrix, offset = found
            fwd_chain = [("curveset", curves), ("matrix", fwd_matrix, offset)]
    if bwd_chain is None:
        hint = None
        if fwd_matrix is not None and ctx.n == 3 and abs(np.linalg.det(fwd_matrix)) > 1e-12:
            hint = np.linalg.inv(fwd_matrix)
        found = ctx.backward_matrix_curves(hint)
        if found is not None:
            matrix, offset, curves, _ = found
            bwd_chain = [("matrix", matrix, offset), ("curveset", curves)]
    return fwd_chain, bwd_chain
