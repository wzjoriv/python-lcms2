"""
v2/v4 lookup-table tags that hold matrix/curve chains exactly.
"""

from __future__ import annotations

import numpy as np

from .._io import tagtypes as tt
from .._io.constants import CS_XYZ, TYPE_MAB, TYPE_MBA
from .._reader import XYZ_U1F15_SCALE, LutStage
from . import curves as cv
from . import legacy_lut
from .context import Context
from .sampling import grid_points
from .structure import chain_as_matrix_curves, chains
from .verify import agrees, call


def encode_lut_tag(ctx: Context, is_b2a: bool, in_curves, clut, out_curves) -> bytes:
    """Encode a v2 lut16 or v4 mAB/mBA tag from input curves, a CLUT and output curves."""
    in_ch = 3 if is_b2a else ctx.n
    out_ch = ctx.n if is_b2a else 3
    if ctx.major == 2:

        def table(curves, count):
            if curves is None:
                return np.tile(np.array([0.0, 1.0]), (count, 1))
            return np.array([cv.evaluate(c, np.linspace(0, 1, cv.TRC_SAMPLES)) for c in curves])

        return tt.encode_lut16(
            in_ch,
            out_ch,
            clut.shape[0],
            np.eye(3),
            np.clip(table(in_curves, in_ch), 0.0, 1.0),
            clut,
            np.clip(table(out_curves, out_ch), 0.0, 1.0),
        )

    def descriptors(curves, count):
        if curves is None:
            return [cv.identity_curve() for _ in range(count)]
        return [cv.sampled_curve(c) for c in curves]

    if is_b2a:
        return tt.encode_lutAtoB_or_BtoA(
            TYPE_MBA,
            in_ch,
            out_ch,
            B_curves=descriptors(in_curves, in_ch),
            clut=clut,
            A_curves=descriptors(out_curves, out_ch),
        )
    return tt.encode_lutAtoB_or_BtoA(
        TYPE_MAB,
        in_ch,
        out_ch,
        A_curves=descriptors(in_curves, in_ch),
        clut=clut,
        B_curves=descriptors(out_curves, out_ch),
    )


_LUT_RTOL = 2e-3

_LUT_MAX_RTOL = 5e-2


def write_exact_lut(ctx: Context, tags: dict, notes: list) -> bool:
    """Write A2B0/B2A0 lookup tables that hold the space's matrix/curve chains exactly."""
    fwd_chain, bwd_chain = chains(ctx)
    if fwd_chain is None or bwd_chain is None:
        return False
    a2b = legacy_lut.encode_chain(fwd_chain, False, ctx.n, ctx.major, ctx.device_samples)
    if a2b is None:
        return False
    b2a = None
    found = chain_as_matrix_curves(bwd_chain, ctx.n) if len(bwd_chain) == 2 else None
    if found is not None and ctx.n <= 3:
        matrix, offset, curves = found
        u = ctx.xyz_finite @ matrix + offset
        ranges = [(float(u[:, j].min()), float(u[:, j].max())) for j in range(ctx.n)]
        warped = _matrix_b2a if ctx.major >= 4 else _lut16_matrix_b2a
        b2a = warped(ctx, matrix, offset, curves, ranges)
    if b2a is None:
        b2a = legacy_lut.encode_chain(bwd_chain, True, ctx.n, ctx.major, ctx.xyz_finite)
    if b2a is None:
        return False

    a2b_stage = LutStage(a2b, CS_XYZ, False, frozenset())
    b2a_stage = LutStage(b2a, CS_XYZ, True, frozenset())
    encodable = np.all((ctx.xyz_samples >= 0) & (ctx.xyz_samples <= XYZ_U1F15_SCALE), axis=-1)
    if encodable.sum() < 16:
        return False
    device, xyz = ctx.device_samples[encodable], ctx.xyz_samples[encodable]
    fwd_ok = agrees(xyz, a2b_stage.decode_pcs(a2b_stage(device)), None, _LUT_RTOL, _LUT_MAX_RTOL)
    got = b2a_stage(b2a_stage.encode_pcs(xyz))
    expected = np.clip(call(ctx.bwd, xyz), 0.0, 1.0)
    bwd_ok = agrees(expected, got, ctx.periods, _LUT_RTOL, _LUT_MAX_RTOL) or agrees(
        call(ctx.fwd, expected), call(ctx.fwd, got), None, _LUT_RTOL, _LUT_MAX_RTOL
    )
    if not (fwd_ok and bwd_ok):
        return False
    tags[b"A2B0"], tags[b"B2A0"] = a2b, b2a
    kinds = " -> ".join(stage[0] for stage in fwd_chain)
    notes.append(f"matrix/curve chain ({kinds}) written as exact A2B0/B2A0 lookup tables")
    return True


_WARPS = (
    (lambda v: v, lambda s: s),
    (lambda v: np.cbrt(v), lambda s: np.asarray(s) ** 3),
    (lambda v: 1.0 - np.cbrt(1.0 - v), lambda s: 1.0 - (1.0 - np.asarray(s)) ** 3),
)


def _best_warp(g):
    """The warp whose sampled M/A curve pair reproduces `g` on [0, 1] most closely."""
    grid = np.linspace(0.0, 1.0, cv.TRC_SAMPLES)
    test = np.unique(np.concatenate([np.linspace(0, 1, 20001), np.geomspace(1e-9, 1, 2000)]))
    test = np.unique(np.concatenate([test, 1.0 - test]))
    exact = g(test)
    best, best_err = _WARPS[0], np.inf
    for warp, unwarp in _WARPS:
        m = np.interp(test, grid, warp(grid))
        approx = np.interp(m, grid, g(unwarp(grid)))
        err = np.abs(approx - exact)
        err = float(np.max(err[np.isfinite(err)])) if np.isfinite(err).any() else np.inf
        if err < best_err:
            best, best_err = (warp, unwarp), err
    return best


def _matrix_b2a(ctx: Context, w, b, h_curves, ranges):
    matrix = np.zeros((3, 3))
    offset = np.zeros(3)
    m_curves, a_curves = [], []
    for j, (h, (lo, hi)) in enumerate(zip(h_curves, ranges, strict=False)):
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        trimmed = cv.finite_range(h, lo - pad, hi + pad, lo, hi)
        if trimmed is None:
            return None
        lo, hi = trimmed
        span = hi - lo if hi > lo else 1.0
        matrix[j] = w[:, j] * XYZ_U1F15_SCALE / span
        offset[j] = (b[j] - lo) / span

        def g(v, h=h, lo=lo, span=span):
            return np.nan_to_num(cv.evaluate(h, lo + np.asarray(v) * span))

        warp, unwarp = _best_warp(g)
        m_curves.append(cv.sampled_curve(warp))
        a_curves.append(cv.sampled_curve(lambda s, g=g, unwarp=unwarp: g(unwarp(s))))
    for _ in range(ctx.n, 3):
        m_curves.append(cv.identity_curve())
    clut = grid_points([(0.0, 1.0)] * 3, 2)[..., : ctx.n]
    return tt.encode_lutAtoB_or_BtoA(
        TYPE_MBA,
        3,
        ctx.n,
        B_curves=[cv.identity_curve() for _ in range(3)],
        matrix=matrix,
        matrix_offset=offset,
        M_curves=m_curves,
        clut=clut,
        A_curves=a_curves,
    )


def _lut16_matrix_b2a(ctx: Context, w, b, h_curves, ranges):
    """lut16 B2A: scaled projection matrix, warping input tables, selecting CLUT, curve outputs."""
    if np.any(w < -1e-12):
        return None
    matrix = np.zeros((3, 3))
    grid = np.linspace(0.0, 1.0, cv.TRC_SAMPLES)
    in_tables = np.tile(grid, (3, 1))
    out_tables = []
    for j, (h, (lo, hi)) in enumerate(zip(h_curves, ranges, strict=False)):
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        trimmed = cv.finite_range(h, lo - pad, hi + pad, lo, hi)
        if trimmed is None:
            return None
        lo, hi = trimmed
        if np.isfinite(cv.evaluate(h, np.array([b[j]]))[0]):
            lo = float(b[j])
        span = hi - lo if hi > lo else 1.0
        reach = hi - b[j]
        if reach <= 0:
            return None
        matrix[j] = w[:, j] * XYZ_U1F15_SCALE / reach

        def g(v, h=h, lo=lo, span=span):
            return np.nan_to_num(cv.evaluate(h, lo + np.asarray(v) * span))

        warp, unwarp = _best_warp(g)
        normalized = np.clip((grid * reach + b[j] - lo) / span, 0.0, 1.0)
        in_tables[j] = warp(normalized)
        out_tables.append(g(unwarp(grid)))
    clut = grid_points([(0.0, 1.0)] * 3, 2)[..., : ctx.n]
    return tt.encode_lut16(
        3,
        ctx.n,
        2,
        matrix,
        np.clip(in_tables, 0.0, 1.0),
        clut,
        np.clip(np.array(out_tables), 0.0, 1.0),
    )
