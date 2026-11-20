"""
v5 multiProcessElements tags: matrix/curve chains and calculator programs.
"""

from __future__ import annotations

from ..._errors import ICCWriteError
from .._io import mpe
from .._tracer.calc import compile_calc_program
from .._tracer.tracing import TraceAbort
from . import curves as cv
from .context import Context
from .structure import chains, stage_ranges
from .verify import agrees, call


def chain_bytes(chain, ranges) -> list:
    """Encode a matrix/curve chain as mpet element bodies."""
    elems = []
    for stage, rng in zip(chain, ranges, strict=False):
        if stage[0] == "matrix":
            elems.append(mpe.encode_matrix_element(stage[1].T, stage[2]))
        else:
            curves = [cv.v5_curve(fn, lo, hi) for fn, (lo, hi) in zip(stage[1], rng, strict=False)]
            elems.append(mpe.encode_curveset_element(curves))
    return elems


def write_mpet(ctx: Context, tags: dict, notes: list) -> bool:
    """Write A2B0/B2A0 mpet chains of matrix and curve-set elements."""
    fwd_chain, bwd_chain = chains(ctx)
    if fwd_chain is None or bwd_chain is None:
        return False
    fwd_ranges = stage_ranges(fwd_chain, ctx.device_samples)
    bwd_ranges = stage_ranges(bwd_chain, ctx.xyz_finite)
    if fwd_ranges is None or bwd_ranges is None:
        return False
    tags[b"A2B0"] = mpe.encode_mpet(ctx.n, 3, chain_bytes(fwd_chain, fwd_ranges))
    tags[b"B2A0"] = mpe.encode_mpet(3, ctx.n, chain_bytes(bwd_chain, bwd_ranges))
    kinds = " -> ".join(s[0] for s in fwd_chain)
    notes.append(f"{len(fwd_chain)}-stage matrix/curve chain ({kinds}) as mpet elements")
    return True


def write_calc(ctx: Context, tags: dict, notes: list) -> bool:
    """Write A2B0/B2A0 mpet calculator elements."""
    if ctx.fwd_root is None or ctx.bwd_root is None:
        return False
    try:
        a2b = mpe.encode_mpet(
            ctx.n,
            3,
            [mpe.encode_calc_element(ctx.n, 3, compile_calc_program(ctx.fwd_root, ctx.n, 3))],
        )
        b2a = mpe.encode_mpet(
            3,
            ctx.n,
            [mpe.encode_calc_element(3, ctx.n, compile_calc_program(ctx.bwd_root, 3, ctx.n))],
        )
    except (TraceAbort, ICCWriteError):
        return False
    a2b_elem = mpe.decode_mpet(a2b)["elements"][0]
    b2a_elem = mpe.decode_mpet(b2a)["elements"][0]
    fwd_got = mpe.eval_calc_element(a2b_elem, ctx.device_samples, 3)
    bwd_got = mpe.eval_calc_element(b2a_elem, ctx.xyz_finite, ctx.n)
    fwd_ok = agrees(ctx.xyz_samples, fwd_got)
    bwd_ok = ctx.backward_agrees(call(ctx.bwd, ctx.xyz_finite), bwd_got)
    if not (fwd_ok and bwd_ok):
        return False
    tags[b"A2B0"], tags[b"B2A0"] = a2b, b2a
    notes.append("whole transform compiled into calculator elements")
    return True
