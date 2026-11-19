"""Encoders, decoders and evaluators for the ICC.2 multiProcessElementsType ('mpet').

Elements: matrixElement ('matf'), curveSetElement ('cvst') of segmented
curves ('curf') with formula ('parf') and sampled ('samf') segments,
CLUTElement ('clut') and calculatorElement ('calc').
"""

from __future__ import annotations

import struct

import numpy as np

from .._errors import ICCParseError, ICCWriteError
from . import binary as bn

FLT_MIN = float(np.finfo(np.float32).tiny)


def _position_table(blobs, base: int) -> tuple:
    positions, body, cursor = [], bytearray(), base
    for blob in blobs:
        positions.append((cursor, len(blob)))
        pad = bn.pad4(len(blob))
        body += blob + b"\x00" * pad
        cursor += len(blob) + pad
    return b"".join(struct.pack(">II", o, s) for o, s in positions), bytes(body)


def encode_mpet(in_ch: int, out_ch: int, element_blobs) -> bytes:
    """Encode an 'mpet' tag body from encoded element bodies."""
    header = b"mpet" + b"\x00" * 4 + struct.pack(">HHI", in_ch, out_ch, len(element_blobs))
    table, body = _position_table(element_blobs, len(header) + 8 * len(element_blobs))
    return header + table + body


def decode_mpet(body: bytes) -> dict:
    """Decode an 'mpet' tag body into ``in_ch``, ``out_ch`` and ``elements``."""
    if body[0:4] != b"mpet":
        raise ICCParseError(f"not an mpet tag: {body[0:4]!r}")
    in_ch, out_ch, n = struct.unpack_from(">HHI", body, 8)
    elements = []
    for i in range(n):
        off, size = struct.unpack_from(">II", body, 16 + 8 * i)
        if off + size > len(body):
            raise ICCParseError("mpet element out of bounds")
        elements.append(decode_element(body[off : off + size]))
    return {"in_ch": in_ch, "out_ch": out_ch, "elements": elements}


def decode_element(buf: bytes) -> dict:
    """Decode one processing element by its signature."""
    decoders = {
        b"matf": decode_matrix_element,
        b"cvst": decode_curveset_element,
        b"clut": decode_clut_element,
        b"calc": decode_calc_element,
    }
    decoder = decoders.get(bn.sig(buf, 0))
    if decoder is None:
        raise ICCParseError(f"unsupported processing element {bn.sig(buf, 0)!r}")
    return decoder(buf)


def encode_matrix_element(matrix: np.ndarray, offset: np.ndarray) -> bytes:
    """Encode a matrixElement ('matf') body from an (out x in) matrix and offset."""
    matrix = np.asarray(matrix, dtype=np.float64)
    out_ch, in_ch = matrix.shape
    return (
        b"matf"
        + b"\x00" * 4
        + struct.pack(">HH", in_ch, out_ch)
        + matrix.astype(">f4").tobytes()
        + np.asarray(offset, dtype=np.float64).reshape(out_ch).astype(">f4").tobytes()
    )


def decode_matrix_element(buf: bytes) -> dict:
    """Decode a matrixElement ('matf') body into ``matrix`` (out x in) and ``offset``."""
    in_ch, out_ch = struct.unpack_from(">HH", buf, 8)
    values = np.frombuffer(buf, dtype=">f4", count=in_ch * out_ch + out_ch, offset=12)
    values = values.astype(np.float64)
    return {
        "kind": "matrix",
        "in_ch": in_ch,
        "out_ch": out_ch,
        "matrix": values[: in_ch * out_ch].reshape(out_ch, in_ch),
        "offset": values[in_ch * out_ch :],
    }


FORMULA_PARAM_COUNTS = {0: 4, 1: 5, 2: 5, 3: 5, 4: 5, 5: 6, 6: 7, 7: 6}


def _fpow(base, g):
    base = np.asarray(base, dtype=np.float64)
    if float(g) != np.floor(g):
        base = np.where(base < 0, 0.0, base)
    with np.errstate(all="ignore"):
        return np.power(base, g)


def _clip_pow(base, g):
    base = np.asarray(base, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(base <= 0, 0.0, np.power(np.maximum(base, 0.0), g))


def _log_arg(arg):
    return np.where(arg <= 0, FLT_MIN, arg)


def eval_formula(func_type: int, p, x: np.ndarray) -> np.ndarray:
    """Evaluate an ICC.2 formula curve segment of type 0-7 at `x`."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        if func_type == 0:
            return _fpow(p[1] * x + p[2], p[0]) + p[3]
        if func_type == 1:
            return p[1] * np.log10(_log_arg(p[2] * _fpow(x, p[0]) + p[3])) + p[4]
        if func_type == 2:
            return p[0] * _fpow(np.full_like(x, p[1]), p[2] * x + p[3]) + p[4]
        if func_type == 3:
            return p[1] * _clip_pow(p[2] * x + p[3], p[0]) + p[4]
        if func_type == 4:
            return p[1] * np.log(_log_arg(p[4] * _fpow(x, p[0]) - p[2])) + p[3]
        if func_type == 5:
            if p[1] == 0:
                return np.full_like(x, p[2])
            return p[5] * np.exp((p[4] * _fpow(x, p[0]) - p[3]) / p[1]) + p[2]
        if func_type == 6:
            xg = _fpow(x, p[1])
            denom = p[3] - p[4] * xg
            safe = np.where(denom == 0, 1.0, denom)
            val = p[5] * _fpow(np.maximum(p[6] * xg - p[2], 0.0) / safe, p[0])
            return np.where(denom == 0, 0.0, val)
        if func_type == 7:
            xg = _fpow(x, p[1])
            denom = 1.0 + p[4] * xg
            safe = np.where(denom == 0, 1.0, denom)
            val = p[5] * _fpow((p[2] + p[3] * xg) / safe, p[0])
            return np.where(denom == 0, 0.0, val)
    raise ICCParseError(f"unknown formula segment type {func_type}")


def constant_segment(value: float):
    """Formula segment ``('formula', 0, params)`` equal to `value`."""
    return ("formula", 0, [1.0, 0.0, 0.0, float(value)])


def _encode_segment(seg) -> bytes:
    if seg[0] == "formula":
        _, func_type, params = seg
        if len(params) != FORMULA_PARAM_COUNTS.get(func_type, -1):
            raise ICCWriteError(f"formula type {func_type} takes a different parameter count")
        return (
            b"parf"
            + b"\x00" * 4
            + struct.pack(">HH", func_type, 0)
            + np.asarray(params, dtype=">f4").tobytes()
        )
    if seg[0] == "sampled":
        samples = np.asarray(seg[1], dtype=np.float64)
        return (
            b"samf"
            + b"\x00" * 4
            + struct.pack(">I", samples.size)
            + samples.astype(">f4").tobytes()
        )
    raise ICCWriteError(f"unknown curve segment kind {seg[0]}")


def encode_segmented_curve(breakpoints, segments) -> bytes:
    """Encode a segmentedCurve ('curf').

    Parameters
    ----------
    breakpoints : sequence of float
        ``len(segments) - 1`` increasing breakpoints; segment ``i`` covers
        ``(breakpoints[i - 1], breakpoints[i]]``.
    segments : list
        ``('formula', func_type, params)`` or ``('sampled', values)``. A
        sampled segment lists the values after its start point, which is
        taken from the previous segment, and cannot be first.

    Returns
    -------
    bytes
    """
    if len(breakpoints) != len(segments) - 1:
        raise ICCWriteError("a segmented curve needs one fewer breakpoint than segments")
    if not segments or segments[0][0] == "sampled":
        raise ICCWriteError("a segmented curve must start with a formula segment")
    return (
        b"curf"
        + b"\x00" * 4
        + struct.pack(">HH", len(segments), 0)
        + np.asarray(breakpoints, dtype=">f4").tobytes()
        + b"".join(_encode_segment(s) for s in segments)
    )


def encode_curveset_element(curves) -> bytes:
    """Encode a curveSetElement ('cvst') from encoded segmented curves, one per channel."""
    n = len(curves)
    header = b"cvst" + b"\x00" * 4 + struct.pack(">HH", n, n)
    table, body = _position_table(curves, len(header) + 8 * n)
    return header + table + body


def _decode_segmented_curve(buf: bytes) -> dict:
    if buf[0:4] != b"curf":
        raise ICCParseError(f"unsupported curve type {buf[0:4]!r}")
    (n,) = struct.unpack_from(">H", buf, 8)
    if n < 1:
        raise ICCParseError("segmented curve without segments")
    pos = 12
    breakpoints = np.frombuffer(buf, dtype=">f4", count=n - 1, offset=pos).astype(np.float64)
    pos += 4 * (n - 1)
    segments = []
    for _ in range(n):
        sig = buf[pos : pos + 4]
        if sig == b"parf":
            (func_type,) = struct.unpack_from(">H", buf, pos + 8)
            count = FORMULA_PARAM_COUNTS.get(func_type)
            if count is None:
                raise ICCParseError(f"unknown formula segment type {func_type}")
            params = np.frombuffer(buf, dtype=">f4", count=count, offset=pos + 12)
            segments.append(("formula", func_type, params.astype(np.float64).tolist()))
            pos += 12 + 4 * count
        elif sig == b"samf":
            (count,) = struct.unpack_from(">I", buf, pos + 8)
            values = np.frombuffer(buf, dtype=">f4", count=count, offset=pos + 12)
            segments.append(("sampled", values.astype(np.float64)))
            pos += 12 + 4 * count
        else:
            raise ICCParseError(f"unsupported curve segment {sig!r}")
    if segments[0][0] == "sampled":
        raise ICCParseError("segmented curve starts with a sampled segment")
    edges = np.concatenate([[-np.inf], breakpoints, [np.inf]])
    resolved: list = []
    for i, seg in enumerate(segments):
        if seg[0] == "sampled":
            start, end = edges[i], edges[i + 1]
            if seg[1].size < 1 or not np.isfinite(end - start) or end <= start:
                raise ICCParseError("sampled segment needs values and a finite range")
            first = _eval_segment(resolved[-1], np.array([start]))[0]
            seg = ("sampled", np.concatenate([[first], seg[1]]), start, end)
        resolved.append(seg)
    return {"breakpoints": breakpoints, "segments": resolved}


def _eval_segment(seg, x: np.ndarray) -> np.ndarray:
    if seg[0] == "formula":
        return eval_formula(seg[1], seg[2], x)
    _, samples, lo, hi = seg
    last = samples.size - 1
    xc = np.clip(np.where(np.isfinite(x), x, lo), lo, hi)
    pos = np.clip((xc - lo) / (hi - lo) * last, 0, last)
    i0 = np.minimum(np.floor(pos).astype(np.int64), last - 1)
    frac = pos - i0
    return samples[i0] * (1 - frac) + samples[i0 + 1] * frac


def decode_curveset_element(buf: bytes) -> dict:
    """Decode a curveSetElement ('cvst') body into per-channel curves."""
    in_ch, out_ch = struct.unpack_from(">HH", buf, 8)
    if in_ch != out_ch:
        raise ICCParseError("curve set with different input and output counts")
    curves = []
    for i in range(in_ch):
        off, size = struct.unpack_from(">II", buf, 12 + 8 * i)
        if off + size > len(buf):
            raise ICCParseError("curve out of bounds")
        curves.append(_decode_segmented_curve(buf[off : off + size]))
    return {"kind": "curveset", "in_ch": in_ch, "out_ch": out_ch, "curves": curves}


def eval_curveset_channel(curve: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate one decoded segmented curve at `x`."""
    x = np.asarray(x, dtype=np.float64)
    segs = curve["segments"]
    if len(segs) == 1:
        return _eval_segment(segs[0], x)
    idx = np.searchsorted(curve["breakpoints"], x, side="left")
    idx = np.where(np.isnan(x), len(segs) - 1, idx)
    out = np.empty_like(x)
    for i, seg in enumerate(segs):
        mask = idx == i
        if np.any(mask):
            out[mask] = _eval_segment(seg, x[mask])
    return out


def encode_clut_element(clut: np.ndarray) -> bytes:
    """Encode a CLUTElement ('clut') whose inputs span [0, 1].

    Parameters
    ----------
    clut : ndarray, shape grid_points + (out_ch,)
        Table samples; the first input varies slowest.

    Returns
    -------
    bytes
    """
    grid = clut.shape[:-1]
    grid_bytes = bytearray(16)
    for i, g in enumerate(grid):
        grid_bytes[i] = g
    return (
        b"clut"
        + b"\x00" * 4
        + struct.pack(">HH", len(grid), clut.shape[-1])
        + bytes(grid_bytes)
        + np.asarray(clut, dtype=">f4").tobytes()
    )


def decode_clut_element(buf: bytes) -> dict:
    """Decode a CLUTElement ('clut') body into its grid and samples."""
    in_ch, out_ch = struct.unpack_from(">HH", buf, 8)
    grid = list(buf[12 : 12 + in_ch])
    if in_ch < 1 or in_ch > 16 or any(g < 2 for g in grid):
        raise ICCParseError("CLUT element with an invalid grid")
    count = int(np.prod(grid)) * out_ch
    values = np.frombuffer(buf, dtype=">f4", count=count, offset=28).astype(np.float64)
    return {
        "kind": "clut",
        "in_ch": in_ch,
        "out_ch": out_ch,
        "grid_points": grid,
        "clut": values.reshape(tuple(grid) + (out_ch,)),
    }


_FLOAT_OPS = frozenset({"data"})
_SIZE_OPS = frozenset({"if  ", "else"})


def encode_calc_element(in_ch: int, out_ch: int, ops) -> bytes:
    """Encode a calculatorElement ('calc') with one main function.

    Parameters
    ----------
    in_ch, out_ch : int
        Channel counts.
    ops : list of tuple
        ``(sig, arg)`` with a 4-character `sig`; `arg` is a float for
        ``'data'``, an operation count for ``'if  '``/``'else'``, and a
        ``(v1, v2)`` selector otherwise.

    Returns
    -------
    bytes
    """
    words = bytearray()
    for sig, arg in ops:
        if len(sig) != 4:
            raise ICCWriteError(f"calculator operation {sig!r} is not a 4-character signature")
        words += sig.encode("ascii")
        if sig in _FLOAT_OPS:
            words += struct.pack(">f", float(arg))
        elif sig in _SIZE_OPS:
            words += struct.pack(">I", int(arg))
        else:
            v1, v2 = arg
            words += struct.pack(">HH", int(v1), int(v2))
    func = b"func" + b"\x00" * 4 + struct.pack(">I", len(ops)) + bytes(words)
    header = b"calc" + b"\x00" * 4 + struct.pack(">HHI", in_ch, out_ch, 0)
    table, body = _position_table([func], len(header) + 8)
    return header + table + body


def decode_calc_element(buf: bytes) -> dict:
    """Decode a calculatorElement ('calc') body into its main function's operations."""
    in_ch, out_ch, n_sub = struct.unpack_from(">HHI", buf, 8)
    if n_sub:
        raise ICCParseError("calculator sub-elements are not supported")
    off, size = struct.unpack_from(">II", buf, 16)
    func = buf[off : off + size]
    if func[0:4] != b"func":
        raise ICCParseError("calculator main function missing")
    (n_ops,) = struct.unpack_from(">I", func, 8)
    if 12 + 8 * n_ops > len(func):
        raise ICCParseError("calculator function truncated")
    ops = []
    for i in range(n_ops):
        sig = func[12 + 8 * i : 16 + 8 * i].decode("latin-1")
        if sig in _FLOAT_OPS:
            (arg,) = struct.unpack_from(">f", func, 16 + 8 * i)
        elif sig in _SIZE_OPS:
            (arg,) = struct.unpack_from(">I", func, 16 + 8 * i)
        else:
            arg = struct.unpack_from(">HH", func, 16 + 8 * i)
        ops.append((sig, arg))
    return {"kind": "calc", "in_ch": in_ch, "out_ch": out_ch, "ops": ops}


def _round_half_away(a):
    return np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))


_UNARY = {
    "neg ": np.negative,
    "sq  ": np.square,
    "sqrt": np.sqrt,
    "cb  ": lambda a: a * a * a,
    "cbrt": np.cbrt,
    "sign": np.sign,
    "abs ": np.abs,
    "flor": np.floor,
    "ceil": np.ceil,
    "trnc": np.trunc,
    "rond": _round_half_away,
    "exp ": np.exp,
    "log ": np.log10,
    "ln  ": np.log,
    "sin ": np.sin,
    "cos ": np.cos,
    "tan ": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "not ": lambda a: (a < 0.5).astype(np.float64),
}


def _fmod(a, b):
    bad = ~np.isfinite(a) | ~np.isfinite(b) | (np.abs(b) < 1e-12)
    return np.where(bad, 0.0, np.fmod(a, np.where(bad, 1.0, b)))


def _divide(a, b):
    return np.where(b != 0, a / np.where(b != 0, b, 1.0), 0.0)


_BINARY = {
    "add ": np.add,
    "sub ": np.subtract,
    "mul ": np.multiply,
    "div ": _divide,
    "mod ": _fmod,
    "pow ": np.power,
    "atn2": lambda a, b: np.arctan2(b, a),
    "lt  ": lambda a, b: (a < b).astype(np.float64),
    "le  ": lambda a, b: (a <= b).astype(np.float64),
    "eq  ": lambda a, b: (a == b).astype(np.float64),
    "neq ": lambda a, b: (a != b).astype(np.float64),
    "ge  ": lambda a, b: (a >= b).astype(np.float64),
    "gt  ": lambda a, b: (a > b).astype(np.float64),
    "vmin": np.minimum,
    "vmax": np.maximum,
    "vand": lambda a, b: ((a >= 0.5) & (b >= 0.5)).astype(np.float64),
    "vor ": lambda a, b: ((a >= 0.5) | (b >= 0.5)).astype(np.float64),
}

_VARIADIC = {
    "sum ": lambda vals: np.sum(vals, axis=0),
    "prod": lambda vals: np.prod(vals, axis=0),
    "min ": lambda vals: np.min(vals, axis=0),
    "max ": lambda vals: np.max(vals, axis=0),
    "and ": lambda vals: np.all(np.stack(vals) >= 0.5, axis=0).astype(np.float64),
    "or  ": lambda vals: np.any(np.stack(vals) >= 0.5, axis=0).astype(np.float64),
}

_CONSTANTS = {"pi  ": np.pi, "+INF": np.inf, "-INF": -np.inf, "NaN ": np.nan}


def _stack_effect(sig: str, arg):
    if sig in _FLOAT_OPS or sig in _CONSTANTS:
        return 0, 1
    if sig in _SIZE_OPS:
        return None
    v1, v2 = arg
    if sig in ("in  ", "tget"):
        return 0, v2 + 1
    if sig in ("out ", "tput"):
        return v2 + 1, 0
    if sig == "tsav":
        return v2 + 1, v2 + 1
    if sig in _UNARY:
        return v1 + 1, v1 + 1
    if sig in _BINARY:
        return 2 * (v1 + 1), v1 + 1
    if sig in _VARIADIC:
        return v1 + 2, 1
    if sig == "copy":
        return v1 + 1, (v1 + 1) * (v2 + 2)
    if sig == "pop ":
        return v1 + 1, 0
    if sig in ("rotl", "rotr"):
        return v1 + 1, v1 + 1
    if sig == "flip":
        return v1 + 2, v1 + 2
    return None


def _branch_layout(ops, i):
    has_else = i + 1 < len(ops) and ops[i + 1][0] == "else"
    then_n = int(ops[i][1])
    else_n = int(ops[i + 1][1]) if has_else else 0
    start = i + (2 if has_else else 1)
    return start, then_n, else_n, has_else


def _check_ops(ops, in_ch, out_ch, written, depth):
    i = 0
    while i < len(ops):
        sig, arg = ops[i]
        if sig == "if  ":
            if depth < 1:
                raise ICCParseError("calculator stack underflow")
            depth -= 1
            start, then_n, else_n, has_else = _branch_layout(ops, i)
            if start + then_n + else_n > len(ops):
                raise ICCParseError("calculator branch runs past its function")
            then_depth = _check_ops(ops[start : start + then_n], in_ch, out_ch, written, depth)
            else_ops = ops[start + then_n : start + then_n + else_n]
            else_depth = _check_ops(else_ops, in_ch, out_ch, written, depth) if has_else else depth
            if then_depth != else_depth:
                raise ICCParseError("calculator branches leave different stack depths")
            depth = then_depth
            i = start + then_n + else_n
            continue
        effect = _stack_effect(sig, arg)
        if effect is None:
            raise ICCParseError(f"unsupported calculator operation {sig!r}")
        needed, pushed = effect
        if sig == "in  " and arg[0] + arg[1] >= in_ch:
            raise ICCParseError(f"calculator reads input {arg[0] + arg[1]} of {in_ch}")
        if sig == "out ":
            if arg[0] + arg[1] >= out_ch:
                raise ICCParseError(f"calculator writes output {arg[0] + arg[1]} of {out_ch}")
            written.update(range(arg[0], arg[0] + arg[1] + 1))
        if depth < needed:
            raise ICCParseError("calculator stack underflow")
        depth += pushed - needed
        i += 1
    return depth


def validate_calc_element(decoded: dict, in_ch: int, out_ch: int) -> None:
    """Check a calculator element's channels, operations and stack use.

    Raises
    ------
    ICCParseError
        If the program cannot run on `in_ch` inputs to fill `out_ch` outputs.
    """
    if decoded["in_ch"] != in_ch or decoded["out_ch"] != out_ch:
        raise ICCParseError("calculator channel counts do not match its position")
    written: set = set()
    _check_ops(decoded["ops"], in_ch, out_ch, written, 0)
    if written != set(range(out_ch)):
        raise ICCParseError("calculator program does not write every output")


def validate_mpet(decoded: dict) -> None:
    """Check that an mpet's element channel counts chain from its input to its output.

    Raises
    ------
    ICCParseError
        If an element's input count differs from the previous element's output count.
    """
    width = decoded["in_ch"]
    for elem in decoded["elements"]:
        if elem["in_ch"] != width:
            raise ICCParseError(f"mpet element expects {elem['in_ch']} channels, gets {width}")
        if elem["kind"] == "calc":
            validate_calc_element(elem, elem["in_ch"], elem["out_ch"])
        width = elem["out_ch"]
    if width != decoded["out_ch"]:
        raise ICCParseError(f"mpet produces {width} channels, declares {decoded['out_ch']}")


def _merge(cond, then_state, else_state):
    then_stack, then_temps, then_out = then_state
    else_stack, else_temps, else_out = else_state
    stack = [np.where(cond, a, b) for a, b in zip(then_stack, else_stack, strict=False)]

    def merge_dict(a, b):
        out = dict(b)
        for k, v in a.items():
            out[k] = np.where(cond, v, b[k]) if k in b else v
        return out

    return stack, merge_dict(then_temps, else_temps), merge_dict(then_out, else_out)


def _run(ops, x, state):
    stack, temps, outputs = state
    i = 0
    while i < len(ops):
        sig, arg = ops[i]
        if sig == "if  ":
            cond = stack.pop() >= 0.5
            start, then_n, else_n, _ = _branch_layout(ops, i)
            then_ops = ops[start : start + then_n]
            else_ops = ops[start + then_n : start + then_n + else_n]
            then_state = _run(then_ops, x, (list(stack), dict(temps), dict(outputs)))
            else_state = _run(else_ops, x, (list(stack), dict(temps), dict(outputs)))
            stack, temps, outputs = _merge(cond, then_state, else_state)
            i = start + then_n + else_n
            continue
        if sig == "data":
            stack.append(np.full(x.shape[:-1], float(np.float32(arg))))
        elif sig in _CONSTANTS:
            stack.append(np.full(x.shape[:-1], _CONSTANTS[sig]))
        else:
            v1, v2 = arg
            if sig == "in  ":
                stack.extend(x[..., v1 + k] for k in range(v2 + 1))
            elif sig == "out ":
                vals = stack[len(stack) - (v2 + 1) :]
                del stack[len(stack) - (v2 + 1) :]
                for k, v in enumerate(vals):
                    outputs[v1 + k] = v
            elif sig == "tget":
                stack.extend(temps[v1 + k] for k in range(v2 + 1))
            elif sig in ("tput", "tsav"):
                vals = stack[len(stack) - (v2 + 1) :]
                if sig == "tput":
                    del stack[len(stack) - (v2 + 1) :]
                for k, v in enumerate(vals):
                    temps[v1 + k] = v
            elif sig in _UNARY:
                n = v1 + 1
                stack[-n:] = [_UNARY[sig](a) for a in stack[-n:]]
            elif sig in _BINARY:
                n = v1 + 1
                a, b = stack[-2 * n : -n], stack[-n:]
                del stack[-2 * n :]
                stack.extend(_BINARY[sig](p, q) for p, q in zip(a, b, strict=False))
            elif sig in _VARIADIC:
                n = v1 + 2
                vals = stack[-n:]
                del stack[-n:]
                stack.append(_VARIADIC[sig](vals))
            elif sig == "copy":
                stack.extend(stack[-(v1 + 1) :] * (v2 + 1))
            elif sig == "pop ":
                del stack[-(v1 + 1) :]
            elif sig in ("rotl", "rotr"):
                n = v1 + 1
                shift = (v2 + 1) % n
                vals = stack[-n:]
                stack[-n:] = (
                    vals[shift:] + vals[:shift] if sig == "rotl" else vals[-shift:] + vals[:-shift]
                )
            elif sig == "flip":
                n = v1 + 2
                stack[-n:] = stack[-n:][::-1]
            else:
                raise ICCParseError(f"unsupported calculator operation {sig!r}")
        i += 1
    return stack, temps, outputs


def eval_calc_element(decoded: dict, x: np.ndarray, out_ch: int) -> np.ndarray:
    """Run a decoded calculator element on ``(..., in_ch)`` inputs.

    Raises
    ------
    ICCParseError
        If the program uses an unsupported operation or misses an output.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        _, _, outputs = _run(decoded["ops"], x, ([], {}, {}))
    if set(outputs) != set(range(out_ch)):
        raise ICCParseError("calculator program does not write every output")
    return np.stack([np.broadcast_to(outputs[i], x.shape[:-1]) for i in range(out_ch)], axis=-1)
