"""
Segmented curves ('curf') and curve-set elements ('cvst').
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ...._errors import ICCParseError, ICCWriteError
from .elements import _position_table

FLT_MIN = float(np.finfo(np.float32).tiny)


FORMULA_PARAM_COUNTS = {0: 4, 1: 5, 2: 5, 3: 5, 4: 5, 5: 6, 6: 7, 7: 6}


def _fpow(base: ArrayLike, g: ArrayLike) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    if float(g) != np.floor(g):
        base = np.where(base < 0, 0.0, base)
    with np.errstate(all="ignore"):
        return np.power(base, g)


def _clip_pow(base: ArrayLike, g: ArrayLike) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(base <= 0, 0.0, np.power(np.maximum(base, 0.0), g))


def _log_arg(arg: np.ndarray) -> np.ndarray:
    return np.where(arg <= 0, FLT_MIN, arg)


def eval_formula(func_type: int, p: Sequence[float], x: np.ndarray) -> np.ndarray:
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


def constant_segment(value: float) -> tuple[str, int, list[float]]:
    """Formula segment ``('formula', 0, params)`` equal to `value`."""
    return ("formula", 0, [1.0, 0.0, 0.0, float(value)])


def _encode_segment(seg: tuple[Any, ...]) -> bytes:
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
        return b"samf" + b"\x00" * 4 + struct.pack(">I", samples.size) + samples.astype(">f4").tobytes()
    raise ICCWriteError(f"unknown curve segment kind {seg[0]}")


def encode_segmented_curve(breakpoints: Sequence[float], segments: Sequence[tuple[Any, ...]]) -> bytes:
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


def encode_curveset_element(curves: Sequence[bytes]) -> bytes:
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


def _eval_segment(seg: tuple[Any, ...], x: np.ndarray) -> np.ndarray:
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
