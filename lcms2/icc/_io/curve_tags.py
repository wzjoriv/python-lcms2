"""
Tag types holding curves: curveType and parametricCurveType.
"""

from __future__ import annotations

__all__ = [
    "decode_curv",
    "encode_curv",
    "decode_para",
    "encode_para",
    "eval_para",
    "invert_para",
    "eval_curve",
    "invert_curve_numeric",
]

import struct
from collections.abc import Sequence

import numpy as np

from ..._errors import ICCParseError
from . import binary as bn

CurveDescriptor = tuple[str, np.ndarray | tuple[int, Sequence[float]]]


def decode_curv(body: bytes) -> np.ndarray:
    """Decode a curveType body: samples in [0, 1], ``[]`` for identity, or ``[gamma]``."""
    count = bn.read_u32(body, 8)
    if count == 0:
        return np.array([], dtype=np.float64)
    if count == 1:
        gamma = bn.read_u8Fixed8(body, 12)
        return np.array([gamma], dtype=np.float64)
    vals = np.frombuffer(body, dtype=">u2", count=count, offset=12).astype(np.float64)
    return vals / 65535.0


def encode_curv(samples: np.ndarray) -> bytes:
    """Encode samples in [0, 1] (``[]`` for identity, ``[gamma]`` for a power) as curveType."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return b"curv" + b"\x00" * 4 + struct.pack(">I", 0)
    if samples.size == 1:
        return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + bn.write_u8Fixed8(samples[0])
    q = np.clip(np.round(samples * 65535.0), 0, 65535).astype(">u2")
    return b"curv" + b"\x00" * 4 + struct.pack(">I", q.size) + q.tobytes()


def decode_para(body: bytes) -> tuple[int, list[float]]:
    """Decode a parametricCurveType tag body into (func_type, params)."""
    func_type = bn.read_u16(body, 8)
    nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(func_type)
    if nparams is None:
        raise ICCParseError(f"unknown parametricCurveType function type {func_type}")
    params = [bn.read_s15Fixed16(body, 12 + 4 * i) for i in range(nparams)]
    return func_type, params


def encode_para(func_type: int, params: Sequence[float]) -> bytes:
    """Encode (func_type, params) as a parametricCurveType tag body."""
    out = b"para" + b"\x00" * 4 + struct.pack(">H", func_type) + b"\x00\x00"
    for p in params:
        out += bn.write_s15Fixed16(p)
    return out


def eval_para(func_type: int, params: Sequence[float], x: np.ndarray) -> np.ndarray:
    """Evaluate a parametricCurveType function (ICC.1:2010 Table 65) at `x`."""
    x = np.asarray(x, dtype=np.float64)
    if func_type == 0:
        (g,) = params
        return np.power(np.clip(x, 0, None), g)
    if func_type == 1:
        g, a, b = params
        thr = -b / a if a != 0 else 0.0
        pos = np.power(np.clip(a * x + b, 0, None), g)
        return np.where(x >= thr, pos, 0.0)
    if func_type == 2:
        g, a, b, c = params
        thr = -b / a if a != 0 else 0.0
        pos = np.power(np.clip(a * x + b, 0, None), g) + c
        return np.where(x >= thr, pos, c)
    if func_type == 3:
        g, a, b, c, d = params
        pos = np.power(np.clip(a * x + b, 0, None), g)
        lin = c * x
        return np.where(x >= d, pos, lin)
    if func_type == 4:
        g, a, b, c, d, e, f = params
        pos = np.power(np.clip(a * x + b, 0, None), g) + e
        lin = c * x + f
        return np.where(x >= d, pos, lin)
    raise ICCParseError(f"unknown parametricCurveType function type {func_type}")


def invert_para(func_type: int, params: Sequence[float], y: np.ndarray) -> np.ndarray:
    """Invert `eval_para` at `y`."""
    y = np.asarray(y, dtype=np.float64)
    if func_type == 0:
        (g,) = params
        return np.power(np.clip(y, 0, None), 1.0 / g)
    if func_type == 1:
        g, a, b = params
        return (np.power(np.clip(y, 0, None), 1.0 / g) - b) / a
    if func_type == 2:
        g, a, b, c = params
        return (np.power(np.clip(y - c, 0, None), 1.0 / g) - b) / a
    if func_type == 3:
        g, a, b, c, d = params
        y_break = c * d
        x_pos = (np.power(np.clip(y, 0, None), 1.0 / g) - b) / a
        x_lin = y / c if c != 0 else np.zeros_like(y)
        return np.where(y >= y_break, x_pos, x_lin)
    if func_type == 4:
        g, a, b, c, d, e, f = params
        y_break = c * d + f
        x_pos = (np.power(np.clip(y - e, 0, None), 1.0 / g) - b) / a
        x_lin = (y - f) / c if c != 0 else np.zeros_like(y)
        return np.where(y >= y_break, x_pos, x_lin)
    raise ICCParseError(f"unknown parametricCurveType function type {func_type}")


def eval_curve(curve: CurveDescriptor, x: np.ndarray) -> np.ndarray:
    """Evaluate a ``('curv', samples)`` or ``('para', (func_type, params))`` curve at `x`."""
    kind, payload = curve
    if kind == "curv":
        samples = payload
        if samples.size == 0:
            return x
        if samples.size == 1:
            return np.power(np.clip(x, 0, None), samples[0])
        n = samples.size
        pos = np.clip(x, 0.0, 1.0) * (n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        frac = pos - lo
        return samples[lo] * (1 - frac) + samples[hi] * frac
    elif kind == "para":
        ft, params = payload
        return eval_para(ft, params, x)
    raise ICCParseError(f"unknown curve kind {kind}")


def invert_curve_numeric(
    curve: CurveDescriptor, y: np.ndarray, lo: float = 0.0, hi: float = 1.0, iters: int = 40
) -> np.ndarray:
    """Invert a monotonic curve at `y` (analytically for 'para', by bisection for 'curv')."""
    kind, payload = curve
    if kind == "para":
        return invert_para(payload[0], payload[1], y)
    y = np.asarray(y, dtype=np.float64)
    if payload.size == 0:
        return y
    if payload.size == 1:
        return np.power(np.clip(y, 0, None), 1.0 / payload[0])
    a = np.full_like(y, lo)
    b = np.full_like(y, hi)
    increasing = eval_curve(curve, np.array([hi])) >= eval_curve(curve, np.array([lo]))
    for _ in range(iters):
        mid = (a + b) / 2
        fm = eval_curve(curve, mid)
        if increasing:
            go_right = fm < y
        else:
            go_right = fm > y
        a = np.where(go_right, mid, a)
        b = np.where(go_right, b, mid)
    return (a + b) / 2
