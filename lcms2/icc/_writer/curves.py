"""Encode one-dimensional curves as ICC curve tags and curve segments."""

from __future__ import annotations

import numpy as np

from .._io import mpe
from .._io import tagtypes as tt

TRC_SAMPLES = 4096
_SEGMENT_SAMPLES = 256
_MAX_SEGMENTS = 128


def evaluate(fn, x: np.ndarray) -> np.ndarray:
    """Evaluate `fn` at `x` as float64; NaN where `fn` raises."""
    x = np.asarray(x, dtype=np.float64)
    try:
        with np.errstate(all="ignore"):
            return np.asarray(fn(x), dtype=np.float64)
    except (ValueError, ArithmeticError, TypeError):
        if x.ndim == 0 or x.size == 1:
            return np.full(x.shape, np.nan)
        return np.array([evaluate(fn, v) for v in x.ravel()]).reshape(x.shape)


def detect_power_law(fn, lo: float, hi: float):
    """Return ``g`` if ``fn(x) == x**g`` on ``[lo, hi]`` (and 0 below 0), else None."""
    test_lo = max(lo, 1e-6)
    if hi <= test_lo:
        return None
    xs = np.geomspace(test_lo, hi, 32)
    ys = evaluate(fn, xs)
    if np.any(ys <= 0) or not np.all(np.isfinite(ys)):
        return None
    if lo < 0:
        below = evaluate(fn, -np.geomspace(1e-6, -lo, 8))
        if not np.allclose(below, 0.0, atol=1e-9):
            return None
    log_x, log_y = np.log(xs), np.log(ys)
    g = float(np.sum(log_x * log_y) / np.sum(log_x * log_x))
    rel_err = np.max(np.abs(np.power(xs, g) - ys) / np.abs(ys))
    return g if rel_err < 1e-6 else None


def detect_signed_power_law(fn, lo: float, hi: float):
    """Return ``g`` if ``fn(x) == sign(x) * |x|**g`` on ``[lo, hi]``, else None."""
    if lo >= 0 or hi <= 0:
        return None
    bound = min(-lo, hi)
    if bound <= 1e-6:
        return None
    xs = np.geomspace(1e-6, bound, 24)
    ys_pos = evaluate(fn, xs)
    ys_neg = evaluate(fn, -xs)
    if np.any(ys_pos <= 0) or not np.all(np.isfinite(ys_pos)):
        return None
    if not np.allclose(ys_neg, -ys_pos, rtol=1e-6, atol=1e-9):
        return None
    log_x, log_y = np.log(xs), np.log(ys_pos)
    g = float(np.sum(log_x * log_y) / np.sum(log_x * log_x))
    rel_err = np.max(np.abs(np.power(xs, g) - ys_pos) / np.abs(ys_pos))
    return g if rel_err < 1e-6 else None


def finite_range(fn, lo: float, hi: float, inner_lo: float, inner_hi: float):
    """Shrink ``[lo, hi]`` toward ``[inner_lo, inner_hi]`` until `fn` is finite at both ends.

    Returns
    -------
    tuple of float or None
        The shrunk range, or None if `fn` is not finite on the inner range.
    """
    inner_lo, inner_hi = min(inner_lo, inner_hi), max(inner_lo, inner_hi)
    probe = evaluate(fn, np.array([inner_lo, inner_hi]))
    if not np.all(np.isfinite(probe)):
        return None

    def shrink(outer, inner):
        if np.isfinite(evaluate(fn, np.array([outer]))[0]):
            return outer
        for _ in range(60):
            mid = 0.5 * (outer + inner)
            if np.isfinite(evaluate(fn, np.array([mid]))[0]):
                inner = mid
            else:
                outer = mid
        return inner

    return shrink(lo, inner_lo), shrink(hi, inner_hi)


def _midpoint_errors(fn, segments) -> list:
    mids = [a + (b - a) * (np.arange(v.size - 1) + 0.5) / (v.size - 1) for a, b, v in segments]
    exact = evaluate(fn, np.concatenate(mids))
    errors, start = [], 0
    for (_, _, values), m in zip(segments, mids, strict=False):
        err = np.abs(exact[start : start + m.size] - 0.5 * (values[:-1] + values[1:]))
        errors.append(float(np.max(err)) if np.all(np.isfinite(err)) else np.inf)
        start += m.size
    return errors


def _sample_segments(fn, bounds) -> list:
    points = [np.linspace(a, b, _SEGMENT_SAMPLES) for a, b in bounds]
    values = evaluate(fn, np.concatenate(points)).reshape(len(bounds), _SEGMENT_SAMPLES)
    return [(a, b, v) for (a, b), v in zip(bounds, values, strict=False)]


def adaptive_segments(fn, lo: float, hi: float, n_first: int = 4096, rel_tol: float = 1e-7):
    """Split ``[lo, hi]`` into uniformly sampled segments whose linear
    interpolation of `fn` stays within `rel_tol` of the curve's range.

    Returns
    -------
    list of tuple
        ``(seg_lo, seg_hi, samples)`` for each segment, in increasing order.
    """
    full = evaluate(fn, np.linspace(lo, hi, n_first))
    finite = full[np.isfinite(full)]
    tol = rel_tol * max(float(np.ptp(finite)) if finite.size else 1.0, 1e-12)
    # breakpoints are stored as float32: narrower segments would collapse
    min_width = 2e-7 * max(abs(lo), abs(hi), 1.0)

    pending, done = [(lo, hi, full)], []
    while pending:
        errors = _midpoint_errors(fn, pending)
        split = []
        for (a, b, values), err in zip(pending, errors, strict=False):
            room = len(done) + len(pending) + len(split) < _MAX_SEGMENTS
            if err <= tol or b - a <= min_width or not room:
                done.append((a, b, values))
            else:
                mid = 0.5 * (a + b)
                split += [(a, mid), (mid, b)]
        pending = _sample_segments(fn, split) if split else []
    done.sort(key=lambda seg: seg[0])
    return [(a, b, np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for a, b, v in done]


def _power_segments(g: float, signed: bool):
    power = ("formula", 0, [g, 1.0, 0.0, 0.0])
    if float(g).is_integer() and (not signed or int(g) % 2 == 1):
        return [], [power]
    if not signed:
        return [0.0], [mpe.constant_segment(0.0), power]
    return [0.0], [("formula", 3, [g, -1.0, -1.0, 0.0, 0.0]), power]


def power_curve(g: float, signed: bool = False) -> bytes:
    """A 'curf' curve for ``x ** g`` (0 below 0), or ``sign(x) * |x| ** g`` when `signed`."""
    return mpe.encode_segmented_curve(*_power_segments(g, signed))


def identity_curf() -> bytes:
    """A 'curf' identity curve."""
    return mpe.encode_segmented_curve([], [("formula", 0, [1.0, 1.0, 0.0, 0.0])])


def v5_curve(fn, lo: float, hi: float) -> bytes:
    """Encode `fn` on ``[lo, hi]`` as a 'curf' curve, constant beyond both ends."""
    g = detect_power_law(fn, lo, hi)
    if g is not None:
        return power_curve(g)
    g = detect_signed_power_law(fn, lo, hi)
    if g is not None:
        return power_curve(g, signed=True)
    pieces = []
    for a, b, samples in adaptive_segments(fn, lo, hi):
        # breakpoints are float32: drop segments that collapse to zero width there
        edge = pieces[-1][1] if pieces else a
        if np.float32(b) <= np.float32(edge):
            continue
        if pieces and np.float32(a) != np.float32(edge):
            a = edge
        pieces.append((a, b, samples))
    breakpoints = [pieces[0][0]] + [b for _, b, _ in pieces]
    segments = [mpe.constant_segment(pieces[0][2][0])]
    segments += [("sampled", samples[1:]) for _, _, samples in pieces]
    segments.append(mpe.constant_segment(pieces[-1][2][-1]))
    return mpe.encode_segmented_curve(breakpoints, segments)


def trc_tag(fn, version_major: int, n: int = TRC_SAMPLES) -> bytes:
    """Encode `fn` on [0, 1] as a 'curv' tag, or as 'para' for a v4 power law."""
    g = detect_power_law(fn, 0.0, 1.0)
    if g is not None:
        if version_major >= 4:
            return tt.encode_para(0, [g])
        if abs(g * 256 - round(g * 256)) < 1e-9:
            return tt.encode_curv(np.array([g]))
    return tt.encode_curv(evaluate(fn, np.linspace(0.0, 1.0, n)))


def sampled_curve(fn, n: int = TRC_SAMPLES):
    """Sample `fn` on [0, 1] into a ``('curv', samples)`` descriptor."""
    return ("curv", evaluate(fn, np.linspace(0.0, 1.0, n)))


def identity_curve():
    """A ``('curv', [])`` identity curve descriptor."""
    return ("curv", np.array([]))
