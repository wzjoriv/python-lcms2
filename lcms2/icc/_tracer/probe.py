"""Numerically detect curve/matrix structure by evaluating a function."""

from __future__ import annotations

import numpy as np

_N_LINE = 257
_N_CHECK = 256


def _evaluate(fn, x: np.ndarray, n_out: int):
    try:
        with np.errstate(all="ignore"):
            y = np.asarray(fn(x), dtype=np.float64)
    except Exception:
        return None
    if y.shape != x.shape[:-1] + (n_out,):
        return None
    return y


def _rank1_direction(samples: np.ndarray, rtol: float):
    finite = np.all(np.isfinite(samples), axis=-1)
    samples = samples[finite]
    if samples.shape[0] < 2:
        return None
    _, s, vt = np.linalg.svd(samples, full_matrices=False)
    if s[0] <= 0.0 or (s.size > 1 and s[1] > rtol * s[0]):
        return None
    direction = vt[0]
    if np.sum(samples @ direction) < 0:
        direction = -direction
    return direction


def _close(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    finite = np.isfinite(a)
    if not np.array_equal(finite, np.isfinite(b)) or not finite.any():
        return False
    scale = max(float(np.max(np.abs(a[finite]))), 1e-12)
    return bool(np.max(np.abs(a[finite] - b[finite])) <= rtol * scale)


def probe_curves_then_matrix(fn, n_in: int, n_out: int, domain=None, rtol: float = 1e-7):
    """Detect ``fn(x) == stack(c_i(x_i)) @ M + offset`` over a device domain.

    Parameters
    ----------
    fn : callable
        Function mapping ``(..., n_in)`` to ``(..., n_out)``.
    n_in, n_out : int
        Input and output channel counts.
    domain : list of (float, float), optional
        Per-channel input range. Defaults to [0, 1] for every channel.
    rtol : float, optional
        Relative tolerance of the structure checks.

    Returns
    -------
    tuple or None
        ``(curves, matrix, offset)`` with `curves` a list of callables,
        `matrix` of shape ``(n_in, n_out)`` whose rows have unit norm, and
        `offset` of shape ``(n_out,)``; None if `fn` has another structure.
    """
    domain = domain or [(0.0, 1.0)] * n_in
    lo = np.array([d[0] for d in domain], dtype=np.float64)
    hi = np.array([d[1] for d in domain], dtype=np.float64)
    for base in (lo.copy(), 0.5 * (lo + hi)):
        found = _probe_curves_then_matrix_at(fn, n_in, n_out, lo, hi, base, rtol)
        if found is not None:
            return found
    return None


def _probe_curves_then_matrix_at(fn, n_in, n_out, lo, hi, base, rtol):
    f0 = _evaluate(fn, base[None, :], n_out)
    if f0 is None or not np.all(np.isfinite(f0)):
        return None
    f0 = f0[0]

    rows = []
    for i in range(n_in):
        pts = np.repeat(base[None, :], _N_LINE, axis=0)
        pts[:, i] = np.linspace(lo[i], hi[i], _N_LINE)
        vals = _evaluate(fn, pts, n_out)
        if vals is None:
            return None
        direction = _rank1_direction(vals - f0, rtol)
        if direction is None:
            return None
        rows.append(direction)
    matrix = np.array(rows)

    coeffs, *_ = np.linalg.lstsq(matrix.T, f0, rcond=None)
    offset = f0 - coeffs @ matrix
    if np.max(np.abs(offset)) <= rtol * max(float(np.max(np.abs(f0))), 1.0):
        offset = np.zeros(n_out)

    def make_curve(i):
        direction = matrix[i]
        shift = coeffs[i]

        def curve(t):
            t = np.asarray(t, dtype=np.float64)
            pts = np.broadcast_to(base, t.shape + (n_in,)).copy()
            pts[..., i] = t
            with np.errstate(all="ignore"):
                vals = np.asarray(fn(pts), dtype=np.float64)
            return (vals - f0) @ direction + shift

        return curve

    curves = [make_curve(i) for i in range(n_in)]

    rng = np.random.default_rng(12345)
    x = lo + (hi - lo) * rng.random((_N_CHECK, n_in))
    expected = _evaluate(fn, x, n_out)
    if expected is None:
        return None
    predicted = np.stack([c(x[:, i]) for i, c in enumerate(curves)], axis=-1) @ matrix + offset
    if not _close(expected, predicted, max(rtol, 1e-6)):
        return None
    return curves, matrix, offset


def _gradients(fn, points: np.ndarray, n_in: int, n_out: int, step: float):
    grads = np.zeros((points.shape[0], n_out, n_in))
    for k in range(n_in):
        delta = np.zeros(n_in)
        delta[k] = step
        vals = [_evaluate(fn, points + m * delta, n_out) for m in (2, 1, -1, -2)]
        if any(v is None for v in vals):
            return None
        grads[:, :, k] = (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * step)
    return grads


def _matrix_curves(fn, matrix: np.ndarray, n_out: int):
    def make_curve(j):
        direction = matrix[:, j]

        def curve(u):
            u = np.asarray(u, dtype=np.float64)
            with np.errstate(all="ignore"):
                vals = np.asarray(fn(u[..., None] * direction), dtype=np.float64)
            return vals[..., j]

        return curve

    return [make_curve(j) for j in range(n_out)]


def _check_matrix_curves(fn, matrix, samples, n_out, rtol, rng):
    curves = _matrix_curves(fn, matrix, n_out)
    idx = rng.choice(samples.shape[0], size=min(_N_CHECK, samples.shape[0]), replace=False)
    check = samples[idx]
    expected = _evaluate(fn, check, n_out)
    if expected is None:
        return None
    projected = check @ matrix
    predicted = np.stack([c(projected[:, j]) for j, c in enumerate(curves)], axis=-1)
    if not _close(expected, predicted, rtol):
        return None
    projected = samples @ matrix
    ranges = [(float(projected[:, j].min()), float(projected[:, j].max())) for j in range(n_out)]
    return matrix, curves, ranges


def probe_matrix_then_curves(
    fn, n_in: int, n_out: int, samples: np.ndarray, matrix_hint=None, rtol: float = 1e-6
):
    """Detect ``fn(x) == stack(c_j((x @ M)_j))`` around the given samples.

    Parameters
    ----------
    fn : callable
        Function mapping ``(..., n_in)`` to ``(..., n_out)``.
    n_in, n_out : int
        Input and output channel counts.
    samples : ndarray, shape (k, n_in)
        Representative inputs.
    matrix_hint : ndarray, shape (n_in, n_out), optional
        Candidate matrix tried before estimating one from gradients.
    rtol : float, optional
        Relative tolerance of the structure checks.

    Returns
    -------
    tuple or None
        ``(matrix, curves, ranges)`` with `matrix` of shape
        ``(n_in, n_out)`` whose columns have unit norm, `curves` a list of
        callables and `ranges` the ``(lo, hi)`` of each curve's input over
        `samples`; None if `fn` has another structure.
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = _evaluate(fn, samples, n_out)
    if y is None:
        return None
    samples = samples[np.all(np.isfinite(y), axis=-1)]
    if samples.shape[0] < 8:
        return None
    rng = np.random.default_rng(54321)

    if matrix_hint is not None and np.shape(matrix_hint) == (n_in, n_out):
        hint = np.asarray(matrix_hint, dtype=np.float64)
        norms = np.linalg.norm(hint, axis=0)
        if np.all(norms > 0):
            found = _check_matrix_curves(fn, hint / norms, samples, n_out, rtol, rng)
            if found is not None:
                return found

    pick = samples[rng.choice(samples.shape[0], size=min(64, samples.shape[0]), replace=False)]
    step = 1e-4 * max(float(np.max(np.abs(samples))), 1e-6)
    grads = _gradients(fn, pick, n_in, n_out, step)
    if grads is None:
        return None
    columns = []
    for j in range(n_out):
        g = grads[:, j, :]
        g = g[np.all(np.isfinite(g), axis=-1)]
        g = g[np.linalg.norm(g, axis=-1) > 0]
        direction = _rank1_direction(g, 1e-4)
        if direction is None:
            return None
        big = np.argmax(np.abs(direction))
        direction = np.where(np.abs(direction) < 1e-9, 0.0, direction)
        direction = direction / np.linalg.norm(direction) * np.sign(direction[big])
        columns.append(direction)
    return _check_matrix_curves(fn, np.array(columns).T, samples, n_out, rtol, rng)
