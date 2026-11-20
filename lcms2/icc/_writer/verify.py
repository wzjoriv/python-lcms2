"""
Sample-based checks that a written representation reproduces a space.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike

VERIFY_RTOL = 1e-5

VERIFY_MAX_RTOL = 1e-3


def call(fn: Callable[[np.ndarray], ArrayLike], x: np.ndarray) -> np.ndarray:
    """Evaluate `fn` at `x` as float64 without floating-point warnings."""
    with np.errstate(all="ignore"):
        return np.asarray(fn(x), dtype=np.float64)


def agrees(
    expected: np.ndarray,
    actual: np.ndarray,
    periods: Mapping[int, float] | None = None,
    rtol: float = VERIFY_RTOL,
    max_rtol: float = VERIFY_MAX_RTOL,
) -> bool:
    """Whether `actual` matches `expected` at the 99th percentile and at the maximum.

    Channels in `periods` are compared modulo their period.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if periods:
        diff = actual - expected
        for ch, period in periods.items():
            diff[..., ch] = (diff[..., ch] + 0.5 * period) % period - 0.5 * period
        actual = expected + diff
    ok = np.isfinite(expected)
    if not ok.any() or not np.all(np.isfinite(actual[ok])):
        return False
    scale = max(float(np.max(np.abs(expected[ok]))), 1.0)
    err = np.abs(expected[ok] - actual[ok])
    return bool(np.percentile(err, 99) <= rtol * scale and err.max() <= max_rtol * scale)
