"""
Sample functions on regular grids for CLUT tags.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


def grid_points(ranges: Iterable[tuple[float, float]], grid_size: int) -> np.ndarray:
    """Regular grid over per-axis ``(lo, hi)`` ranges, shape ``(grid_size,)*k + (k,)``."""
    axes = [np.linspace(lo, hi, grid_size) for lo, hi in ranges]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def sample_grid(
    fn: Callable[[np.ndarray], ArrayLike], ranges: Sequence[tuple[float, float]], grid_size: int
) -> np.ndarray:
    """Evaluate `fn` on a regular grid, returning ``(grid_size,)*k + (out_ch,)``."""
    pts = grid_points(ranges, grid_size)
    k = pts.shape[-1]
    with np.errstate(all="ignore"):
        flat = np.asarray(fn(pts.reshape(-1, k)), dtype=np.float64)
    return flat.reshape(pts.shape[:-1] + (flat.shape[-1],))
