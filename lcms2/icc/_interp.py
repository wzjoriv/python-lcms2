"""
Multilinear interpolation over regular-grid lookup tables.
"""

from __future__ import annotations

import itertools

import numpy as np


def sample_clut(clut: np.ndarray, coords: np.ndarray, circular_channels=frozenset()) -> np.ndarray:
    """Interpolate `clut` at `coords` in [0, 1].

    Parameters
    ----------
    clut : ndarray, shape grid_points + (out_ch,)
        Table samples.
    coords : ndarray, shape (..., k)
        Normalized input coordinates; clipped to [0, 1].
    circular_channels : frozenset of int, optional
        Output channels holding a fraction of a turn, blended along the circle.

    Returns
    -------
    ndarray, shape (..., out_ch)
    """
    k = coords.shape[-1]
    grid_shape = clut.shape[:k]
    out_ch = clut.shape[-1]

    flat_coords = coords.reshape(-1, k)
    n = flat_coords.shape[0]

    pos = np.empty_like(flat_coords)
    for d in range(k):
        pos[:, d] = np.clip(flat_coords[:, d], 0.0, 1.0) * (grid_shape[d] - 1)

    lo = np.floor(pos).astype(np.int64)
    for d in range(k):
        lo[:, d] = np.clip(lo[:, d], 0, grid_shape[d] - 2 if grid_shape[d] > 1 else 0)
    frac = pos - lo

    lin_ch = [c for c in range(out_ch) if c not in circular_channels]
    circ_ch = [c for c in range(out_ch) if c in circular_channels]

    out_lin = np.zeros((n, len(lin_ch)), dtype=np.float64) if lin_ch else None
    cos_acc = np.zeros((n, len(circ_ch)), dtype=np.float64) if circ_ch else None
    sin_acc = np.zeros((n, len(circ_ch)), dtype=np.float64) if circ_ch else None

    for corner in itertools.product([0, 1], repeat=k):
        weight = np.ones(n, dtype=np.float64)
        idx = []
        for d in range(k):
            if grid_shape[d] == 1:
                cidx = np.zeros(n, dtype=np.int64)
                w = np.ones(n, dtype=np.float64)
            else:
                cidx = lo[:, d] + corner[d]
                w = frac[:, d] if corner[d] == 1 else (1.0 - frac[:, d])
            idx.append(cidx)
            weight *= w
        corner_vals = clut[tuple(idx) + (slice(None),)]
        if lin_ch:
            out_lin += weight[:, None] * corner_vals[:, lin_ch]
        if circ_ch:
            angles = corner_vals[:, circ_ch] * (2 * np.pi)
            cos_acc += weight[:, None] * np.cos(angles)
            sin_acc += weight[:, None] * np.sin(angles)

    out = np.zeros((n, out_ch), dtype=np.float64)
    if lin_ch:
        out[:, lin_ch] = out_lin
    if circ_ch:
        hue = np.mod(np.arctan2(sin_acc, cos_acc) / (2 * np.pi), 1.0)
        out[:, circ_ch] = hue

    return out.reshape(coords.shape[:-1] + (out_ch,))
