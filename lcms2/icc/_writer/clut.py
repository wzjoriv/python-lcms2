"""
Sampled lookup-table tags for spaces without an exact representation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ...colorimetry import D50
from .._interp import sample_clut
from .._io import mpe
from .._reader import XYZ_U1F15_SCALE
from . import curves as cv
from .context import Context
from .lut import encode_lut_tag
from .sampling import sample_grid
from .structure import identity


def cube_root(t: ArrayLike) -> np.ndarray:
    """Signed cube root."""
    t = np.asarray(t, dtype=np.float64)
    return np.sign(t) * np.abs(t) ** (1.0 / 3.0)


_A2B_GRID = {1: 255, 2: 65, 3: 33}

_A2B_GRID_DEFAULT = 17

_B2A_GRID = 33


def _white_knots(grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Cube-root PCS white per channel and the grid node it is moved onto."""
    white = cube_root(D50 / XYZ_U1F15_SCALE)
    return white, np.round(white * (grid - 1)) / (grid - 1)


def _warp(s: np.ndarray, knots: tuple[np.ndarray, np.ndarray], i: int) -> np.ndarray:
    """Piecewise-linear map of cube-root PCS channel ``i`` that puts white on a grid node."""
    w, k = knots[0][i], knots[1][i]
    s = np.asarray(s, dtype=np.float64)
    return np.where(s <= w, s * (k / w), k + (s - w) * ((1.0 - k) / (1.0 - w)))


def _unwarp(t: np.ndarray, knots: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Inverse of `_warp` for all three channels (last axis)."""
    w, k = knots
    t = np.asarray(t, dtype=np.float64)
    return np.where(t <= k, t * (w / k), w + (t - k) * ((1.0 - w) / (1.0 - k)))


def cube(t: ArrayLike) -> np.ndarray:
    """Cube."""
    return np.asarray(t, dtype=np.float64) ** 3


def write_clut(ctx: Context, tags: dict, notes: list, a2b_only: bool = False) -> None:
    """Write sampled A2B0/B2A0 lookup tables for any space."""
    a2b_grid = ctx.grid_size or _A2B_GRID.get(ctx.n, _A2B_GRID_DEFAULT)
    b2a_grid = ctx.grid_size or _B2A_GRID
    linear = np.nan_to_num(sample_grid(ctx.fwd, [(0.0, 1.0)] * ctx.n, a2b_grid))
    shaped = cube_root(linear)
    probe = ctx.device_samples[: min(2048, len(ctx.device_samples))]
    truth = ctx.xyz_samples[: len(probe)]
    keep = np.all(np.isfinite(truth), axis=-1)
    err_linear = np.abs(sample_clut(linear, probe) - truth)[keep]
    err_shaped = np.abs(cube(sample_clut(shaped, probe)) - truth)[keep]
    use_shaping = err_shaped.size == 0 or np.percentile(err_shaped, 99) <= np.percentile(err_linear, 99)
    a2b = shaped if use_shaping else linear
    out_curve = cube if use_shaping else identity

    if ctx.major == 5:
        xyz = ctx.xyz_finite
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        pad = 0.05 * (hi - lo) + 1e-6
        s_lo = cube_root(np.where(lo < 0, lo - pad, np.maximum(lo - pad, 0.0)))
        s_hi = cube_root(hi + pad)
        span = s_hi - s_lo
        b2a = sample_grid(lambda s: ctx.bwd(s**3), list(zip(s_lo, s_hi, strict=False)), b2a_grid)
        b2a = np.clip(np.nan_to_num(b2a), 0.0, 1.0)
        a2b_elems = [mpe.encode_clut_element(a2b)]
        if use_shaping:
            a2b_elems.append(mpe.encode_curveset_element([cv.power_curve(3.0, signed=True)] * 3))
        tags[b"A2B0"] = mpe.encode_mpet(ctx.n, 3, a2b_elems)
        if a2b_only:
            notes.append(f"sampled onto a {a2b_grid}^{ctx.n} A2B0 grid")
            return
        tags[b"B2A0"] = mpe.encode_mpet(
            3,
            ctx.n,
            [
                mpe.encode_curveset_element([cv.power_curve(1.0 / 3.0, signed=True)] * 3),
                mpe.encode_matrix_element(np.diag(1.0 / span), -s_lo / span),
                mpe.encode_clut_element(b2a),
            ],
        )
    else:
        knots = _white_knots(b2a_grid)
        b2a = sample_grid(
            lambda t: ctx.bwd(_unwarp(t, knots) ** 3 * XYZ_U1F15_SCALE), [(0.0, 1.0)] * 3, b2a_grid
        )
        pcs_max = cube_root(XYZ_U1F15_SCALE) if use_shaping else XYZ_U1F15_SCALE
        a2b_enc = np.clip(a2b / pcs_max, 0.0, 1.0)
        b2a_enc = np.clip(np.nan_to_num(b2a), 0.0, 1.0)
        tags[b"A2B0"] = encode_lut_tag(ctx, False, None, a2b_enc, [out_curve] * 3 if use_shaping else None)
        if a2b_only:
            notes.append(f"sampled onto a {a2b_grid}^{ctx.n} A2B0 grid")
            return
        b_curves = [lambda u, i=i: _warp(cube_root(u), knots, i) for i in range(3)]
        tags[b"B2A0"] = encode_lut_tag(ctx, True, b_curves, b2a_enc, None)
    shaping = "cube-root-shaped " if use_shaping else ""
    notes.append(f"sampled onto a {shaping}{a2b_grid}^{ctx.n} A2B0 grid and a {b2a_grid}^3 B2A0 grid")
