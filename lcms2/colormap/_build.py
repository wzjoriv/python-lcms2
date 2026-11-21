"""
Interpolation and gamut mapping behind `Colormap.from_colors`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..spaces import ColorSpace, get, oklab, srgb

GAMUTS = ("chroma", "clip", "none")
_EPS = 1e-9


def resolve_space(space: Any) -> ColorSpace:
    """Turn a registered name or a `ColorSpace` instance into an instance."""
    if isinstance(space, ColorSpace):
        return space
    if isinstance(space, str):
        key = space.lower().replace("_", "-")
        try:
            return get(key)
        except KeyError:
            raise ValueError(f"unknown colour space {space!r}; registered: {ColorSpace.list()}") from None
        except TypeError as e:
            raise ValueError(f"colour space {space!r} needs constructor arguments; pass an instance") from e
    raise TypeError(f"space must be a name or a ColorSpace, got {type(space).__name__}")


def _inside(rgb: np.ndarray) -> np.ndarray:
    return np.all((rgb >= -_EPS) & (rgb <= 1 + _EPS), axis=-1)


def gamut_map_oklab(lab: np.ndarray, mode: str = "chroma") -> np.ndarray:
    """Map OKLab colours to sRGB.

    ``"chroma"`` shrinks OKLCh chroma at constant lightness and hue (vectorised
    bisection) until the colour fits; ``"clip"`` clips; ``"none"`` leaves values.
    """
    if mode not in GAMUTS:
        raise ValueError(f"gamut must be one of {GAMUTS}, got {mode!r}")
    lab = np.asarray(lab, dtype=np.float64)
    xyz = oklab.to_xyz(lab)
    rgb = srgb.from_xyz(xyz)
    if mode == "none":
        return rgb
    if mode == "clip":
        return np.clip(rgb, 0.0, 1.0)
    bad = ~_inside(rgb)
    if bad.any():
        lch = oklab_to_lch(lab[bad])
        lch[:, 0] = np.clip(lch[:, 0], 0.0, 1.0)
        lo = np.zeros(len(lch))
        hi = lch[:, 1].copy()
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            trial = srgb.from_xyz(oklab.to_xyz(lch_to_lab(np.stack([lch[:, 0], mid, lch[:, 2]], -1))))
            ok = _inside(trial)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        mapped = srgb.from_xyz(oklab.to_xyz(lch_to_lab(np.stack([lch[:, 0], lo, lch[:, 2]], -1))))
        rgb[bad] = mapped
    return np.clip(rgb, 0.0, 1.0)


def oklab_to_lch(lab: np.ndarray) -> np.ndarray:
    """(L, a, b) to (L, C, h) with h in turns."""
    return np.stack(
        [
            lab[..., 0],
            np.hypot(lab[..., 1], lab[..., 2]),
            np.mod(np.arctan2(lab[..., 2], lab[..., 1]) / (2 * np.pi), 1.0),
        ],
        axis=-1,
    )


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """(L, C, h) with h in turns to (L, a, b)."""
    ang = 2 * np.pi * lch[..., 2]
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(ang), lch[..., 1] * np.sin(ang)], axis=-1)


def oklch_to_srgb(lch: np.ndarray, mode: str = "chroma") -> np.ndarray:
    """Gamut-map OKLCh (hue in turns) colours to sRGB."""
    return gamut_map_oklab(lch_to_lab(np.asarray(lch, dtype=np.float64)), mode)


def _unwrap_hue(h: np.ndarray, achromatic: np.ndarray) -> np.ndarray:
    """Make hue continuous along the shortest arc, borrowing hue for achromatic stops."""
    h = h.copy()
    idx = np.arange(len(h))
    if achromatic.all():
        return np.zeros_like(h)
    good = np.where(~achromatic, idx, -1)
    fwd = np.maximum.accumulate(good)
    back = np.where(~achromatic, idx, len(h))
    back = np.minimum.accumulate(back[::-1])[::-1]
    src = np.where(fwd >= 0, fwd, back)
    h = h[src]
    d = np.mod(np.diff(h) + 0.5, 1.0) - 0.5
    return h[0] + np.concatenate([[0.0], np.cumsum(d)])


def interpolate(
    colors: np.ndarray, space: ColorSpace, positions: np.ndarray, n: int, gamut: str
) -> np.ndarray:
    """Sample ``n`` colours along control ``colors`` (sRGB) interpolated in ``space``."""
    same = type(space) is type(srgb)
    vals = np.array(colors, dtype=np.float64) if same else np.asarray(srgb.to(space, colors), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    circular = sorted(space.circular_device_channels)
    out = np.empty((n, vals.shape[1]))
    for ch in range(vals.shape[1]):
        col = vals[:, ch]
        period = space.device_domain[ch][1] - space.device_domain[ch][0]
        if ch in circular and len(col) > 1:
            lo, hi = space.device_domain[1] if len(space.device_domain) > 1 else (0.0, 1.0)
            achromatic = vals[:, 1] <= 1e-3 * (hi - lo) if vals.shape[1] > 1 else np.zeros(len(col), bool)
            col = _unwrap_hue(np.mod(col / period, 1.0), achromatic) * period
        out[:, ch] = np.interp(t, positions, col)
        if ch in circular:
            out[:, ch] = np.mod(out[:, ch], period)
    rgb = out.copy() if same else space.to(srgb, out)
    if gamut == "chroma":
        bad = ~_inside(rgb)
        if bad.any():
            rgb[bad] = gamut_map_oklab(space.to(oklab, out[bad]), "chroma")
        rgb = np.clip(rgb, 0.0, 1.0)
    elif gamut == "clip":
        rgb = np.clip(rgb, 0.0, 1.0)
    elif gamut != "none":
        raise ValueError(f"gamut must be one of {GAMUTS}, got {gamut!r}")
    # Control points that land exactly on a sample are reproduced bit for bit.
    k = np.rint(positions * (n - 1)).astype(int)
    hit = np.isclose(t[k], positions, atol=1e-12)
    rgb[k[hit]] = colors[hit]
    return rgb
