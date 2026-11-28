"""
HSV over sRGB.
"""

from __future__ import annotations

import numpy as np

from ._base import ColorSpace
from ._srgb import srgb


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV (hue in degrees) to RGB."""
    h, s, v = hsv[..., 0] / 60.0, hsv[..., 1], hsv[..., 2]
    sector = np.floor(h)
    f = h - sector
    i = np.mod(sector, 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV (hue in degrees, 0..360); hue is 0 for grays."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    chromatic = delta > 1e-12
    safe_delta = np.where(chromatic, delta, 1.0)
    s = np.where(maxc > 1e-12, delta / np.where(maxc > 1e-12, maxc, 1.0), 0.0)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.select([maxc == r, maxc == g], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    h = np.where(chromatic, np.mod(h * 60.0 + 360.0, 360.0), 0.0)
    return np.stack([h, s, maxc], axis=-1)


class HSVSpace(ColorSpace):
    """HSV over sRGB; hue in degrees (circular), saturation and value in 0..1.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, hsv
    >>> srgb.to(hsv, [0.0, 1.0, 0.0]).round(2).tolist()
    [120.0, 1.0, 1.0]
    """

    name = "hsv"
    n_channels = 3
    icc_colour_space_signature = b"HSV "
    circular_device_channels = frozenset({0})
    device_domain = [(0.0, 360.0), (0.0, 1.0), (0.0, 1.0)]

    def to_xyz(self, hsv: np.ndarray) -> np.ndarray:
        return srgb.to_xyz(hsv_to_rgb(np.asarray(hsv, dtype=np.float64)))

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return rgb_to_hsv(srgb.from_xyz(xyz))


hsv = HSVSpace()

__all__ = ["HSVSpace", "hsv", "hsv_to_rgb", "rgb_to_hsv"]
