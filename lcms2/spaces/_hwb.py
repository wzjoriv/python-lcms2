"""HWB over sRGB."""

from __future__ import annotations

import numpy as np

from ._base import ColorSpace
from ._hsv import hsv_to_rgb, rgb_to_hsv
from ._srgb import srgb


class HWBSpace(ColorSpace):
    """Hue, whiteness, blackness over sRGB (CSS Color 4); whiteness plus
    blackness above 1 is normalized to 1."""

    name = "hwb"
    n_channels = 3
    icc_colour_space_signature = b"3CLR"
    circular_device_channels = frozenset({0})

    def to_xyz(self, hwb: np.ndarray) -> np.ndarray:
        """Convert HWB to CIE XYZ (D50)."""
        hwb = np.asarray(hwb, dtype=np.float64)
        w, bl = hwb[..., 1], hwb[..., 2]
        total = w + bl
        norm = np.where(total > 1.0, total, 1.0)
        w, bl = w / norm, bl / norm
        v = 1.0 - bl
        s = np.where(v > 1e-12, 1.0 - w / np.where(v > 1e-12, v, 1.0), 0.0)
        return srgb.to_xyz(hsv_to_rgb(np.stack([hwb[..., 0], s, v], axis=-1)))

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to HWB."""
        hsv = rgb_to_hsv(srgb.from_xyz(xyz))
        s, v = hsv[..., 1], hsv[..., 2]
        return np.stack([hsv[..., 0], (1.0 - s) * v, 1.0 - v], axis=-1)


hwb = HWBSpace()

__all__ = ["HWBSpace", "hwb"]
