"""Formula CMYK over sRGB."""

from __future__ import annotations

import numpy as np

from ._base import ColorSpace
from ._srgb import srgb


class CMYKSpace(ColorSpace):
    """CMYK from the complement of sRGB with ``K = min(C, M, Y)``; not an ink model."""

    name = "cmyk"
    n_channels = 4
    icc_colour_space_signature = b"CMYK"

    def to_xyz(self, cmyk: np.ndarray) -> np.ndarray:
        """Convert CMYK to CIE XYZ (D50)."""
        cmyk = np.asarray(cmyk, dtype=np.float64)
        k = 1.0 - cmyk[..., 3]
        rgb = np.stack([(1.0 - cmyk[..., i]) * k for i in range(3)], axis=-1)
        return srgb.to_xyz(rgb)

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to CMYK."""
        cmy = 1.0 - srgb.from_xyz(xyz)
        k = np.minimum(np.minimum(cmy[..., 0], cmy[..., 1]), cmy[..., 2])
        ink = k < 1.0 - 1e-12
        denom = np.where(ink, 1.0 - k, 1.0)
        c, m, y = (np.where(ink, (cmy[..., i] - k) / denom, 0.0) for i in range(3))
        return np.stack([c, m, y, k], axis=-1)


cmyk = CMYKSpace()

__all__ = ["CMYKSpace", "cmyk"]
