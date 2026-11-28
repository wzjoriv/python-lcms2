"""
OKLCh.
"""

from __future__ import annotations

import numpy as np

from ..colorimetry import lab_to_lch, lch_to_lab
from ._base import ColorSpace
from ._oklab import oklab


class OKLChSpace(ColorSpace):
    """Cylindrical OKLab (CSS ``oklch``); hue in degrees.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, oklch
    >>> srgb.to(oklch, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.63, 0.26, 29.23]
    """

    name = "oklch"
    icc_colour_space_signature = b"3CLR"
    circular_device_channels = frozenset({2})
    device_domain = [(0.0, 1.0), (0.0, 0.5), (0.0, 360.0)]

    def to_xyz(self, oklch: np.ndarray) -> np.ndarray:
        return oklab.to_xyz(lch_to_lab(np.asarray(oklch, dtype=np.float64)))

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return lab_to_lch(oklab.from_xyz(xyz))


oklch = OKLChSpace()

__all__ = ["OKLChSpace", "oklch"]
