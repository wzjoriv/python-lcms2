"""
OKLab.
"""

from __future__ import annotations

import numpy as np

from ._base import ColorSpace
from ._srgb import D65_TO_D50, SRGB_TO_XYZ_D65

OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
# OKLAB_M1 maps linear sRGB to LMS, so LMS -> XYZ goes through linear sRGB.
_LMS_TO_XYZ_D50 = D65_TO_D50 @ SRGB_TO_XYZ_D65 @ np.linalg.inv(OKLAB_M1)
_XYZ_D50_TO_LMS = np.linalg.inv(_LMS_TO_XYZ_D50)
_M2_INV = np.linalg.inv(OKLAB_M2)


class OKLabSpace(ColorSpace):
    """OKLab (Ottosson 2020) on linear sRGB primaries, adapted to D50."""

    name = "oklab"
    icc_colour_space_signature = b"3CLR"
    device_domain = [(0.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)]

    def to_xyz(self, oklab: np.ndarray) -> np.ndarray:
        """Convert OKLab to CIE XYZ (D50)."""
        lms_ = np.asarray(oklab, dtype=np.float64) @ _M2_INV.T
        return (lms_**3) @ _LMS_TO_XYZ_D50.T

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to OKLab."""
        lms = np.asarray(xyz, dtype=np.float64) @ _XYZ_D50_TO_LMS.T
        return np.cbrt(lms) @ OKLAB_M2.T


oklab = OKLabSpace()

__all__ = ["OKLabSpace", "oklab"]
