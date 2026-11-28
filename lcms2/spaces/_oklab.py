"""
OKLab.
"""

from __future__ import annotations

import numpy as np

from ._base import ColorSpace
from ._rgb import mat3
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
    """OKLab (Ottosson 2020) on linear sRGB primaries, adapted to D50.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, oklab
    >>> srgb.to(oklab, [1.0, 1.0, 1.0]).round(2).tolist()
    [1.0, 0.0, 0.0]
    """

    name = "oklab"
    icc_colour_space_signature = b"3CLR"
    device_domain = [(0.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)]

    def to_xyz(self, oklab: np.ndarray) -> np.ndarray:
        lms_ = mat3(_M2_INV, oklab)
        return mat3(_LMS_TO_XYZ_D50, lms_**3)

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return mat3(OKLAB_M2, np.cbrt(mat3(_XYZ_D50_TO_LMS, xyz)))


oklab = OKLabSpace()

__all__ = ["OKLabSpace", "oklab"]
