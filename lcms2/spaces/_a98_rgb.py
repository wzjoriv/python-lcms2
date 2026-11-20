"""
Adobe RGB (1998).
"""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50, d65_white_matrix

A98_TO_XYZ_D65 = np.array(
    [
        [0.5766690, 0.1855582, 0.1882286],
        [0.2973450, 0.6273636, 0.0752915],
        [0.0270314, 0.0706889, 0.9913375],
    ]
)

A98_GAMMA = 563.0 / 256.0


def a98_decode(v: np.ndarray) -> np.ndarray:
    """Adobe RGB transfer function, device to linear (``v ** 2.19921875``)."""
    with np.errstate(invalid="ignore"):
        return np.power(v, A98_GAMMA)


def a98_encode(v: np.ndarray) -> np.ndarray:
    """Adobe RGB transfer function, linear to device."""
    with np.errstate(invalid="ignore"):
        return np.power(v, 1.0 / A98_GAMMA)


class A98RGBSpace(RGBSpace):
    """Adobe RGB (1998) (CSS ``a98-rgb``). Negative values give NaN."""

    name = "a98-rgb"
    matrix = D65_TO_D50 @ d65_white_matrix(A98_TO_XYZ_D65)
    decode = staticmethod(a98_decode)
    encode = staticmethod(a98_encode)


a98_rgb = A98RGBSpace()

__all__ = ["A98RGBSpace", "a98_rgb"]
