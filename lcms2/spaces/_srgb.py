"""
sRGB (IEC 61966-2-1).
"""

from __future__ import annotations

import numpy as np

from ..colorimetry import D50, D65, adaptation_matrix
from ._rgb import RGBSpace

D65_TO_D50 = adaptation_matrix(D65, D50)

SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def d65_white_matrix(rgb_to_xyz_d65: np.ndarray) -> np.ndarray:
    """Scale the rows of an RGB-to-XYZ (D65) matrix so RGB white maps to `D65`.

    Published matrices are rounded and some assume a slightly different D65,
    which would leave white off the D65 point (and so off D50 after adaptation).
    """
    return rgb_to_xyz_d65 * (D65 / (rgb_to_xyz_d65 @ np.ones(3)))[:, None]


def srgb_decode(v: np.ndarray) -> np.ndarray:
    """sRGB transfer function, device to linear; linear extension below 0."""
    with np.errstate(invalid="ignore"):
        return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def srgb_encode(v: np.ndarray) -> np.ndarray:
    """sRGB transfer function, linear to device; linear extension below 0."""
    with np.errstate(invalid="ignore"):
        return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


class SRGBSpace(RGBSpace):
    """sRGB: IEC 61966-2-1 primaries and transfer function, D65 adapted to D50."""

    name = "srgb"
    matrix = D65_TO_D50 @ SRGB_TO_XYZ_D65
    decode = staticmethod(srgb_decode)
    encode = staticmethod(srgb_encode)


srgb = SRGBSpace()

__all__ = ["SRGBSpace", "d65_white_matrix", "srgb", "srgb_decode", "srgb_encode"]
