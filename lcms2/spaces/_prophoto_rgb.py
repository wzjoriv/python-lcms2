"""
ProPhoto RGB (ROMM RGB).
"""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace

PROPHOTO_TO_XYZ_D50 = np.array(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100],
    ]
)


def prophoto_decode(v: np.ndarray) -> np.ndarray:
    """ProPhoto transfer function, device to linear."""
    with np.errstate(invalid="ignore"):
        return np.where(v < 16.0 / 512.0, v / 16.0, np.power(v, 1.8))


def prophoto_encode(v: np.ndarray) -> np.ndarray:
    """ProPhoto transfer function, linear to device."""
    with np.errstate(invalid="ignore"):
        return np.where(v < 1.0 / 512.0, v * 16.0, np.power(v, 1.0 / 1.8))


class ProPhotoRGBSpace(RGBSpace):
    """ProPhoto RGB / ROMM RGB (CSS ``prophoto-rgb``), D50 white."""

    name = "prophoto-rgb"
    matrix = PROPHOTO_TO_XYZ_D50
    decode = staticmethod(prophoto_decode)
    encode = staticmethod(prophoto_encode)


prophoto_rgb = ProPhotoRGBSpace()

__all__ = ["ProPhotoRGBSpace", "prophoto_rgb"]
