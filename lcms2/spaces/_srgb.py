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
    """Apply the sRGB transfer function from device values to linear light.

    Parameters
    ----------
    v : ndarray
        Device values, 0..1; the curve extends linearly below 0.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._srgb import srgb_decode
    >>> srgb_decode(np.array([0.5, 1.0])).round(4).tolist()
    [0.214, 1.0]
    """
    if not isinstance(v, np.ndarray) or v.ndim == 0:
        with np.errstate(invalid="ignore"):
            return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))
    out = v + 0.055
    out /= 1.055
    with np.errstate(invalid="ignore"):
        np.power(out, 2.4, out=out)
    np.divide(v, 12.92, out=out, where=v <= 0.04045)
    return out


def srgb_encode(v: np.ndarray) -> np.ndarray:
    """Apply the inverse sRGB transfer function from linear light to device values.

    Parameters
    ----------
    v : ndarray
        Linear-light values, 0..1; the curve extends linearly below 0.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._srgb import srgb_encode
    >>> srgb_encode(np.array([0.5, 1.0])).round(4).tolist()
    [0.7354, 1.0]
    """
    with np.errstate(invalid="ignore"):
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
        out = np.power(v, 1.0 / 2.4)
    out *= 1.055
    out -= 0.055
    np.multiply(v, 12.92, out=out, where=v <= 0.0031308)
    return out


class SRGBSpace(RGBSpace):
    """sRGB: IEC 61966-2-1 primaries and transfer function, D65 adapted to D50.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, xyz_d50
    >>> srgb.to(xyz_d50, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.44, 0.22, 0.01]
    """

    name = "srgb"
    matrix = D65_TO_D50 @ SRGB_TO_XYZ_D65
    decode = staticmethod(srgb_decode)
    encode = staticmethod(srgb_encode)


srgb = SRGBSpace()

__all__ = ["SRGBSpace", "d65_white_matrix", "srgb", "srgb_decode", "srgb_encode"]
