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
    """Apply the ProPhoto RGB transfer function from device values to linear light.

    Parameters
    ----------
    v : ndarray
        Device values, 0..1.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._prophoto_rgb import prophoto_decode
    >>> prophoto_decode(np.array([0.5, 1.0])).round(4).tolist()
    [0.2872, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.where(v < 16.0 / 512.0, v / 16.0, np.power(v, 1.8))


def prophoto_encode(v: np.ndarray) -> np.ndarray:
    """Apply the inverse ProPhoto RGB transfer function from linear light to device values.

    Parameters
    ----------
    v : ndarray
        Linear-light values, 0..1.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._prophoto_rgb import prophoto_encode
    >>> prophoto_encode(np.array([0.5, 1.0])).round(4).tolist()
    [0.6804, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.where(v < 1.0 / 512.0, v * 16.0, np.power(v, 1.0 / 1.8))


class ProPhotoRGBSpace(RGBSpace):
    """ProPhoto RGB / ROMM RGB (CSS ``prophoto-rgb``), D50 white.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, prophoto_rgb
    >>> srgb.to(prophoto_rgb, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.7, 0.28, 0.1]
    """

    name = "prophoto-rgb"
    matrix = PROPHOTO_TO_XYZ_D50
    decode = staticmethod(prophoto_decode)
    encode = staticmethod(prophoto_encode)


prophoto_rgb = ProPhotoRGBSpace()

__all__ = ["ProPhotoRGBSpace", "prophoto_rgb"]
