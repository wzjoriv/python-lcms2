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
    """Apply the Adobe RGB (1998) transfer function from device values to linear light.

    Parameters
    ----------
    v : ndarray
        Device values, 0..1; negatives give NaN.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._a98_rgb import a98_decode
    >>> a98_decode(np.array([0.5, 1.0])).round(4).tolist()
    [0.2178, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.power(v, A98_GAMMA)


def a98_encode(v: np.ndarray) -> np.ndarray:
    """Apply the inverse Adobe RGB (1998) transfer function from linear light to device values.

    Parameters
    ----------
    v : ndarray
        Linear-light values, 0..1; negatives give NaN.

    Returns
    -------
    ndarray
        The transformed values, same shape as `v`.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._a98_rgb import a98_encode
    >>> a98_encode(np.array([0.5, 1.0])).round(4).tolist()
    [0.7297, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.power(v, 1.0 / A98_GAMMA)


class A98RGBSpace(RGBSpace):
    """Adobe RGB (1998) (CSS ``a98-rgb``). Negative values give NaN.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, a98_rgb
    >>> srgb.to(a98_rgb, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.86, 0.0, 0.0]
    """

    name = "a98-rgb"
    matrix = D65_TO_D50 @ d65_white_matrix(A98_TO_XYZ_D65)
    decode = staticmethod(a98_decode)
    encode = staticmethod(a98_encode)


a98_rgb = A98RGBSpace()

__all__ = ["A98RGBSpace", "a98_rgb"]
