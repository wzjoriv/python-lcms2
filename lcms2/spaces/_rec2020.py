"""
ITU-R BT.2020.
"""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50, d65_white_matrix

REC2020_TO_XYZ_D65 = np.array(
    [
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ]
)

REC2020_ALPHA = 1.099296826809442
REC2020_BETA = 0.018053968510807


def rec2020_decode(v: np.ndarray) -> np.ndarray:
    """Apply the ITU-R BT.2020 transfer function from device values to linear light.

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
    >>> from lcms2.spaces._rec2020 import rec2020_decode
    >>> rec2020_decode(np.array([0.5, 1.0])).round(4).tolist()
    [0.2597, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.where(
            v < REC2020_BETA * 4.5,
            v / 4.5,
            np.power((v + REC2020_ALPHA - 1.0) / REC2020_ALPHA, 1.0 / 0.45),
        )


def rec2020_encode(v: np.ndarray) -> np.ndarray:
    """Apply the inverse ITU-R BT.2020 transfer function from linear light to device values.

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
    >>> from lcms2.spaces._rec2020 import rec2020_encode
    >>> rec2020_encode(np.array([0.5, 1.0])).round(4).tolist()
    [0.7054, 1.0]
    """
    with np.errstate(invalid="ignore"):
        return np.where(
            v < REC2020_BETA,
            v * 4.5,
            REC2020_ALPHA * np.power(v, 0.45) - (REC2020_ALPHA - 1.0),
        )


class Rec2020Space(RGBSpace):
    """ITU-R BT.2020 primaries and transfer function (CSS ``rec2020``).

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, rec2020
    >>> srgb.to(rec2020, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.79, 0.23, 0.07]
    """

    name = "rec2020"
    matrix = D65_TO_D50 @ d65_white_matrix(REC2020_TO_XYZ_D65)
    decode = staticmethod(rec2020_decode)
    encode = staticmethod(rec2020_encode)


rec2020 = Rec2020Space()

__all__ = ["Rec2020Space", "rec2020"]
