"""
Display P3.
"""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50, d65_white_matrix, srgb_decode, srgb_encode

P3_TO_XYZ_D65 = np.array(
    [
        [0.48657095, 0.26566770, 0.19821728],
        [0.22897456, 0.69173852, 0.07928691],
        [0.00000000, 0.04511338, 1.04394437],
    ]
)


class DisplayP3Space(RGBSpace):
    """Display P3: DCI-P3 primaries, D65 white and the sRGB transfer function.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, display_p3
    >>> srgb.to(display_p3, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.92, 0.2, 0.14]
    """

    name = "display-p3"
    matrix = D65_TO_D50 @ d65_white_matrix(P3_TO_XYZ_D65)
    decode = staticmethod(srgb_decode)
    encode = staticmethod(srgb_encode)


display_p3 = DisplayP3Space()

__all__ = ["DisplayP3Space", "display_p3"]
