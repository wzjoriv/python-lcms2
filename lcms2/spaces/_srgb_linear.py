"""
Linear-light sRGB.
"""

from __future__ import annotations

from ._rgb import RGBSpace
from ._srgb import SRGBSpace


class SRGBLinearSpace(RGBSpace):
    """sRGB primaries with linear-light values (CSS ``srgb-linear``).

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, srgb_linear
    >>> srgb.to(srgb_linear, [0.5, 0.5, 0.5]).round(2).tolist()
    [0.21, 0.21, 0.21]
    """

    name = "srgb-linear"
    matrix = SRGBSpace.matrix


srgb_linear = SRGBLinearSpace()

__all__ = ["SRGBLinearSpace", "srgb_linear"]
