"""
CIE XYZ relative to D65.
"""

from __future__ import annotations

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50


class XYZD65Space(RGBSpace):
    """CIE XYZ relative to D65 (CSS ``xyz-d65``), Bradford-adapted to D50.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import xyz_d50, xyz_d65
    >>> xyz_d65.to(xyz_d50, [0.95, 1.0, 1.09]).round(2).tolist()
    [0.96, 1.0, 0.83]
    """

    name = "xyz-d65"
    icc_colour_space_signature = b"XYZ "
    matrix = D65_TO_D50


xyz_d65 = XYZD65Space()

__all__ = ["XYZD65Space", "xyz_d65"]
