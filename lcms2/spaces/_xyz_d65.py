"""
CIE XYZ relative to D65.
"""

from __future__ import annotations

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50


class XYZD65Space(RGBSpace):
    """CIE XYZ relative to D65 (CSS ``xyz-d65``), Bradford-adapted to D50."""

    name = "xyz-d65"
    icc_colour_space_signature = b"XYZ "
    matrix = D65_TO_D50


xyz_d65 = XYZD65Space()

__all__ = ["XYZD65Space", "xyz_d65"]
