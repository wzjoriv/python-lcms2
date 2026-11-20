"""
Linear-light sRGB.
"""

from __future__ import annotations

from ._rgb import RGBSpace
from ._srgb import SRGBSpace


class SRGBLinearSpace(RGBSpace):
    """sRGB primaries with linear-light values (CSS ``srgb-linear``)."""

    name = "srgb-linear"
    matrix = SRGBSpace.matrix


srgb_linear = SRGBLinearSpace()

__all__ = ["SRGBLinearSpace", "srgb_linear"]
