"""Linear-light BT.2100."""

from __future__ import annotations

from ._rec2020 import Rec2020Space
from ._rgb import RGBSpace


class Rec2100LinearSpace(RGBSpace):
    """BT.2020 primaries with linear light; 1.0 is the reference white."""

    name = "rec2100-linear"
    matrix = Rec2020Space.matrix


rec2100_linear = Rec2100LinearSpace()

__all__ = ["Rec2100LinearSpace", "rec2100_linear"]
