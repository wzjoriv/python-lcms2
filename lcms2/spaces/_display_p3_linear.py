"""Linear-light Display P3."""

from __future__ import annotations

from ._display_p3 import DisplayP3Space
from ._rgb import RGBSpace


class DisplayP3LinearSpace(RGBSpace):
    """Display P3 primaries with linear-light values."""

    name = "display-p3-linear"
    matrix = DisplayP3Space.matrix


display_p3_linear = DisplayP3LinearSpace()

__all__ = ["DisplayP3LinearSpace", "display_p3_linear"]
