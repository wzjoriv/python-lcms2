"""
Linear-light Display P3.
"""

from __future__ import annotations

from ._display_p3 import DisplayP3Space
from ._rgb import RGBSpace


class DisplayP3LinearSpace(RGBSpace):
    """Display P3 primaries with linear-light values.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, display_p3_linear
    >>> srgb.to(display_p3_linear, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.82, 0.03, 0.02]
    """

    name = "display-p3-linear"
    matrix = DisplayP3Space.matrix


display_p3_linear = DisplayP3LinearSpace()

__all__ = ["DisplayP3LinearSpace", "display_p3_linear"]
