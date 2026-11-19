"""CIE XYZ relative to D50."""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace


class XYZD50Space(RGBSpace):
    """CIE XYZ relative to D50 (CSS ``xyz-d50``); identical to the PCS."""

    name = "xyz-d50"
    icc_colour_space_signature = b"XYZ "
    matrix = np.eye(3)


xyz_d50 = XYZD50Space()

__all__ = ["XYZD50Space", "xyz_d50"]
