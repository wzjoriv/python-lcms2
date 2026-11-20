"""
Colorimetry: color differences, white points, CIECAM02 and basic color space conversions.

Every function works on NumPy arrays whose last axis holds the three color components,
broadcasts between operands, and never modifies its inputs. XYZ values use the scale where
the white point has ``Y = 1`` (so :data:`D50` is about ``(0.9642, 1, 0.8249)``).
"""

from ._ciecam02 import CIECAM02
from ._convert import (
    adapt,
    adaptation_matrix,
    lab_to_lch,
    lab_to_xyz,
    lch_to_lab,
    xyy_to_xyz,
    xyz_to_lab,
    xyz_to_xyy,
)
from ._core import D50, D65, D50_xyY, D65_xyY
from ._delta_e import delta_e
from ._whites import temperature_from_white_point, white_point_from_temperature

__all__ = [
    "CIECAM02",
    "D50",
    "D50_xyY",
    "D65",
    "D65_xyY",
    "adapt",
    "adaptation_matrix",
    "delta_e",
    "lab_to_lch",
    "lab_to_xyz",
    "lch_to_lab",
    "temperature_from_white_point",
    "white_point_from_temperature",
    "xyy_to_xyz",
    "xyz_to_lab",
    "xyz_to_xyy",
]
