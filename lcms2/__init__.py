"""Color management for NumPy, built on LittleCMS2.

Examples
--------
>>> import numpy as np, lcms2
>>> lcms2.convert(np.array([255, 0, 0], np.uint8), "sRGB", "Lab", intent="relative").round(2)
array([54.29, 80.81, 69.89])
"""

__version__ = "0.2.0"
__author__ = "Josue N Rivera"

from . import _lcms2
from ._constants import ColorSpaceSignature, DeviceClass, Flag, Intent
from ._convert import convert
from ._data import RGB_SPACES, WHITE_POINTS
from ._errors import CMSError
from ._formats import FORMATS
from ._profile import Profile
from ._transform import Transform, get_alarm_codes, set_adaptation_state, set_alarm_codes


def get_version() -> str:
    """Version of the linked LittleCMS library, e.g. ``"2.20"``."""
    v = _lcms2.get_version()
    return f"{v // 1000}.{v % 1000 // 10}"


__all__ = [
    "CMSError",
    "ColorSpaceSignature",
    "DeviceClass",
    "FORMATS",
    "Flag",
    "Intent",
    "Profile",
    "RGB_SPACES",
    "Transform",
    "WHITE_POINTS",
    "convert",
    "get_alarm_codes",
    "get_version",
    "set_adaptation_state",
    "set_alarm_codes",
]
