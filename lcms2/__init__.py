"""
Color management for NumPy, built on LittleCMS2.

Examples
--------
>>> import numpy as np, lcms2
>>> lcms2.convert(np.array([255, 0, 0], np.uint8), "sRGB", "Lab", intent="relative").round(2)
array([54.29, 80.81, 69.89])
"""

__version__ = "0.3.0"
__author__ = "Josue N Rivera"

from . import _lcms2, cgats, colorimetry, colormap, icc, spaces
from ._constants import ColorSpaceSignature, DeviceClass, Flag, Intent
from ._convert import convert
from ._curves import ToneCurve
from ._data import RGB_SPACES, WHITE_POINTS
from ._errors import CMSError, ICCParseError, ICCWriteError
from ._formats import FORMATS
from ._parallel import get_threads, set_threads
from ._parse import parse_color, parse_colors, to_hex
from ._profile import Profile
from ._transform import (
    Transform,
    get_alarm_codes,
    set_adaptation_state,
    set_alarm_codes,
)
from .cgats import CGATS
from .colorimetry import CIECAM02, delta_e
from .colormap import Colormap
from .spaces import ColorSpace


def get_version() -> str:
    """Version of the linked LittleCMS library, e.g. ``"2.20"``."""
    v = _lcms2.get_version()
    return f"{v // 1000}.{v % 1000 // 10}"


__all__ = [
    "CGATS",
    "CIECAM02",
    "ColorSpace",
    "Colormap",
    "ToneCurve",
    "cgats",
    "colorimetry",
    "colormap",
    "delta_e",
    "icc",
    "parse_color",
    "parse_colors",
    "spaces",
    "to_hex",
    "CMSError",
    "ICCParseError",
    "ICCWriteError",
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
    "set_threads",
    "get_threads",
]
