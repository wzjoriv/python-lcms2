"""
Color spaces defined by conversions to and from CIE XYZ (D50).

Every space is a `ColorSpace` subclass with a ready-to-use instance named in
lower case (``srgb``, ``oklab``, ...). Convert with `ColorSpace.to`::

    >>> import numpy as np
    >>> from lcms2.spaces import srgb, oklab
    >>> srgb.to(oklab, np.array([1.0, 1.0, 1.0]))  # doctest: +SKIP

`ColorSpace.list` returns the registered names, and `ICCSpace` reads a space
from an ICC profile. Colorimetry helpers live in `lcms2.colorimetry`.
"""

from __future__ import annotations

from typing import Any

from ._a98_rgb import A98RGBSpace, a98_rgb
from ._base import ColorSpace, get
from ._cmyk import CMYKSpace, cmyk
from ._display_p3 import DisplayP3Space, display_p3
from ._display_p3_linear import DisplayP3LinearSpace, display_p3_linear
from ._gray import GraySpace, gray
from ._hsv import HSVSpace, hsv
from ._hwb import HWBSpace, hwb
from ._lch import LChSpace, lch
from ._loggray import LogGraySpace, loggray
from ._oklab import OKLabSpace, oklab
from ._oklch import OKLChSpace, oklch
from ._prophoto_rgb import ProPhotoRGBSpace, prophoto_rgb
from ._rec2020 import Rec2020Space, rec2020
from ._rec2100_hlg import Rec2100HLGSpace, rec2100_hlg
from ._rec2100_linear import Rec2100LinearSpace, rec2100_linear
from ._rec2100_pq import Rec2100PQSpace, rec2100_pq
from ._rgb import RGBSpace
from ._srgb import SRGBSpace, srgb
from ._srgb_linear import SRGBLinearSpace, srgb_linear
from ._xyz_d50 import XYZD50Space, xyz_d50
from ._xyz_d65 import XYZD65Space, xyz_d65

__all__ = [
    "ColorSpace",
    "get",
    "ICCSpace",
    "RGBSpace",
    "SRGBSpace",
    "srgb",
    "SRGBLinearSpace",
    "srgb_linear",
    "DisplayP3Space",
    "display_p3",
    "DisplayP3LinearSpace",
    "display_p3_linear",
    "A98RGBSpace",
    "a98_rgb",
    "ProPhotoRGBSpace",
    "prophoto_rgb",
    "Rec2020Space",
    "rec2020",
    "Rec2100LinearSpace",
    "rec2100_linear",
    "Rec2100PQSpace",
    "rec2100_pq",
    "Rec2100HLGSpace",
    "rec2100_hlg",
    "XYZD50Space",
    "xyz_d50",
    "XYZD65Space",
    "xyz_d65",
    "GraySpace",
    "gray",
    "LogGraySpace",
    "loggray",
    "OKLabSpace",
    "oklab",
    "OKLChSpace",
    "oklch",
    "LChSpace",
    "lch",
    "HSVSpace",
    "hsv",
    "HWBSpace",
    "hwb",
    "CMYKSpace",
    "cmyk",
]


def __getattr__(name: str) -> Any:
    if name == "ICCSpace":
        from ..icc import ICCSpace

        return ICCSpace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
