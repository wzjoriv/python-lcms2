"""
BT.2100 with the HLG transfer function.
"""

from __future__ import annotations

import numpy as np

from ._rec2020 import Rec2020Space
from ._rgb import RGBSpace

HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * np.log(4.0 * HLG_A)


def hlg_encode(lin: np.ndarray) -> np.ndarray:
    """BT.2100 HLG OETF: scene linear light to signal."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            lin <= 1.0 / 12.0,
            np.sqrt(3.0 * lin),
            HLG_A * np.log(np.maximum(12.0 * lin - HLG_B, 1e-300)) + HLG_C,
        )


def hlg_decode(e: np.ndarray) -> np.ndarray:
    """BT.2100 HLG inverse OETF: signal to scene linear light."""
    with np.errstate(over="ignore"):
        return np.where(e <= 0.5, e * e / 3.0, (np.exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0)


class Rec2100HLGSpace(RGBSpace):
    """BT.2020 primaries with the HLG OETF (no OOTF); signal 1.0 maps to the
    reference white."""

    name = "rec2100-hlg"
    matrix = Rec2020Space.matrix
    decode = staticmethod(hlg_decode)
    encode = staticmethod(hlg_encode)


rec2100_hlg = Rec2100HLGSpace()

__all__ = ["Rec2100HLGSpace", "rec2100_hlg", "hlg_decode", "hlg_encode"]
