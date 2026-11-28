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
    """Apply the BT.2100 HLG OETF from scene linear light to signal.

    Parameters
    ----------
    lin : ndarray
        Scene linear light.

    Returns
    -------
    ndarray
        HLG signal, 0..1.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._rec2100_hlg import hlg_encode
    >>> hlg_encode(np.array([0.0, 1.0])).round(4).tolist()
    [0.0, 1.0]
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            lin <= 1.0 / 12.0,
            np.sqrt(3.0 * lin),
            HLG_A * np.log(np.maximum(12.0 * lin - HLG_B, 1e-300)) + HLG_C,
        )


def hlg_decode(e: np.ndarray) -> np.ndarray:
    """Apply the BT.2100 HLG inverse OETF from signal to scene linear light.

    Parameters
    ----------
    e : ndarray
        HLG signal, 0..1.

    Returns
    -------
    ndarray
        Scene linear light.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._rec2100_hlg import hlg_decode
    >>> hlg_decode(np.array([0.5, 1.0])).round(4).tolist()
    [0.0833, 1.0]
    """
    with np.errstate(over="ignore"):
        return np.where(e <= 0.5, e * e / 3.0, (np.exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0)


class Rec2100HLGSpace(RGBSpace):
    """BT.2020 primaries with the HLG OETF (no OOTF).

    Signal 1.0 maps to the reference white.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Examples
    --------
    >>> from lcms2.spaces import srgb, rec2100_hlg
    >>> srgb.to(rec2100_hlg, [1.0, 0.0, 0.0]).round(2).tolist()
    [0.91, 0.46, 0.22]
    """

    name = "rec2100-hlg"
    matrix = Rec2020Space.matrix
    decode = staticmethod(hlg_decode)
    encode = staticmethod(hlg_encode)


rec2100_hlg = Rec2100HLGSpace()

__all__ = ["Rec2100HLGSpace", "rec2100_hlg", "hlg_decode", "hlg_encode"]
