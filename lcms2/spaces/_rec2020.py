"""ITU-R BT.2020."""

from __future__ import annotations

import numpy as np

from ._rgb import RGBSpace
from ._srgb import D65_TO_D50, d65_white_matrix

REC2020_TO_XYZ_D65 = np.array(
    [
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ]
)

REC2020_ALPHA = 1.099296826809442
REC2020_BETA = 0.018053968510807


def rec2020_decode(v: np.ndarray) -> np.ndarray:
    """BT.2020 transfer function, device to linear."""
    with np.errstate(invalid="ignore"):
        return np.where(
            v < REC2020_BETA * 4.5,
            v / 4.5,
            np.power((v + REC2020_ALPHA - 1.0) / REC2020_ALPHA, 1.0 / 0.45),
        )


def rec2020_encode(v: np.ndarray) -> np.ndarray:
    """BT.2020 transfer function, linear to device."""
    with np.errstate(invalid="ignore"):
        return np.where(
            v < REC2020_BETA,
            v * 4.5,
            REC2020_ALPHA * np.power(v, 0.45) - (REC2020_ALPHA - 1.0),
        )


class Rec2020Space(RGBSpace):
    """ITU-R BT.2020 primaries and transfer function (CSS ``rec2020``)."""

    name = "rec2020"
    matrix = D65_TO_D50 @ d65_white_matrix(REC2020_TO_XYZ_D65)
    decode = staticmethod(rec2020_decode)
    encode = staticmethod(rec2020_encode)


rec2020 = Rec2020Space()

__all__ = ["Rec2020Space", "rec2020"]
