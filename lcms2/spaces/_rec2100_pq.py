"""
BT.2100 with the PQ transfer function.
"""

from __future__ import annotations

import numpy as np

from ._rec2020 import Rec2020Space
from ._rgb import RGBSpace

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0


def pq_decode(e: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 EOTF: signal to normalized linear light (1.0 = 10000 cd/m2)."""
    with np.errstate(invalid="ignore"):
        ep = np.power(e, 1.0 / PQ_M2)
        return np.power(np.maximum(ep - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * ep), 1.0 / PQ_M1)


def pq_encode(y: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 inverse EOTF: normalized linear light to signal."""
    with np.errstate(invalid="ignore"):
        yp = np.power(y, PQ_M1)
        return np.power((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp), PQ_M2)


class Rec2100PQSpace(RGBSpace):
    """BT.2020 primaries with the PQ curve; signal 1.0 maps to the reference
    white. Negative values give NaN."""

    name = "rec2100-pq"
    matrix = Rec2020Space.matrix
    decode = staticmethod(pq_decode)
    encode = staticmethod(pq_encode)


rec2100_pq = Rec2100PQSpace()

__all__ = ["Rec2100PQSpace", "rec2100_pq", "pq_decode", "pq_encode"]
