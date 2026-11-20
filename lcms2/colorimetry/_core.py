"""Reference whites and argument checking shared by the colorimetry modules."""

from __future__ import annotations

import numpy as np

__all__ = ["D50", "D50_xyY", "D65", "D65_xyY"]


def _const(values) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    arr.flags.writeable = False
    return arr


#: ICC D50 illuminant as XYZ with ``Y = 1``.
D50 = _const([0.9642, 1.0, 0.8249])
#: D65 illuminant as XYZ with ``Y = 1``.
D65 = _const([0.95047, 1.0, 1.08883])
#: D50 as xyY.
D50_xyY = _const([0.3457, 0.3585, 1.0])
#: D65 as xyY.
D65_xyY = _const([0.3127, 0.3290, 1.0])


def _triple(x, name: str) -> np.ndarray:
    """Float64 array with a last axis of length 3, with a helpful error otherwise."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have a last axis of length 3 (shape (..., 3)), got shape {arr.shape}")
    return arr
