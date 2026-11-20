"""
Base class for RGB spaces defined by a transfer function and primaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import ColorSpace


def identity(v: np.ndarray, /) -> np.ndarray:
    """Identity transfer function."""
    return v


class RGBSpace(ColorSpace):
    """An RGB space: per-channel transfer function, then a 3x3 matrix to XYZ (D50).

    Subclasses set `matrix` and optionally `decode`/`encode`.

    Attributes
    ----------
    matrix : ndarray, shape (3, 3)
        Maps linear RGB column vectors to XYZ (D50).
    decode : staticmethod
        Transfer function from device values to linear light.
    encode : staticmethod
        Transfer function from linear light to device values.
    """

    _abstract = True
    n_channels = 3
    matrix: np.ndarray = np.eye(3)

    decode = staticmethod(identity)
    encode = staticmethod(identity)

    def __init__(self, **kwargs: Any) -> None:
        self._inverse = np.linalg.inv(self.matrix)
        super().__init__(**kwargs)

    def to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Convert device RGB to CIE XYZ (D50)."""
        return self.decode(np.asarray(rgb, dtype=np.float64)) @ self.matrix.T

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to device RGB."""
        return self.encode(np.asarray(xyz, dtype=np.float64) @ self._inverse.T)


__all__ = ["RGBSpace"]
