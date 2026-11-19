"""Gamma-encoded grayscale."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import ColorSpace
from ._colorimetry import D50


class GraySpace(ColorSpace):
    """Grayscale with luminance ``Y = g ** gamma`` along a white point.

    Parameters
    ----------
    gamma : float, optional
        Exponent from device value to luminance. Default 2.2.
    white : array_like, shape (3,), optional
        XYZ of device value 1. Default D50 (copied, never aliased).
    **kwargs
        Passed to `ColorSpace`.

    Raises
    ------
    ValueError
        If `gamma` is not positive and finite, or `white` does not have shape ``(3,)``.
    """

    name = "gray"
    n_channels = 1

    def __init__(self, gamma: float = 2.2, white: Any = None, **kwargs: Any) -> None:
        if not (isinstance(gamma, (int, float, np.number)) and np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive number, got {gamma!r}")
        self.gamma = float(gamma)
        self.white = np.array(D50 if white is None else white, dtype=np.float64)
        if self.white.shape != (3,):
            raise ValueError(f"white must have shape (3,), got {self.white.shape}")
        super().__init__(**kwargs)

    def to_xyz(self, gray: np.ndarray) -> np.ndarray:
        """Convert gray values to CIE XYZ (D50). Negative values give NaN."""
        gray = np.asarray(gray, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            y = np.power(gray[..., 0], self.gamma)
        return y[..., None] * self.white

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to gray values from the Y channel."""
        xyz = np.asarray(xyz, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return np.power(xyz[..., 1:2] / self.white[1], 1.0 / self.gamma)


gray = GraySpace()

__all__ = ["GraySpace", "gray"]
