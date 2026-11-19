"""Logarithmic grayscale."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import ColorSpace
from ._colorimetry import D50


class LogGraySpace(ColorSpace):
    """Grayscale with luminance ``Y = black * (1 / black) ** g`` along a white point.

    Parameters
    ----------
    black : float, optional
        Luminance of device value 0. Default 0.001.
    white : array_like, shape (3,), optional
        XYZ of device value 1. Default D50 (copied, never aliased).
    **kwargs
        Passed to `ColorSpace`.

    Raises
    ------
    ValueError
        If `black` is not in ``(0, 1)`` or `white` does not have shape ``(3,)``.
    """

    name = "loggray"
    n_channels = 1

    def __init__(self, black: float = 0.001, white: Any = None, **kwargs: Any) -> None:
        if not (isinstance(black, (int, float, np.number)) and 0.0 < black < 1.0):
            raise ValueError(f"black must be a number in (0, 1), got {black!r}")
        self.black = float(black)
        self.white = np.array(D50 if white is None else white, dtype=np.float64)
        if self.white.shape != (3,):
            raise ValueError(f"white must have shape (3,), got {self.white.shape}")
        super().__init__(**kwargs)

    def to_xyz(self, gray: np.ndarray) -> np.ndarray:
        """Convert gray values to CIE XYZ (D50)."""
        gray = np.asarray(gray, dtype=np.float64)
        y = self.black * np.power(1.0 / self.black, gray[..., 0])
        return y[..., None] * self.white

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to gray values. Non-positive Y gives NaN or -inf."""
        xyz = np.asarray(xyz, dtype=np.float64)
        y = xyz[..., 1:2] / self.white[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(y / self.black) / np.log(1.0 / self.black)


loggray = LogGraySpace()

__all__ = ["LogGraySpace", "loggray"]
