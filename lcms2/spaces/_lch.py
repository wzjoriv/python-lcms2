"""
CIE LCh(ab).
"""

from __future__ import annotations

import numpy as np

from ..colorimetry import D50, lab_to_lch, lab_to_xyz, lch_to_lab, xyz_to_lab
from ._base import ColorSpace


class LChSpace(ColorSpace):
    """CIE LCh(ab) relative to D50 (CSS ``lch``); hue in degrees."""

    name = "lch"
    icc_colour_space_signature = b"3CLR"
    circular_device_channels = frozenset({2})
    device_domain = [(0.0, 100.0), (0.0, 150.0), (0.0, 360.0)]

    def to_xyz(self, lch: np.ndarray) -> np.ndarray:
        """Convert LCh to CIE XYZ (D50)."""
        return lab_to_xyz(lch_to_lab(np.asarray(lch, dtype=np.float64)), white=D50)

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to LCh."""
        return lab_to_lch(xyz_to_lab(np.asarray(xyz, dtype=np.float64), white=D50))


lch = LChSpace()

__all__ = ["LChSpace", "lch", "lab_to_lch", "lch_to_lab"]
