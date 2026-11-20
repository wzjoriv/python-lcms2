"""
CIE LCh(ab).
"""

from __future__ import annotations

import numpy as np

from ..colorimetry import D50, lab_to_xyz, xyz_to_lab
from ._base import ColorSpace


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Convert (L, a, b) to (L, C, h) with hue as a fraction of a turn."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return np.stack([L, np.hypot(a, b), np.mod(np.arctan2(b, a) / (2 * np.pi), 1.0)], axis=-1)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """Convert (L, C, h) with hue as a fraction of a turn to (L, a, b)."""
    L, c, h = lch[..., 0], lch[..., 1], lch[..., 2]
    angle = 2 * np.pi * h
    return np.stack([L, c * np.cos(angle), c * np.sin(angle)], axis=-1)


class LChSpace(ColorSpace):
    """CIE LCh(ab) relative to D50 (CSS ``lch``); hue as a fraction of a turn."""

    name = "lch"
    icc_colour_space_signature = b"3CLR"
    circular_device_channels = frozenset({2})
    device_domain = [(0.0, 100.0), (0.0, 150.0), (0.0, 1.0)]

    def to_xyz(self, lch: np.ndarray) -> np.ndarray:
        """Convert LCh to CIE XYZ (D50)."""
        return lab_to_xyz(lch_to_lab(np.asarray(lch, dtype=np.float64)), white=D50)

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ (D50) to LCh."""
        return lab_to_lch(xyz_to_lab(np.asarray(xyz, dtype=np.float64), white=D50))


lch = LChSpace()

__all__ = ["LChSpace", "lch", "lab_to_lch", "lch_to_lab"]
