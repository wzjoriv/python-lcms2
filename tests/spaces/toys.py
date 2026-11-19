"""Toy color spaces shared by the tests."""

from __future__ import annotations

import numpy as np

from lcms2.spaces import ColorSpace


class TwoChannelChromaSpace(ColorSpace):
    """Device (u, v) mapped to XYZ ``(u, 1, v)``."""

    name = "twochannelchroma"
    n_channels = 2

    def to_xyz(self, uv):
        uv = np.asarray(uv, dtype=np.float64)
        return np.stack([uv[..., 0], np.ones_like(uv[..., 0]), uv[..., 1]], axis=-1)

    def from_xyz(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        return np.stack([xyz[..., 0], xyz[..., 2]], axis=-1)
