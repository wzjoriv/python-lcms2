"""
Conversions between XYZ, Lab, LCh and xyY, and chromatic adaptation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._core import D50, D50_xyY, _triple

__all__ = [
    "adapt",
    "adaptation_matrix",
    "lab_to_lch",
    "lab_to_xyz",
    "lch_to_lab",
    "xyy_to_xyz",
    "xyz_to_lab",
    "xyz_to_xyy",
]


_EPS = (24.0 / 116.0) ** 3


def _white(white: ArrayLike | None) -> np.ndarray:
    return D50 if white is None else _triple(white, "white")


def xyz_to_lab(xyz: ArrayLike, white: ArrayLike | None = None) -> np.ndarray:
    """XYZ to CIELAB (same formulas as LittleCMS).

    Parameters
    ----------
    xyz : array_like
        XYZ of shape ``(..., 3)`` with ``Y = 1`` at the white.
    white : array_like, optional
        Reference white XYZ; defaults to D50.

    Examples
    --------
    >>> from lcms2.colorimetry import D50, xyz_to_lab
    >>> xyz_to_lab(D50).round(6).tolist()
    [100.0, 0.0, 0.0]
    """
    t = _triple(xyz, "xyz") / _white(white)
    f = np.where(t > _EPS, np.cbrt(t), (841.0 / 108.0) * t + 16.0 / 116.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], axis=-1
    )


def lab_to_xyz(lab: ArrayLike, white: ArrayLike | None = None) -> np.ndarray:
    """CIELAB to XYZ; inverse of :func:`xyz_to_lab`.

    Examples
    --------
    >>> from lcms2.colorimetry import lab_to_xyz
    >>> lab_to_xyz([100, 0, 0]).round(4).tolist()
    [0.9642, 1.0, 0.8249]
    """
    lab = _triple(lab, "lab")
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    t = np.where(f > 24.0 / 116.0, f**3, (108.0 / 841.0) * (f - 16.0 / 116.0))
    return t * _white(white)


def lab_to_lch(lab: ArrayLike) -> np.ndarray:
    """CIELAB to LCh (hue in degrees, 0..360).

    Examples
    --------
    >>> from lcms2.colorimetry import lab_to_lch
    >>> lab_to_lch([50, 0, 10]).round(3).tolist()
    [50.0, 10.0, 90.0]
    """
    lab = _triple(lab, "lab")
    h = np.mod(np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) + 360.0, 360.0)
    return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), h], axis=-1)


def lch_to_lab(lch: ArrayLike) -> np.ndarray:
    """LCh to CIELAB; inverse of :func:`lab_to_lch`.

    Examples
    --------
    >>> from lcms2.colorimetry import lch_to_lab
    >>> lch_to_lab([50, 10, 90]).round(3).tolist()
    [50.0, 0.0, 10.0]
    """
    lch = _triple(lch, "lch")
    h = np.radians(lch[..., 2])
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(h), lch[..., 1] * np.sin(h)], axis=-1)


def xyz_to_xyy(xyz: ArrayLike) -> np.ndarray:
    """XYZ to xyY. Black (XYZ all zero) gets the D50 chromaticity.

    Examples
    --------
    >>> from lcms2.colorimetry import D65, xyz_to_xyy
    >>> xyz_to_xyy(D65).round(4).tolist()
    [0.3127, 0.329, 1.0]
    """
    xyz = _triple(xyz, "xyz")
    s = xyz.sum(axis=-1, keepdims=True)
    ok = s != 0
    xy = np.where(ok, xyz[..., :2] / np.where(ok, s, 1.0), D50_xyY[:2])
    return np.concatenate([xy, xyz[..., 1:2]], axis=-1)


def xyy_to_xyz(xyy: ArrayLike) -> np.ndarray:
    """xyY to XYZ. A zero ``y`` gives black.

    Examples
    --------
    >>> from lcms2.colorimetry import D65_xyY, xyy_to_xyz
    >>> xyy_to_xyz(D65_xyY).round(4).tolist()
    [0.9505, 1.0, 1.0891]
    """
    xyy = _triple(xyy, "xyy")
    x, y, big_y = xyy[..., 0], xyy[..., 1], xyy[..., 2]
    ok = y != 0
    safe = np.where(ok, y, 1.0)
    return np.stack(
        [
            np.where(ok, x * big_y / safe, 0.0),
            np.where(ok, big_y, 0.0),
            np.where(ok, (1.0 - x - y) * big_y / safe, 0.0),
        ],
        axis=-1,
    )


_CAT = {
    "bradford": np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]),
    "von_kries": np.array([[0.4002, 0.7076, -0.0808], [-0.2263, 1.1653, 0.0457], [0.0, 0.0, 0.9182]]),
    "xyz_scaling": np.eye(3),
}


def adaptation_matrix(src_white: ArrayLike, dst_white: ArrayLike, method: str = "bradford") -> np.ndarray:
    """The 3x3 chromatic adaptation matrix between two white points.

    Parameters
    ----------
    src_white, dst_white : array_like
        XYZ of the source and destination whites.
    method : {"bradford", "von_kries", "xyz_scaling"}
        Adaptation transform.

    Returns
    -------
    numpy.ndarray
        Matrix ``M`` of shape ``(3, 3)`` such that ``M @ xyz`` adapts a single color from
        `src_white` to `dst_white`.

    Raises
    ------
    ValueError
        If `method` is unknown.

    Examples
    --------
    >>> from lcms2.colorimetry import D50, D65, adaptation_matrix
    >>> (adaptation_matrix(D65, D50) @ D65).round(4).tolist()
    [0.9642, 1.0, 0.8249]
    """
    key = str(method).lower().replace(" ", "_").replace("-", "_")
    if key not in _CAT:
        raise ValueError(f"unknown adaptation method {method!r}; use one of {sorted(_CAT)}")
    m = _CAT[key]
    gain = (m @ _triple(dst_white, "dst_white")) / (m @ _triple(src_white, "src_white"))
    return np.linalg.inv(m) @ np.diag(gain) @ m


def adapt(xyz: ArrayLike, src_white: ArrayLike, dst_white: ArrayLike, method: str = "bradford") -> np.ndarray:
    """Chromatic adaptation of XYZ colors from one white point to another.

    Parameters
    ----------
    xyz : array_like
        XYZ of shape ``(..., 3)``.
    src_white, dst_white : array_like
        XYZ of the source and destination whites.
    method : {"bradford", "von_kries", "xyz_scaling"}
        Adaptation transform.

    Examples
    --------
    >>> from lcms2.colorimetry import D50, D65, adapt
    >>> adapt(D65, D65, D50).round(4).tolist()
    [0.9642, 1.0, 0.8249]
    """
    return _triple(xyz, "xyz") @ adaptation_matrix(src_white, dst_white, method).T
