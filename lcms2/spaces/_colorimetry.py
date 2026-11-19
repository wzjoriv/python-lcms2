"""White points, CIE Lab conversion, Bradford adaptation and ICC Lab encodings."""

from __future__ import annotations

__all__ = [
    "D50",
    "D65",
    "bradford_matrix",
    "decode_lab_pcs",
    "encode_lab_pcs",
    "lab_to_xyz",
    "xyz_to_lab",
]

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249])
D65 = np.array([0.95047, 1.0, 1.08883])
# Shared defaults: read-only so no caller can corrupt them through an alias.
D50.setflags(write=False)
D65.setflags(write=False)

_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)


def bradford_matrix(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    """Build a Bradford chromatic adaptation matrix between two white points.

    Parameters
    ----------
    src_white : ndarray, shape (3,)
        XYZ tristimulus values of the source white point.
    dst_white : ndarray, shape (3,)
        XYZ tristimulus values of the destination white point.

    Returns
    -------
    ndarray, shape (3, 3)
        Matrix ``M`` with ``M @ xyz`` adapting from `src_white` to `dst_white`.
    """
    src_cone = _BRADFORD @ src_white
    dst_cone = _BRADFORD @ dst_white
    scale = np.diag(dst_cone / src_cone)
    return _BRADFORD_INV @ scale @ _BRADFORD


def _f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)


def _finv(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0 / 29.0))


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = D50) -> np.ndarray:
    """Convert CIE XYZ to CIE 1976 L*a*b*.

    Parameters
    ----------
    xyz : ndarray, shape (..., 3)
        CIE XYZ tristimulus values.
    white : ndarray, shape (3,), optional
        Reference white. Default D50.

    Returns
    -------
    ndarray, shape (..., 3)
        L*, a*, b* values.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    xr = xyz[..., 0] / white[0]
    yr = xyz[..., 1] / white[1]
    zr = xyz[..., 2] / white[2]
    fx, fy, fz = _f(xr), _f(yr), _f(zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray = D50) -> np.ndarray:
    """Convert CIE 1976 L*a*b* to CIE XYZ.

    Parameters
    ----------
    lab : ndarray, shape (..., 3)
        L*, a*, b* values.
    white : ndarray, shape (3,), optional
        Reference white. Default D50.

    Returns
    -------
    ndarray, shape (..., 3)
        CIE XYZ tristimulus values.
    """
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    x = _finv(fx) * white[0]
    y = _finv(fy) * white[1]
    z = _finv(fz) * white[2]
    return np.stack([x, y, z], axis=-1)


_LEGACY16_SCALE = 65535.0 / 65280.0


def decode_lab_pcs(encoded01: np.ndarray, legacy16: bool = False) -> np.ndarray:
    """Decode ICC Lab PCS values stored in [0, 1] into real L*a*b*.

    Parameters
    ----------
    encoded01 : ndarray, shape (..., 3)
        Encoded (L, a, b) values in [0, 1].
    legacy16 : bool, optional
        Use the 16-bit legacy encoding of lut16Type ('mft2') tags, where
        0xFF00 is L*=100 and a*=b*=127. Default False (v4 encoding).

    Returns
    -------
    ndarray, shape (..., 3)
        L*, a*, b* values.
    """
    e = np.asarray(encoded01, dtype=np.float64)
    if legacy16:
        e = e * _LEGACY16_SCALE
    L = e[..., 0] * 100.0
    a = e[..., 1] * 255.0 - 128.0
    b = e[..., 2] * 255.0 - 128.0
    return np.stack([L, a, b], axis=-1)


def encode_lab_pcs(lab: np.ndarray, legacy16: bool = False) -> np.ndarray:
    """Encode real L*a*b* values into ICC Lab PCS values in [0, 1].

    Parameters
    ----------
    lab : ndarray, shape (..., 3)
        L*, a*, b* values.
    legacy16 : bool, optional
        Use the 16-bit legacy encoding of lut16Type ('mft2') tags.
        Default False (v4 encoding).

    Returns
    -------
    ndarray, shape (..., 3)
        Encoded (L, a, b) values, clipped to [0, 1].
    """
    lab = np.asarray(lab, dtype=np.float64)
    e = np.stack(
        [lab[..., 0] / 100.0, (lab[..., 1] + 128.0) / 255.0, (lab[..., 2] + 128.0) / 255.0],
        axis=-1,
    )
    if legacy16:
        e = e / _LEGACY16_SCALE
    return np.clip(e, 0.0, 1.0)
