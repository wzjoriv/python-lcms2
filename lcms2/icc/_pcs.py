"""ICC Lab PCS encodings (private)."""

from __future__ import annotations

import numpy as np

__all__ = ["decode_lab_pcs", "encode_lab_pcs"]

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
