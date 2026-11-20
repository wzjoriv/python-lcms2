"""
White points and correlated color temperature.
"""

from __future__ import annotations

import numpy as np

from .. import _lcms2
from ._core import _triple

__all__ = ["temperature_from_white_point", "white_point_from_temperature"]


def white_point_from_temperature(kelvin) -> np.ndarray:
    """Chromaticity of the CIE daylight illuminant at a correlated color temperature.

    Parameters
    ----------
    kelvin : float or array_like
        Temperature in kelvin, 4000 to 25000.

    Returns
    -------
    numpy.ndarray
        xyY with ``Y = 1``; shape ``(3,)`` for a scalar, ``(..., 3)`` for an array.

    Raises
    ------
    ValueError
        If a temperature is outside 4000-25000 K.

    Examples
    --------
    >>> from lcms2.colorimetry import white_point_from_temperature
    >>> white_point_from_temperature(6504).round(3).tolist()
    [0.313, 0.329, 1.0]
    """
    return _lcms2.white_from_temp(np.asarray(kelvin, dtype=np.float64))


def temperature_from_white_point(xyy) -> np.ndarray:
    """Correlated color temperature of a white point.

    Parameters
    ----------
    xyy : array_like
        xyY of shape ``(..., 3)``, or just ``(x, y)`` of shape ``(..., 2)``.

    Returns
    -------
    numpy.ndarray
        Temperature in kelvin (a scalar for a single white point).

    Examples
    --------
    >>> from lcms2.colorimetry import D65_xyY, temperature_from_white_point
    >>> round(float(temperature_from_white_point(D65_xyY)))
    6504
    """
    arr = np.asarray(xyy, dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        arr = np.concatenate([arr, np.ones(arr.shape[:-1] + (1,))], axis=-1)
    out = _lcms2.temp_from_white(_triple(arr, "xyy"))
    return out[()] if out.ndim == 0 else out
