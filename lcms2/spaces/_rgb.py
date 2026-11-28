"""
Base class for RGB spaces defined by a transfer function and primaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._lcms2 import mat3 as _mat3
from ._base import ColorSpace


def identity(v: np.ndarray, /) -> np.ndarray:
    """Return the values unchanged.

    Parameters
    ----------
    v : ndarray
        Values of any shape.

    Returns
    -------
    ndarray
        `v` itself.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces._rgb import identity
    >>> identity(np.array([0.25, 0.5])).tolist()
    [0.25, 0.5]
    """
    return v


def mat3(matrix: np.ndarray, values: Any) -> Any:
    """``values @ matrix.T`` for (..., 3) values; NumPy arrays take the BLAS-free, thread-safe C path."""
    if isinstance(values, (np.ndarray, list, tuple)):
        return _mat3(matrix, values)
    return values @ matrix.T


class RGBSpace(ColorSpace):
    """An RGB space: per-channel transfer function, then a 3x3 matrix to XYZ (D50).

    Subclasses set `matrix` and optionally `decode`/`encode`.

    Parameters
    ----------
    **kwargs
        Passed to `ColorSpace`.

    Attributes
    ----------
    matrix : ndarray, shape (3, 3)
        Maps linear RGB column vectors to XYZ (D50).
    decode : staticmethod
        Transfer function from device values to linear light.
    encode : staticmethod
        Transfer function from linear light to device values.

    Examples
    --------
    >>> from lcms2.spaces import srgb_linear
    >>> srgb_linear.to_xyz([1.0, 1.0, 1.0]).round(4).tolist()
    [0.9642, 1.0, 0.8249]
    """

    _abstract = True
    n_channels = 3
    matrix: np.ndarray = np.eye(3)

    decode = staticmethod(identity)
    encode = staticmethod(identity)

    def __init__(self, **kwargs: Any) -> None:
        """Create the space.

        Parameters
        ----------
        **kwargs
            Passed to `ColorSpace`.

        Examples
        --------
        >>> from lcms2.spaces import SRGBSpace
        >>> SRGBSpace(name="my-srgb").name
        'my-srgb'
        """
        self._inverse = np.linalg.inv(self.matrix)
        super().__init__(**kwargs)

    def to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        return mat3(self.matrix, self.decode(np.asarray(rgb, dtype=np.float64)))

    def from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return self.encode(mat3(self._inverse, xyz))


__all__ = ["RGBSpace", "mat3"]
