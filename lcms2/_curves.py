"""
One-dimensional tone curves (gamma, parametric and tabulated).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _lcms2

__all__ = ["ToneCurve"]

# Number of parameters each parametric curve type takes (the sign selects the inverse).
_PARAM_COUNTS = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7, 6: 4, 7: 5, 8: 5, 108: 1, 109: 1}


class ToneCurve:
    """A one-dimensional tone curve mapping 0..1 to 0..1.

    Build curves with the class methods :meth:`gamma`, :meth:`parametric`,
    :meth:`tabulated` or the presets :meth:`srgb`, :meth:`rec709` and :meth:`linear`,
    then evaluate them by calling the object on any array.

    Parameters
    ----------
    source : float or array_like
        A float builds a pure gamma curve; a 1-D ``uint16`` or float array builds a
        tabulated curve. (Internal code also passes an existing curve handle.)

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2._curves import ToneCurve
    >>> g = ToneCurve.gamma(2.0)
    >>> g(np.array([0.0, 0.5, 1.0])).round(4).tolist()
    [0.0, 0.25, 1.0]
    """

    def __init__(self, source: float | ArrayLike | Any):
        if isinstance(source, (int, float, np.floating, np.integer)) and not isinstance(source, bool):
            source = _lcms2.curve_gamma(float(source))
        elif not _is_capsule(source):
            source = _lcms2.curve_tabulated(_table_array(source))
        self._handle = source

    @classmethod
    def _from_handle(cls, handle: Any) -> ToneCurve:
        """Wrap a curve capsule returned by the extension."""
        self = cls.__new__(cls)
        self._handle = handle
        return self

    # ---- Construction ----
    @classmethod
    def gamma(cls, gamma: float) -> ToneCurve:
        """Pure power-law curve ``y = x ** gamma``.

        Parameters
        ----------
        gamma : float
            The exponent (2.2 is a typical display gamma).

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(float(ToneCurve.gamma(2.2)(0.5)), 4)
        0.2176
        """
        return cls(_lcms2.curve_gamma(float(gamma)))

    @classmethod
    def parametric(cls, type: int, params: ArrayLike) -> ToneCurve:
        """Parametric curve of one of the ICC / LittleCMS types.

        Parameters
        ----------
        type : int
            1 (gamma), 2 (CIE 122-1966), 3 (IEC 61966-3), 4 (sRGB-style), 5, 6, 7, 8,
            108 or 109 (sigmoids). A negative type is the inverse curve.
        params : sequence of float
            Parameters for the type; the required count is 1, 3, 4, 5, 7, 4, 5, 5, 1, 1
            for types 1-8, 108, 109 respectively.

        Raises
        ------
        ValueError
            For an unknown type or the wrong number of parameters.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> ToneCurve.parametric(1, [2.0])(0.5)
        np.float64(0.25)
        """
        type = int(type)
        if abs(type) not in _PARAM_COUNTS:
            raise ValueError(
                f"unsupported parametric curve type {type}; use one of {sorted(_PARAM_COUNTS)} "
                "(negative for the inverse)"
            )
        values = np.atleast_1d(np.asarray(params, dtype=np.float64))
        if values.ndim != 1 or values.size != _PARAM_COUNTS[abs(type)]:
            raise ValueError(
                f"parametric curve type {type} takes {_PARAM_COUNTS[abs(type)]} parameter(s), "
                f"got {values.size}"
            )
        return cls(_lcms2.curve_parametric(type, values))

    @classmethod
    def tabulated(cls, values: ArrayLike) -> ToneCurve:
        """Curve interpolated from a table of samples evenly spaced on 0..1.

        Parameters
        ----------
        values : array_like
            1-D ``uint16`` (0..65535) or floating point (0..1) samples, at least 2.

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2._curves import ToneCurve
        >>> ToneCurve.tabulated(np.array([0.0, 0.25, 1.0]))(0.5)
        np.float64(0.25)
        """
        return cls(_lcms2.curve_tabulated(_table_array(values)))

    @classmethod
    def srgb(cls) -> ToneCurve:
        """The sRGB transfer curve (encoded to linear).

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(float(ToneCurve.srgb()(0.5)), 4)
        0.214
        """
        return cls.parametric(4, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])

    @classmethod
    def rec709(cls) -> ToneCurve:
        """The Rec. 709 transfer curve (encoded to linear).

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(float(ToneCurve.rec709()(1.0)), 4)
        1.0
        """
        return cls.parametric(4, [1 / 0.45, 1 / 1.099, 0.099 / 1.099, 1 / 4.5, 0.081])

    @classmethod
    def linear(cls) -> ToneCurve:
        """The identity curve.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> ToneCurve.linear().is_linear
        True
        """
        return cls.gamma(1.0)

    # ---- Evaluation ----
    def __call__(self, x: ArrayLike) -> np.ndarray | np.generic:
        """Evaluate the curve over an array (or scalar) of any shape.

        Floating point input (0..1) gives ``float64`` output; ``uint16`` input gives
        ``uint16`` output. Other integer dtypes are rejected to avoid guessing a scale.

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2._curves import ToneCurve
        >>> ToneCurve.gamma(2.0)(np.array([[0.0, 1.0], [0.5, 0.5]])).shape
        (2, 2)
        """
        arr = np.asarray(x)
        if arr.dtype == np.uint16:
            out = _lcms2.curve_eval16(self._handle, arr)
        elif arr.dtype.kind in "iu":
            raise TypeError(
                f"cannot evaluate a curve on {arr.dtype} data; pass floats in 0..1 or a uint16 array "
                "(for example x.astype(np.float64) / 255)"
            )
        else:
            out = _lcms2.curve_eval(self._handle, arr)
        return out[()] if out.ndim == 0 else out

    # ---- Derived curves ----
    def inverse(self, n: int = 4096) -> ToneCurve:
        """Inverse curve, analytic when possible, otherwise sampled at ``n`` points.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(float(ToneCurve.gamma(2.0).inverse()(0.25)), 4)
        0.5
        """
        return ToneCurve(_lcms2.curve_reverse(self._handle, int(n)))

    def join(self, other: ToneCurve, n: int = 4096) -> ToneCurve:
        """Join two curves: ``x -> other.inverse()(self(x))``, sampled at ``n`` points.

        This is the curve that, applied after ``other``, reproduces ``self``; joining a
        curve with itself gives the identity. To apply one curve after another, use
        :meth:`then`.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(float(ToneCurve.gamma(2.0).join(ToneCurve.gamma(2.0))(0.3)), 3)
        0.3
        """
        return ToneCurve(_lcms2.curve_join(self._handle, other, int(n)))

    def then(self, other: ToneCurve, n: int = 4096) -> ToneCurve:
        """Compose two curves: ``x -> other(self(x))``, sampled at ``n`` points.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> c = ToneCurve.gamma(2.0).then(ToneCurve.gamma(1.5))
        >>> round(float(c(0.5)), 3)
        0.125
        """
        if not isinstance(other, ToneCurve):
            raise TypeError(f"expected a ToneCurve, got {type(other).__name__}")
        x = np.linspace(0.0, 1.0, int(n))
        return ToneCurve.tabulated(np.asarray(other(np.asarray(self(x), dtype=np.float64)), dtype=np.float64))

    def smooth(self, lam: float) -> ToneCurve:
        """Smoothed copy of the curve, for noisy tabulated curves (raises ``CMSError`` if the result
        would not be monotonic).

        Parameters
        ----------
        lam : float
            Smoothing strength; larger is smoother (0.1 to 1000 is typical).

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2._curves import ToneCurve
        >>> table = (np.linspace(0, 1, 64) ** 2.2 * 65535).astype(np.uint16)
        >>> ToneCurve.tabulated(table).smooth(1.0).is_monotonic
        True
        """
        return ToneCurve(_lcms2.curve_smooth(self._handle, float(lam)))

    def copy(self) -> ToneCurve:
        """Independent duplicate of the curve.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> c = ToneCurve.gamma(2.0)
        >>> c.copy() is c
        False
        """
        return ToneCurve(_lcms2.curve_copy(self._handle))

    def estimate_gamma(self, precision: float = 0.01) -> float:
        """Estimate the gamma that best matches the curve.

        Parameters
        ----------
        precision : float
            Maximum acceptable mean error. When no single gamma reaches it (as for the sRGB
            curve), the tolerance is relaxed step by step up to 0.5 before :class:`~lcms2.LCMSError` is
            raised.

        Examples
        --------
        >>> from lcms2._curves import ToneCurve
        >>> round(ToneCurve.gamma(2.2).estimate_gamma(), 1)
        2.2
        """
        tol = float(precision)
        while True:
            try:
                return _lcms2.curve_estimate_gamma(self._handle, tol)
            except _lcms2.LCMSError:
                if tol >= 0.5:
                    raise
                tol = min(tol * 2.0, 0.5)

    # ---- Properties ----
    @property
    def is_linear(self) -> bool:
        """True when the curve is the identity."""
        return _lcms2.curve_info(self._handle)["linear"]

    @property
    def is_monotonic(self) -> bool:
        """True when the curve never decreases (or never increases)."""
        return _lcms2.curve_info(self._handle)["monotonic"]

    @property
    def is_descending(self) -> bool:
        """True when the curve goes from high to low."""
        return _lcms2.curve_info(self._handle)["descending"]

    @property
    def is_multisegment(self) -> bool:
        """True when the curve is built from several segments."""
        return _lcms2.curve_info(self._handle)["multisegment"]

    @property
    def parametric_type(self) -> int:
        """Parametric type (negative for inverses); 0 for tabulated curves."""
        return _lcms2.curve_info(self._handle)["parametric_type"]

    @property
    def table(self) -> np.ndarray:
        """The curve as a ``uint16`` table (4096 entries for parametric curves)."""
        return _lcms2.curve_table(self._handle)

    def __repr__(self) -> str:
        info = _lcms2.curve_info(self._handle)
        kind = f"parametric type {info['parametric_type']}" if info["parametric_type"] else "tabulated"
        try:
            gamma = f", gamma~{self.estimate_gamma(0.05):.3g}"
        except _lcms2.LCMSError:
            gamma = ""
        return f"ToneCurve({kind}, {info['entries']} entries{gamma})"


def _is_capsule(obj: object) -> bool:
    return type(obj).__name__ == "PyCapsule"


def _table_array(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"curve values must be a 1-D array, got shape {arr.shape}")
    if arr.dtype == np.uint16 or arr.dtype.kind == "f":
        return arr
    if arr.dtype.kind in "iu":
        raise TypeError(
            f"integer curve values must be uint16 (0..65535), got {arr.dtype}; "
            "use .astype(np.uint16) or pass floats in 0..1"
        )
    raise TypeError(f"curve values must be a uint16 or float array, got {arr.dtype}")
