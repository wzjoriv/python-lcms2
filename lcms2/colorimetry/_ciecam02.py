"""
The CIECAM02 color appearance model.
"""

from __future__ import annotations

import numpy as np

from .. import _lcms2
from ._core import _triple

__all__ = ["CIECAM02"]

_SURROUNDS = {"average": 1, "dim": 2, "dark": 3, "cutsheet": 4}


class CIECAM02:
    """CIECAM02 color appearance model under fixed viewing conditions.

    Parameters
    ----------
    white_point : array_like
        XYZ of the adopted white, with ``Y = 1`` (for example :data:`D65`).
    adapting_luminance : float
        Luminance of the adapting field ``La`` in cd/m2 (typically 20% of the white's).
    background : float
        Relative luminance of the background ``Yb`` in percent of the white (20 is typical).
    surround : {"average", "dim", "dark", "cutsheet"}
        Viewing surround.
    degree : float or "auto"
        Degree of adaptation D in 0..1, or ``"auto"`` to compute it from ``La`` and the surround.

    Examples
    --------
    >>> from lcms2.colorimetry import CIECAM02, D65
    >>> cam = CIECAM02(D65, adapting_luminance=64, background=20)
    >>> round(float(cam.forward(D65)[0]), 2)
    100.0
    """

    def __init__(
        self,
        white_point,
        adapting_luminance: float = 64.0,
        background: float = 20.0,
        surround: str = "average",
        degree="auto",
    ):
        white = np.asarray(white_point, dtype=np.float64)
        if white.shape != (3,) or white[1] <= 0:
            raise ValueError(f"white_point must be an XYZ triple with Y > 0, got {white_point!r}")
        key = str(surround).lower()
        if key not in _SURROUNDS:
            raise ValueError(f"unknown surround {surround!r}; use one of {sorted(_SURROUNDS)}")
        if isinstance(degree, str):
            if degree != "auto":
                raise ValueError(f"degree must be a number in 0..1 or 'auto', got {degree!r}")
            d = -1.0
        else:
            d = float(degree)
            if not 0.0 <= d <= 1.0:
                raise ValueError(f"degree of adaptation must be in 0..1, got {d}")
        self.white_point = white / white[1]
        self.adapting_luminance = float(adapting_luminance)
        self.background = float(background)
        self.surround = key
        self.degree = "auto" if d < 0 else d
        self._args = (
            self.white_point * 100.0,
            self.background,
            self.adapting_luminance,
            _SURROUNDS[key],
            d,
        )

    def _run(self, values, name, reverse):
        arr = _triple(values, name)
        w, yb, la, surround, d = self._args
        return _lcms2.cam02(np.ascontiguousarray(arr), w, yb, la, surround, d, reverse)

    def forward(self, xyz) -> np.ndarray:
        """XYZ (``Y = 1`` scale) to JCh (lightness, chroma, hue in degrees).

        Examples
        --------
        >>> from lcms2.colorimetry import CIECAM02, D65
        >>> CIECAM02(D65, 64).forward([[0.2, 0.2, 0.2], D65]).shape
        (2, 3)
        """
        return self._run(np.asarray(xyz, dtype=np.float64) * 100.0, "xyz", False)

    def reverse(self, jch) -> np.ndarray:
        """JCh back to XYZ (``Y = 1`` scale).

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2.colorimetry import CIECAM02, D65
        >>> cam = CIECAM02(D65, 64)
        >>> bool(np.allclose(cam.reverse(cam.forward([0.2, 0.3, 0.4])), [0.2, 0.3, 0.4], atol=1e-4))
        True
        """
        return self._run(jch, "jch", True) / 100.0

    def __repr__(self) -> str:
        return (
            f"CIECAM02(white_point={self.white_point.round(4).tolist()}, "
            f"adapting_luminance={self.adapting_luminance:g}, background={self.background:g}, "
            f"surround={self.surround!r}, degree={self.degree!r})"
        )
