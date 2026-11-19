"""Colorimetry: color differences, white points, CIECAM02 and basic color space conversions.

Every function works on NumPy arrays whose last axis holds the three color components,
broadcasts between operands, and never modifies its inputs. XYZ values use the scale where
the white point has ``Y = 1`` (so :data:`D50` is about ``(0.9642, 1, 0.8249)``).
"""

from __future__ import annotations

import numpy as np

from . import _lcms2

__all__ = [
    "CIECAM02",
    "D50",
    "D50_xyY",
    "D65",
    "D65_xyY",
    "adapt",
    "delta_e",
    "lab_to_lch",
    "lab_to_xyz",
    "lch_to_lab",
    "temperature_from_white_point",
    "white_point_from_temperature",
    "xyy_to_xyz",
    "xyz_to_lab",
    "xyz_to_xyy",
]


def _const(values) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    arr.flags.writeable = False
    return arr


#: ICC D50 illuminant as XYZ with ``Y = 1``.
D50 = _const([0.9642, 1.0, 0.8249])
#: D65 illuminant as XYZ with ``Y = 1``.
D65 = _const([0.95047, 1.0, 1.08883])
#: D50 as xyY.
D50_xyY = _const([0.3457, 0.3585, 1.0])
#: D65 as xyY.
D65_xyY = _const([0.3127, 0.3290, 1.0])


def _triple(x, name: str) -> np.ndarray:
    """Float64 array with a last axis of length 3, with a helpful error otherwise."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have a last axis of length 3 (shape (..., 3)), got shape {arr.shape}")
    return arr


# -- color differences ------------------------------------------------------------------------

_DE_METHODS = {"76": 0, "94": 1, "2000": 2, "cmc": 3, "bfd": 4}
_DE_PARAMS = {2: ("Kl", "Kc", "Kh"), 3: ("l", "c")}
_DE_DEFAULTS = {2: (1.0, 1.0, 1.0), 3: (2.0, 1.0)}


def delta_e(lab1, lab2, method: str = "2000", **params) -> np.ndarray:
    """Color difference between CIELAB colors.

    Parameters
    ----------
    lab1, lab2 : array_like
        Lab values of shape ``(..., 3)``; the two operands broadcast against each other.
    method : {"2000", "76", "94", "cmc", "bfd"}
        CIEDE2000 (default), CIE76 (Euclidean), CIE94, CMC l:c or BFD. A leading "de" or
        "cie" is accepted ("de2000").
    **params
        ``Kl``, ``Kc``, ``Kh`` (all default 1) for ``"2000"``; ``l`` (2) and ``c`` (1) for
        ``"cmc"``.

    Returns
    -------
    numpy.ndarray
        Differences with the broadcast shape minus the last axis (a scalar for one pair).

    Raises
    ------
    ValueError
        For an unknown method or an array whose last axis is not 3.
    TypeError
        For a parameter the chosen method does not take.

    Examples
    --------
    >>> from lcms2.colorimetry import delta_e
    >>> round(float(delta_e([50, 2.6772, -79.7751], [50, 0, -82.7485])), 4)
    2.0425
    >>> float(delta_e([50, 0, 0], [53, 4, 0], method="76"))
    5.0
    """
    key = str(method).lower().removeprefix("cie").removeprefix("de").strip()
    if key not in _DE_METHODS:
        raise ValueError(f"unknown delta E method {method!r}; use one of {sorted(_DE_METHODS)}")
    code = _DE_METHODS[key]
    names = _DE_PARAMS.get(code, ())
    extra = set(params) - set(names)
    if extra:
        accepted = f"accepts {', '.join(names)}" if names else "takes no parameters"
        raise TypeError(f"delta E method {key!r} {accepted}; got {', '.join(sorted(extra))}")
    values = [params.get(n, d) for n, d in zip(names, _DE_DEFAULTS.get(code, ()), strict=False)]
    a, b = np.broadcast_arrays(_triple(lab1, "lab1"), _triple(lab2, "lab2"))
    out = _lcms2.delta_e(np.ascontiguousarray(a), np.ascontiguousarray(b), code, *map(float, values))
    return out[()] if out.ndim == 0 else out


# -- white points -----------------------------------------------------------------------------


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


# -- CIECAM02 ---------------------------------------------------------------------------------

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


# -- XYZ / Lab / LCh / xyY ----------------------------------------------------------------------

_EPS = (24.0 / 116.0) ** 3


def _white(white) -> np.ndarray:
    return D50 if white is None else _triple(white, "white")


def xyz_to_lab(xyz, white=None) -> np.ndarray:
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


def lab_to_xyz(lab, white=None) -> np.ndarray:
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


def lab_to_lch(lab) -> np.ndarray:
    """CIELAB to LCh (hue in degrees, 0..360).

    Examples
    --------
    >>> from lcms2.colorimetry import lab_to_lch
    >>> lab_to_lch([50, 0, 10]).round(3).tolist()
    [50.0, 10.0, 90.0]
    """
    lab = _triple(lab, "lab")
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), h], axis=-1)


def lch_to_lab(lch) -> np.ndarray:
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


def xyz_to_xyy(xyz) -> np.ndarray:
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


def xyy_to_xyz(xyy) -> np.ndarray:
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


def adapt(xyz, src_white, dst_white, method: str = "bradford") -> np.ndarray:
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
    key = str(method).lower().replace(" ", "_").replace("-", "_")
    if key not in _CAT:
        raise ValueError(f"unknown adaptation method {method!r}; use one of {sorted(_CAT)}")
    m = _CAT[key]
    gain = (m @ _triple(dst_white, "dst_white")) / (m @ _triple(src_white, "src_white"))
    full = np.linalg.inv(m) @ np.diag(gain) @ m
    return _triple(xyz, "xyz") @ full.T
