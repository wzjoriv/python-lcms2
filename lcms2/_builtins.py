"""
Built-in profile names, aliases and white point resolution.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from . import _lcms2
from ._data import RGB_SPACES, WHITE_POINTS

__all__ = ["builtin_names", "create_builtin", "lookup_builtin", "resolve_white_point"]

WhitePoint = str | float | Sequence[float] | None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _rgb(name: str) -> Callable[[], Any]:
    def make() -> Any:
        s = RGB_SPACES[name]
        return _lcms2.profile_create_rgb(resolve_white_point(s["white_point"]), s["primaries"], [s["curve"]])

    return make


#: canonical name -> factory returning a profile capsule
_FACTORIES: dict[str, Callable[[], Any]] = {
    "sRGB": _lcms2.profile_create_srgb,
    "Lab": lambda: _lcms2.profile_create_lab(4, None),
    "Lab v2": lambda: _lcms2.profile_create_lab(2, None),
    "XYZ": _lcms2.profile_create_xyz,
    "NULL": _lcms2.profile_create_null,
    **{name: _rgb(name) for name in RGB_SPACES},
}

_ALIASES = {
    "adobergb": "Adobe RGB (1998)",
    "adobe": "Adobe RGB (1998)",
    "argb": "Adobe RGB (1998)",
    "p3": "Display P3",
    "displayp3": "Display P3",
    "dcip3": "DCI-P3",
    "rec2020": "Rec.2020",
    "bt2020": "Rec.2020",
    "rec709": "Rec.709",
    "bt709": "Rec.709",
    "prophoto": "ProPhoto RGB",
    "romm": "ProPhoto RGB",
    "romm rgb": "ProPhoto RGB",
    "ecirgb": "ECI RGB v2",
    "widegamut": "Wide Gamut RGB",
    "pal": "PAL/SECAM RGB",
    "secam": "PAL/SECAM RGB",
    "smptec": "SMPTE-C RGB",
    "ntsc": "NTSC RGB",
    "cielab": "Lab",
    "lab4": "Lab",
    "labv4": "Lab",
    "lab2": "Lab v2",
    "ciexyz": "XYZ",
}


def _build_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for name in _FACTORIES:
        if name.startswith("_"):
            continue  # exact-match only, e.g. "_sRGB"
        n = _norm(name)
        index[n] = name
        if n.endswith("rgb") and len(n) > 3:
            index.setdefault(n[:-3], name)
    for alias, name in _ALIASES.items():
        index[_norm(alias)] = name
    return index


_INDEX = _build_index()


def builtin_names() -> list[str]:
    """Canonical names of the built-in profiles."""
    return list(_FACTORIES)


def lookup_builtin(name: str) -> str | None:
    """The canonical built-in name for ``name`` (case-insensitive, lenient), or None."""
    if name in _FACTORIES:
        return name
    return _INDEX.get(_norm(name))


def create_builtin(name: str) -> Any:
    """Profile capsule of a canonical built-in name."""
    return _FACTORIES[name]()


def white_point_from_temperature(kelvin: float) -> tuple[float, float, float]:
    """xyY of the CIE daylight locus at a correlated colour temperature (4000-25000 K)."""
    t = float(kelvin)
    if not 4000 <= t <= 25000:
        raise ValueError(f"temperature must be between 4000 and 25000 K for the daylight locus, got {t:g}")
    t2, t3 = t * t, t * t * t
    if t <= 7000:
        x = -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063
    else:
        x = -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040
    y = -3.000 * x * x + 2.870 * x - 0.275
    return (x, y, 1.0)


def resolve_white_point(
    white_point: WhitePoint, degrees: int = 2, temperature: float | None = None
) -> tuple[float, float, float] | None:
    """xyY from an illuminant name ("D65"), an (x, y[, Y]) sequence or a temperature in kelvin.

    Raises
    ------
    ValueError
        For unknown illuminants or observers, listing the valid ones.
    """
    if temperature is not None:
        if white_point is not None:
            raise ValueError("give either white_point or temperature, not both")
        return white_point_from_temperature(temperature)
    if white_point is None:
        return None
    if isinstance(white_point, str):
        if degrees not in WHITE_POINTS:
            raise ValueError(f"degrees must be 2 or 10, got {degrees!r}")
        table = WHITE_POINTS[degrees]
        key = {k.upper(): k for k in table}.get(white_point.upper())
        if key is None:
            raise ValueError(
                f"unknown illuminant {white_point!r} for the {degrees}° observer; known: {', '.join(table)}"
            )
        return tuple(table[key]["xyY"])  # type: ignore[return-value]
    if isinstance(white_point, (int, float)):
        return white_point_from_temperature(white_point)
    values = tuple(float(v) for v in white_point)
    if len(values) == 2:
        values = values + (1.0,)
    if len(values) != 3:
        raise ValueError(f"white_point must be (x, y) or (x, y, Y), got {len(values)} values")
    return values  # type: ignore[return-value]
