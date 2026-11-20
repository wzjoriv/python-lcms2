"""
Reference tables: CIE illuminants and RGB colour spaces (CSV files next to this module).
"""

from __future__ import annotations

import csv
from pathlib import Path

_HERE = Path(__file__).parent


def _rows(name: str) -> list[dict[str, str]]:
    with open(_HERE / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _illuminants() -> dict[int, dict[str, dict]]:
    out: dict[int, dict[str, dict]] = {2: {}, 10: {}}
    for r in _rows("illuminants.csv"):
        for degrees, x, y in ((2, r["x2"], r["y2"]), (10, r["x10"], r["y10"])):
            if x:
                out[degrees][r["name"]] = {
                    "xyY": (float(x), float(y), 1.0),
                    "CCT": float(r["cct"]),
                    "description": r["description"],
                }
    return out


def _rgb_spaces() -> dict[str, dict]:
    out = {}
    for r in _rows("rgb_spaces.csv"):
        white = r["white"]
        if " " in white:
            white = tuple(float(v) for v in white.split()) + (1.0,)
        out[r["name"]] = {
            "white_point": white,
            "primaries": tuple(
                (float(r[f"{c}_x"]), float(r[f"{c}_y"]), 1.0) for c in ("red", "green", "blue")
            ),
            "curve": (int(r["curve_type"]), tuple(float(v) for v in r["curve_params"].split())),
        }
    return out


#: ``WHITE_POINTS[degrees][name]`` -> ``{"xyY", "CCT", "description"}`` for 2° and 10° observers.
WHITE_POINTS = _illuminants()

#: ``RGB_SPACES[name]`` -> ``{"white_point", "primaries", "curve"}``; curve is a LittleCMS
#: parametric ``(type, params)`` pair.
RGB_SPACES = _rgb_spaces()

__all__ = ["RGB_SPACES", "WHITE_POINTS"]
