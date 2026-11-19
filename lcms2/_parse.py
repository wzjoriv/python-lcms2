"""Parse colours written in any common form into sRGB floats in 0..1."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

__all__ = ["parse_color", "parse_colors", "to_hex"]

_NAMES_TEXT = """
aliceblue:F0F8FF antiquewhite:FAEBD7 aqua:00FFFF aquamarine:7FFFD4 azure:F0FFFF beige:F5F5DC
bisque:FFE4C4 black:000000 blanchedalmond:FFEBCD blue:0000FF blueviolet:8A2BE2 brown:A52A2A
burlywood:DEB887 cadetblue:5F9EA0 chartreuse:7FFF00 chocolate:D2691E coral:FF7F50
cornflowerblue:6495ED cornsilk:FFF8DC crimson:DC143C cyan:00FFFF darkblue:00008B darkcyan:008B8B
darkgoldenrod:B8860B darkgray:A9A9A9 darkgreen:006400 darkgrey:A9A9A9 darkkhaki:BDB76B
darkmagenta:8B008B darkolivegreen:556B2F darkorange:FF8C00 darkorchid:9932CC darkred:8B0000
darksalmon:E9967A darkseagreen:8FBC8F darkslateblue:483D8B darkslategray:2F4F4F
darkslategrey:2F4F4F darkturquoise:00CED1 darkviolet:9400D3 deeppink:FF1493 deepskyblue:00BFFF
dimgray:696969 dimgrey:696969 dodgerblue:1E90FF firebrick:B22222 floralwhite:FFFAF0
forestgreen:228B22 fuchsia:FF00FF gainsboro:DCDCDC ghostwhite:F8F8FF gold:FFD700
goldenrod:DAA520 gray:808080 green:008000 greenyellow:ADFF2F grey:808080 honeydew:F0FFF0
hotpink:FF69B4 indianred:CD5C5C indigo:4B0082 ivory:FFFFF0 khaki:F0E68C lavender:E6E6FA
lavenderblush:FFF0F5 lawngreen:7CFC00 lemonchiffon:FFFACD lightblue:ADD8E6 lightcoral:F08080
lightcyan:E0FFFF lightgoldenrodyellow:FAFAD2 lightgray:D3D3D3 lightgreen:90EE90 lightgrey:D3D3D3
lightpink:FFB6C1 lightsalmon:FFA07A lightseagreen:20B2AA lightskyblue:87CEFA
lightslategray:778899 lightslategrey:778899 lightsteelblue:B0C4DE lightyellow:FFFFE0 lime:00FF00
limegreen:32CD32 linen:FAF0E6 magenta:FF00FF maroon:800000 mediumaquamarine:66CDAA
mediumblue:0000CD mediumorchid:BA55D3 mediumpurple:9370DB mediumseagreen:3CB371
mediumslateblue:7B68EE mediumspringgreen:00FA9A mediumturquoise:48D1CC mediumvioletred:C71585
midnightblue:191970 mintcream:F5FFFA mistyrose:FFE4E1 moccasin:FFE4B5 navajowhite:FFDEAD
navy:000080 oldlace:FDF5E6 olive:808000 olivedrab:6B8E23 orange:FFA500 orangered:FF4500
orchid:DA70D6 palegoldenrod:EEE8AA palegreen:98FB98 paleturquoise:AFEEEE palevioletred:DB7093
papayawhip:FFEFD5 peachpuff:FFDAB9 peru:CD853F pink:FFC0CB plum:DDA0DD powderblue:B0E0E6
purple:800080 rebeccapurple:663399 red:FF0000 rosybrown:BC8F8F royalblue:4169E1
saddlebrown:8B4513 salmon:FA8072 sandybrown:F4A460 seagreen:2E8B57 seashell:FFF5EE sienna:A0522D
silver:C0C0C0 skyblue:87CEEB slateblue:6A5ACD slategray:708090 slategrey:708090 snow:FFFAFA
springgreen:00FF7F steelblue:4682B4 tan:D2B48C teal:008080 thistle:D8BFD8 tomato:FF6347
turquoise:40E0D0 violet:EE82EE wheat:F5DEB3 white:FFFFFF whitesmoke:F5F5F5 yellow:FFFF00
yellowgreen:9ACD32
"""
CSS_NAMES = {
    key: tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    for key, value in (item.split(":") for item in _NAMES_TEXT.split())
}

_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
_FUNC = re.compile(r"^rgba?\((.*)\)$", re.IGNORECASE)
_HEX = re.compile(r"^#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$", re.IGNORECASE)


def _channel(token: str, index: int, text: str) -> float:
    """One CSS function argument: a number (0..255 for colour, 0..1 for alpha) or a percentage."""
    token = token.strip()
    percent = token.endswith("%")
    body = token[:-1] if percent else token
    if not re.fullmatch(_NUM, body, re.IGNORECASE):
        raise ValueError(f"cannot parse colour {text!r}: bad component {token!r}")
    x = float(body)
    if percent:
        x /= 100.0
    elif index < 3:
        x /= 255.0
    return float(min(max(x, 0.0), 1.0))


def _parse_string(text: str) -> np.ndarray:
    s = text.strip()
    if _HEX.match(s):
        digits = s[1:]
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        return np.array([int(digits[i : i + 2], 16) for i in range(0, len(digits), 2)]) / 255.0
    m = _FUNC.match(s)
    if m:
        parts = [p for p in re.split(r"[\s,/]+", m.group(1).strip()) if p]
        if len(parts) not in (3, 4):
            raise ValueError(f"cannot parse colour {text!r}: expected 3 or 4 components, got {len(parts)}")
        return np.array([_channel(p, i, text) for i, p in enumerate(parts)])
    key = s.lower().replace(" ", "")
    if key in CSS_NAMES:
        return np.array(CSS_NAMES[key])
    raise ValueError(
        f"cannot parse colour {text!r}: expected '#rgb', '#rrggbb[aa]', a CSS colour name or 'rgb(r, g, b)'"
    )


def _parse_numeric(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape[0] not in (3, 4):
        raise ValueError(f"cannot parse colour {value!r}: expected 3 or 4 components, got shape {arr.shape}")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"cannot parse colour {value!r}: components must be numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"cannot parse colour {value!r}: components must be finite")
    if arr.dtype.kind in "iu":
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError(f"cannot parse colour {value!r}: integer components must be in 0..255")
        return arr.astype(np.float64) / 255.0
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError(
            f"cannot parse colour {value!r}: float components must be in 0..1 (use integers for 0..255)"
        )
    return arr.astype(np.float64)


def parse_color(value: Any) -> np.ndarray:
    """Parse one colour into sRGB floats in 0..1.

    Parameters
    ----------
    value : str or sequence or ndarray
        ``"#rgb"``, ``"#rgba"``, ``"#rrggbb"``, ``"#rrggbbaa"``, a CSS4 colour name
        (case-insensitive), ``"rgb(...)"`` / ``"rgba(...)"`` (numbers 0..255 or
        percentages), or 3 or 4 numbers: floats in 0..1, or integers in 0..255.

    Returns
    -------
    ndarray, shape (3,) or (4,)
        Floats in 0..1; the fourth entry, when present, is alpha.

    Raises
    ------
    ValueError
        If the value is not a recognisable colour; the message names the value.

    Examples
    --------
    >>> from lcms2._parse import parse_color
    >>> parse_color("#ff8000").round(3)
    array([1.   , 0.502, 0.   ])
    >>> parse_color("RebeccaPurple").round(3)
    array([0.4, 0.2, 0.6])
    >>> parse_color("rgb(100%, 0%, 0%)")
    array([1., 0., 0.])
    >>> parse_color((255, 0, 0))
    array([1., 0., 0.])
    """
    if isinstance(value, str):
        return _parse_string(value)
    if isinstance(value, (bytes, dict, set)):
        raise ValueError(f"cannot parse colour {value!r}: unsupported type {type(value).__name__}")
    try:
        return _parse_numeric(value)
    except ValueError:
        raise
    except (TypeError, OverflowError) as e:
        raise ValueError(f"cannot parse colour {value!r}: {e}") from e


def parse_colors(values: Any) -> np.ndarray:
    """Parse a list of colours into an ``(N, 3)`` or ``(N, 4)`` float array.

    Parameters
    ----------
    values : sequence or ndarray
        Any mix of the forms `parse_color` accepts, or an ``(N, 3|4)`` array of floats
        (0..1) or integers (0..255). Colours with and without alpha cannot be mixed.

    Returns
    -------
    ndarray, shape (N, 3) or (N, 4)

    Raises
    ------
    ValueError
        If any entry cannot be parsed or the alpha channels are mixed.

    Examples
    --------
    >>> from lcms2._parse import parse_colors
    >>> parse_colors(["red", "#00f"]).shape
    (2, 3)
    """
    if isinstance(values, str):
        raise ValueError(f"expected a sequence of colours, got the string {values!r}")
    if isinstance(values, np.ndarray) and values.ndim == 2:
        if values.shape[1] not in (3, 4):
            raise ValueError(f"expected an (N, 3) or (N, 4) array, got shape {values.shape}")
        return np.stack([_parse_numeric(row) for row in values]) if len(values) else values.reshape(0, 3)
    items = [parse_color(v) for v in values]
    if not items:
        raise ValueError("expected at least one colour")
    if len({len(c) for c in items}) != 1:
        raise ValueError("cannot mix colours with and without alpha in one list")
    return np.stack(items)


def to_hex(rgb: Any) -> str | list[str]:
    """Format sRGB floats in 0..1 as ``"#rrggbb"`` (``"#rrggbbaa"`` with alpha).

    Parameters
    ----------
    rgb : array_like, shape (3,), (4,) or (N, 3|4)
        Colours as floats in 0..1; out-of-range values are clipped.

    Returns
    -------
    str or list of str
        One string for a single colour, a list for an array of colours.

    Examples
    --------
    >>> from lcms2._parse import to_hex
    >>> to_hex([1.0, 0.5, 0.0])
    '#ff8000'
    >>> to_hex([[0, 0, 0], [1, 1, 1]])
    ['#000000', '#ffffff']
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected shape (3,), (4,) or (N, 3|4), got {arr.shape}")
    b = np.rint(np.clip(arr, 0.0, 1.0) * 255).astype(int)
    if arr.ndim == 1:
        return "#" + "".join(f"{v:02x}" for v in b)
    return ["#" + "".join(f"{v:02x}" for v in row) for row in b]
