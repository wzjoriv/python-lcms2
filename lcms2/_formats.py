"""Pixel formats: explicit format strings and inference from NumPy arrays."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from . import _lcms2

__all__ = ["FORMATS", "FormatInfo", "as_array", "dtype_spec", "format_info", "parse_format"]

#: Every LittleCMS ``TYPE_*`` format: name -> (format, dtype, channels incl. extra, extra, planar).
FORMATS: dict[str, tuple[int, str, int, int, bool]] = dict(_lcms2.FORMATS)
_UPPER = {k.upper(): k for k in FORMATS}

#: dtype -> (bytes per sample as LittleCMS counts them, is_float); doubles are 0 bytes.
_DTYPES = {
    np.dtype(np.uint8): (1, False),
    np.dtype(np.uint16): (2, False),
    np.dtype(np.float16): (2, True),
    np.dtype(np.float32): (4, True),
    np.dtype(np.float64): (0, True),
}
ACCEPTED = "uint8, uint16, float16, float32 or float64"


class FormatInfo(NamedTuple):
    """Decoded pixel format."""

    dtype: np.dtype
    channels: int
    extra: int
    planar: bool


def format_info(fmt: int) -> FormatInfo:
    """Decode a LittleCMS pixel format number."""
    dtype, channels, extra, planar = _lcms2.format_info(int(fmt))
    return FormatInfo(np.dtype(dtype), channels, extra, planar)


def format_name(fmt: int) -> str:
    """The ``FORMATS`` name of a format number, or its hex value."""
    for name, v in FORMATS.items():
        if v[0] == fmt:
            return name
    return f"0x{fmt:08x}"


def parse_format(fmt: str | int) -> int:
    """A format number from a case-insensitive ``FORMATS`` name (``"rgb_8"``) or an int.

    Raises
    ------
    ValueError
        For unknown names, suggesting close matches.
    """
    if isinstance(fmt, int) and not isinstance(fmt, bool):
        return fmt
    if not isinstance(fmt, str):
        raise TypeError(f"a pixel format must be a name such as 'RGB_8' or an int, got {fmt!r}")
    key = _UPPER.get(fmt.upper())
    if key is None:
        import difflib

        close = difflib.get_close_matches(fmt.upper(), _UPPER, n=3)
        hint = f"; did you mean {', '.join(_UPPER[c] for c in close)}?" if close else ""
        raise ValueError(f"unknown pixel format {fmt!r}{hint} (see lcms2.FORMATS)")
    return FORMATS[key][0]


def is_format_name(value: Any) -> bool:
    """Whether ``value`` is a string naming a pixel format."""
    return isinstance(value, str) and value.upper() in _UPPER


def dtype_spec(dtype: Any) -> tuple[int, bool]:
    """``(bytes, is_float)`` of a supported dtype, as LittleCMS formats count them.

    Raises
    ------
    TypeError
        For unsupported dtypes, naming the accepted ones.
    """
    dt = np.dtype(dtype)
    spec = _DTYPES.get(dt) if dt.isnative else None
    if spec is None:
        raise TypeError(
            f"unsupported dtype {dt}; pixel data must be {ACCEPTED} "
            f"(convert with .astype(np.float64) or .astype(np.uint16))"
        )
    return spec


def _infer_sequence_dtype(arr: np.ndarray) -> np.dtype:
    if arr.dtype.kind == "f":
        return np.dtype(np.float64)
    if arr.dtype.kind in "iub":
        if arr.size == 0:
            return np.dtype(np.float64)
        lo, hi = int(arr.min()), int(arr.max())
        if lo >= 0 and hi <= 255:
            return np.dtype(np.uint8)
        if lo >= 0 and hi <= 65535:
            return np.dtype(np.uint16)
        raise ValueError(
            f"integer pixel values must be in 0..65535, got {lo}..{hi}; pass floats or a NumPy array instead"
        )
    raise TypeError(f"pixel lists must contain numbers, got {arr.dtype}")


def as_array(data: Any, dtype: Any = None) -> np.ndarray:
    """A C-contiguous array of pixel data.

    Lists and tuples of ints become ``uint8`` when all values fit 0..255, else ``uint16``; any float
    makes ``float64``. With ``dtype`` the data is converted to it instead (used by explicit formats).
    """
    if isinstance(data, np.ndarray):
        arr = data
    elif dtype is not None:
        arr = np.asarray(data, dtype=dtype)
    else:
        arr = np.asarray(data)
        if not isinstance(data, (list, tuple)) and arr.dtype in _DTYPES:
            pass
        else:
            arr = arr.astype(_infer_sequence_dtype(arr))
    return np.ascontiguousarray(arr)
