"""
Tag types holding XYZ numbers, fixed-point arrays and text.
"""

from __future__ import annotations

__all__ = [
    "decode_XYZType",
    "encode_XYZType",
    "decode_sf32",
    "encode_sf32",
    "decode_text",
    "encode_text",
    "decode_desc_v2",
    "encode_desc_v2",
    "decode_mluc",
    "encode_mluc",
]

import struct
from collections.abc import Iterable, Sequence

import numpy as np

from . import binary as bn


def decode_XYZType(body: bytes) -> list[tuple[float, float, float]]:
    """Decode an XYZType tag body into a list of (X, Y, Z) tuples."""
    n = (len(body) - 8) // 12
    return [bn.read_XYZNumber(body, 8 + 12 * i) for i in range(n)]


def encode_XYZType(xyz_list: Iterable[Sequence[float]]) -> bytes:
    """Encode a list of (X, Y, Z) tuples as an XYZType tag body."""
    out = b"XYZ " + b"\x00" * 4
    for xyz in xyz_list:
        out += bn.write_XYZNumber(xyz)
    return out


def decode_sf32(body: bytes) -> np.ndarray:
    """Decode an s15Fixed16ArrayType tag body into a float64 array."""
    n = (len(body) - 8) // 4
    return np.array([bn.read_s15Fixed16(body, 8 + 4 * i) for i in range(n)], dtype=np.float64)


def encode_sf32(values: np.ndarray) -> bytes:
    """Encode a float array as an s15Fixed16ArrayType tag body."""
    out = b"sf32" + b"\x00" * 4
    for v in np.asarray(values, dtype=np.float64).ravel():
        out += bn.write_s15Fixed16(v)
    return out


def decode_text(body: bytes) -> str:
    """Decode a textType ('text') tag body into a plain ASCII string."""
    s = body[8:]
    nul = s.find(b"\x00")
    if nul >= 0:
        s = s[:nul]
    return s.decode("ascii", errors="replace")


def encode_text(s: str) -> bytes:
    """Encode a plain ASCII string as a textType ('text') tag body."""
    b = s.encode("ascii", errors="replace") + b"\x00"
    return b"text" + b"\x00" * 4 + b


def decode_desc_v2(body: bytes) -> str:
    """Decode a v2 textDescriptionType ('desc') tag body's ASCII portion."""
    ascii_count = bn.read_u32(body, 8)
    raw = body[12 : 12 + ascii_count]
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("ascii", errors="replace")


def encode_desc_v2(s: str) -> bytes:
    """Encode an ASCII string as a v2 textDescriptionType ('desc') tag body."""
    ascii_bytes = s.encode("ascii", errors="replace") + b"\x00"
    ascii_count = len(ascii_bytes)
    out = b"desc" + b"\x00" * 4
    out += struct.pack(">I", ascii_count)
    out += ascii_bytes
    out += struct.pack(">I", 0) + struct.pack(">I", 0)
    out += struct.pack(">H", 0) + b"\x00" + b"\x00" * 67
    return out


def decode_mluc(body: bytes) -> dict:
    """Decode a multiLocalizedUnicodeType ('mluc') tag body.

    Returns
    -------
    dict
        Mapping of ``(lang, country)`` (2-letter ISO codes) to decoded
        text.
    """
    n = bn.read_u32(body, 8)
    rec_size = bn.read_u32(body, 12)
    result = {}
    for i in range(n):
        off = 16 + i * rec_size
        lang = body[off : off + 2].decode("ascii", errors="replace")
        country = body[off + 2 : off + 4].decode("ascii", errors="replace")
        length = bn.read_u32(body, off + 4)
        str_off = bn.read_u32(body, off + 8)
        raw = body[str_off : str_off + length]
        text = raw.decode("utf-16-be", errors="replace")
        result[(lang, country)] = text
    return result


def encode_mluc(strings: dict) -> bytes:
    """Encode a multiLocalizedUnicodeType ('mluc') tag body.

    Parameters
    ----------
    strings : dict
        Mapping of ``(lang, country)`` (2-letter ISO codes) to text.
    """
    n = len(strings)
    header = b"mluc" + b"\x00" * 4 + struct.pack(">II", n, 12)
    records = b""
    storage = b""
    storage_base = 16 + n * 12
    for (lang, country), text in strings.items():
        enc = text.encode("utf-16-be")
        records += lang.encode("ascii").ljust(2, b"\x00")[:2]
        records += country.encode("ascii").ljust(2, b"\x00")[:2]
        records += struct.pack(">II", len(enc), storage_base + len(storage))
        storage += enc
    return header + records + storage
