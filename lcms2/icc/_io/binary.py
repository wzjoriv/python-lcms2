"""
Big-endian encoders and decoders for ICC number types.
"""

from __future__ import annotations

import datetime as _dt
import struct
from collections.abc import Sequence


def read_s15Fixed16(buf: bytes, off: int) -> float:
    (v,) = struct.unpack_from(">i", buf, off)
    return v / 65536.0


def write_s15Fixed16(x: float) -> bytes:
    """Encode a float as an s15Fixed16Number, clamped to its range."""
    v = int(round(x * 65536.0))
    v = max(-(2**31), min(2**31 - 1, v))
    return struct.pack(">i", v)


def read_u16Fixed16(buf: bytes, off: int) -> float:
    (v,) = struct.unpack_from(">I", buf, off)
    return v / 65536.0


def write_u16Fixed16(x: float) -> bytes:
    v = int(round(x * 65536.0))
    v = max(0, min(2**32 - 1, v))
    return struct.pack(">I", v)


def read_u1Fixed15(buf: bytes, off: int) -> float:
    (v,) = struct.unpack_from(">H", buf, off)
    return v / 32768.0


def write_u1Fixed15(x: float) -> bytes:
    v = int(round(x * 32768.0))
    v = max(0, min(0xFFFF, v))
    return struct.pack(">H", v)


def read_u8Fixed8(buf: bytes, off: int) -> float:
    (v,) = struct.unpack_from(">H", buf, off)
    return v / 256.0


def write_u8Fixed8(x: float) -> bytes:
    v = int(round(x * 256.0))
    v = max(0, min(0xFFFF, v))
    return struct.pack(">H", v)


def read_u8(buf: bytes, off: int) -> int:
    return buf[off]


def read_u16(buf: bytes, off: int) -> int:
    return struct.unpack_from(">H", buf, off)[0]


def read_u32(buf: bytes, off: int) -> int:
    return struct.unpack_from(">I", buf, off)[0]


def read_u64(buf: bytes, off: int) -> int:
    return struct.unpack_from(">Q", buf, off)[0]


def read_s32(buf: bytes, off: int) -> int:
    return struct.unpack_from(">i", buf, off)[0]


def write_u16(v: int) -> bytes:
    return struct.pack(">H", v & 0xFFFF)


def write_u32(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def write_u64(v: int) -> bytes:
    return struct.pack(">Q", v & 0xFFFFFFFFFFFFFFFF)


def write_s32(v: int) -> bytes:
    return struct.pack(">i", v)


def read_dateTime(buf: bytes, off: int) -> _dt.datetime:
    (year, month, day, hour, minute, second) = struct.unpack_from(">HHHHHH", buf, off)
    try:
        return _dt.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return _dt.datetime(1970, 1, 1)


def write_dateTime(dt: _dt.datetime) -> bytes:
    return struct.pack(">HHHHHH", dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def read_XYZNumber(buf: bytes, off: int) -> tuple[float, float, float]:
    x = read_s15Fixed16(buf, off)
    y = read_s15Fixed16(buf, off + 4)
    z = read_s15Fixed16(buf, off + 8)
    return (x, y, z)


def write_XYZNumber(xyz: Sequence[float]) -> bytes:
    x, y, z = xyz
    return write_s15Fixed16(x) + write_s15Fixed16(y) + write_s15Fixed16(z)


def sig(buf: bytes, off: int, length: int = 4) -> bytes:
    return bytes(buf[off : off + length])


def pad4(n: int) -> int:
    """Number of padding bytes that bring `n` to a multiple of 4."""
    return (4 - (n % 4)) % 4
