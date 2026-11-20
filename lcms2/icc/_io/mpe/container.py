"""
The multiProcessElements container and its element dispatch.
"""

from __future__ import annotations

import struct

from ...._errors import ICCParseError
from .. import binary as bn
from .calc import decode_calc_element, validate_calc_element
from .curves import decode_curveset_element
from .elements import _position_table, decode_clut_element, decode_matrix_element


def encode_mpet(in_ch: int, out_ch: int, element_blobs) -> bytes:
    """Encode an 'mpet' tag body from encoded element bodies."""
    header = b"mpet" + b"\x00" * 4 + struct.pack(">HHI", in_ch, out_ch, len(element_blobs))
    table, body = _position_table(element_blobs, len(header) + 8 * len(element_blobs))
    return header + table + body


def decode_mpet(body: bytes) -> dict:
    """Decode an 'mpet' tag body into ``in_ch``, ``out_ch`` and ``elements``."""
    if body[0:4] != b"mpet":
        raise ICCParseError(f"not an mpet tag: {body[0:4]!r}")
    in_ch, out_ch, n = struct.unpack_from(">HHI", body, 8)
    elements = []
    for i in range(n):
        off, size = struct.unpack_from(">II", body, 16 + 8 * i)
        if off + size > len(body):
            raise ICCParseError("mpet element out of bounds")
        elements.append(decode_element(body[off : off + size]))
    return {"in_ch": in_ch, "out_ch": out_ch, "elements": elements}


def decode_element(buf: bytes) -> dict:
    """Decode one processing element by its signature."""
    decoders = {
        b"matf": decode_matrix_element,
        b"cvst": decode_curveset_element,
        b"clut": decode_clut_element,
        b"calc": decode_calc_element,
    }
    decoder = decoders.get(bn.sig(buf, 0))
    if decoder is None:
        raise ICCParseError(f"unsupported processing element {bn.sig(buf, 0)!r}")
    return decoder(buf)


def validate_mpet(decoded: dict) -> None:
    """Check that an mpet's element channel counts chain from its input to its output.

    Raises
    ------
    ICCParseError
        If an element's input count differs from the previous element's output count.
    """
    width = decoded["in_ch"]
    for elem in decoded["elements"]:
        if elem["in_ch"] != width:
            raise ICCParseError(f"mpet element expects {elem['in_ch']} channels, gets {width}")
        if elem["kind"] == "calc":
            validate_calc_element(elem, elem["in_ch"], elem["out_ch"])
        width = elem["out_ch"]
    if width != decoded["out_ch"]:
        raise ICCParseError(f"mpet produces {width} channels, declares {decoded['out_ch']}")
