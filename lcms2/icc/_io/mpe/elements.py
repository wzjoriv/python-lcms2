"""Matrix and CLUT elements, and the element position table."""

from __future__ import annotations

import struct

import numpy as np

from ...._errors import ICCParseError
from .. import binary as bn


def _position_table(blobs, base: int) -> tuple:
    positions, body, cursor = [], bytearray(), base
    for blob in blobs:
        positions.append((cursor, len(blob)))
        pad = bn.pad4(len(blob))
        body += blob + b"\x00" * pad
        cursor += len(blob) + pad
    return b"".join(struct.pack(">II", o, s) for o, s in positions), bytes(body)


def encode_matrix_element(matrix: np.ndarray, offset: np.ndarray) -> bytes:
    """Encode a matrixElement ('matf') body from an (out x in) matrix and offset."""
    matrix = np.asarray(matrix, dtype=np.float64)
    out_ch, in_ch = matrix.shape
    return (
        b"matf"
        + b"\x00" * 4
        + struct.pack(">HH", in_ch, out_ch)
        + matrix.astype(">f4").tobytes()
        + np.asarray(offset, dtype=np.float64).reshape(out_ch).astype(">f4").tobytes()
    )


def decode_matrix_element(buf: bytes) -> dict:
    """Decode a matrixElement ('matf') body into ``matrix`` (out x in) and ``offset``."""
    in_ch, out_ch = struct.unpack_from(">HH", buf, 8)
    values = np.frombuffer(buf, dtype=">f4", count=in_ch * out_ch + out_ch, offset=12)
    values = values.astype(np.float64)
    return {
        "kind": "matrix",
        "in_ch": in_ch,
        "out_ch": out_ch,
        "matrix": values[: in_ch * out_ch].reshape(out_ch, in_ch),
        "offset": values[in_ch * out_ch :],
    }


def encode_clut_element(clut: np.ndarray) -> bytes:
    """Encode a CLUTElement ('clut') whose inputs span [0, 1].

    Parameters
    ----------
    clut : ndarray, shape grid_points + (out_ch,)
        Table samples; the first input varies slowest.

    Returns
    -------
    bytes
    """
    grid = clut.shape[:-1]
    grid_bytes = bytearray(16)
    for i, g in enumerate(grid):
        grid_bytes[i] = g
    return (
        b"clut"
        + b"\x00" * 4
        + struct.pack(">HH", len(grid), clut.shape[-1])
        + bytes(grid_bytes)
        + np.asarray(clut, dtype=">f4").tobytes()
    )


def decode_clut_element(buf: bytes) -> dict:
    """Decode a CLUTElement ('clut') body into its grid and samples."""
    in_ch, out_ch = struct.unpack_from(">HH", buf, 8)
    grid = list(buf[12 : 12 + in_ch])
    if in_ch < 1 or in_ch > 16 or any(g < 2 for g in grid):
        raise ICCParseError("CLUT element with an invalid grid")
    count = int(np.prod(grid)) * out_ch
    values = np.frombuffer(buf, dtype=">f4", count=count, offset=28).astype(np.float64)
    return {
        "kind": "clut",
        "in_ch": in_ch,
        "out_ch": out_ch,
        "grid_points": grid,
        "clut": values.reshape(tuple(grid) + (out_ch,)),
    }
