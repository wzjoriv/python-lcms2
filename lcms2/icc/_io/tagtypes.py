"""
Encoders and decoders for lut8Type, lut16Type, lutAtoBType and lutBtoAType.

The curve, text and XYZ tag types live in sibling modules and are re-exported
here, so ``tagtypes`` stays the one namespace of tag codecs.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..._errors import ICCParseError, ICCWriteError
from . import binary as bn
from .curve_tags import (  # noqa: F401
    CurveDescriptor,
    decode_curv,
    decode_para,
    encode_curv,
    encode_para,
    eval_curve,
    eval_para,
    invert_curve_numeric,
    invert_para,
)
from .text_tags import (  # noqa: F401
    decode_desc_v2,
    decode_mluc,
    decode_sf32,
    decode_text,
    decode_XYZType,
    encode_desc_v2,
    encode_mluc,
    encode_sf32,
    encode_text,
    encode_XYZType,
)


def decode_lut8(body: bytes) -> dict[str, Any]:
    """Decode a lut8Type ('mft1') body into matrix, tables and CLUT."""
    in_ch = bn.read_u8(body, 8)
    out_ch = bn.read_u8(body, 9)
    grid = bn.read_u8(body, 10)
    matrix = np.array([bn.read_s15Fixed16(body, 12 + 4 * i) for i in range(9)]).reshape(3, 3)
    off = 48
    in_tables = np.frombuffer(body, dtype=np.uint8, count=in_ch * 256, offset=off).astype(np.float64) / 255.0
    in_tables = in_tables.reshape(in_ch, 256)
    off += in_ch * 256
    clut_n = grid**in_ch * out_ch
    clut = np.frombuffer(body, dtype=np.uint8, count=clut_n, offset=off).astype(np.float64) / 255.0
    off += clut_n
    out_tables = (
        np.frombuffer(body, dtype=np.uint8, count=out_ch * 256, offset=off).astype(np.float64) / 255.0
    )
    out_tables = out_tables.reshape(out_ch, 256)
    clut_shape = tuple([grid] * in_ch) + (out_ch,)
    clut = clut.reshape(clut_shape)
    return {
        "in_ch": in_ch,
        "out_ch": out_ch,
        "grid": grid,
        "matrix": matrix,
        "in_tables": in_tables,
        "clut": clut,
        "out_tables": out_tables,
    }


def decode_lut16(body: bytes) -> dict[str, Any]:
    """Decode a lut16Type ('mft2') body into matrix, tables and CLUT."""
    in_ch = bn.read_u8(body, 8)
    out_ch = bn.read_u8(body, 9)
    grid = bn.read_u8(body, 10)
    matrix = np.array([bn.read_s15Fixed16(body, 12 + 4 * i) for i in range(9)]).reshape(3, 3)
    n_in_entries = bn.read_u16(body, 48)
    n_out_entries = bn.read_u16(body, 50)
    off = 52
    in_tables = (
        np.frombuffer(body, dtype=">u2", count=in_ch * n_in_entries, offset=off).astype(np.float64) / 65535.0
    )
    in_tables = in_tables.reshape(in_ch, n_in_entries)
    off += 2 * in_ch * n_in_entries
    clut_n = grid**in_ch * out_ch
    clut = np.frombuffer(body, dtype=">u2", count=clut_n, offset=off).astype(np.float64) / 65535.0
    off += 2 * clut_n
    out_tables = (
        np.frombuffer(body, dtype=">u2", count=out_ch * n_out_entries, offset=off).astype(np.float64)
        / 65535.0
    )
    out_tables = out_tables.reshape(out_ch, n_out_entries)
    clut_shape = tuple([grid] * in_ch) + (out_ch,)
    clut = clut.reshape(clut_shape)
    return {
        "in_ch": in_ch,
        "out_ch": out_ch,
        "grid": grid,
        "matrix": matrix,
        "in_tables": in_tables,
        "clut": clut,
        "out_tables": out_tables,
    }


def encode_lut16(
    in_ch: int,
    out_ch: int,
    grid: int,
    matrix: np.ndarray,
    in_tables: np.ndarray,
    clut: np.ndarray,
    out_tables: np.ndarray,
) -> bytes:
    """Encode a lut16Type ('mft2') tag body from its component parts."""
    n_in_entries = in_tables.shape[1]
    n_out_entries = out_tables.shape[1]
    out = bytearray()
    out += b"mft2" + b"\x00" * 4
    out += bytes([in_ch, out_ch, grid, 0])
    for v in np.asarray(matrix, dtype=np.float64).ravel():
        out += bn.write_s15Fixed16(v)
    out += struct.pack(">HH", n_in_entries, n_out_entries)
    q = np.clip(np.round(in_tables * 65535.0), 0, 65535).astype(">u2")
    out += q.tobytes()
    qc = np.clip(np.round(clut.ravel() * 65535.0), 0, 65535).astype(">u2")
    out += qc.tobytes()
    qo = np.clip(np.round(out_tables * 65535.0), 0, 65535).astype(">u2")
    out += qo.tobytes()
    return bytes(out)


def _decode_subtag_curves(body: bytes, offset: int, n_curves: int) -> tuple[list[CurveDescriptor], int]:
    """Decode `n_curves` consecutive padded curve sub-tags starting at `offset`."""
    curves = []
    pos = offset
    for _ in range(n_curves):
        tsig = bn.sig(body, pos)
        if tsig == b"curv":
            count = bn.read_u32(body, pos + 8)
            if count == 0:
                size = 12
            elif count == 1:
                size = 14
            else:
                size = 12 + 2 * count
            samples = decode_curv(body[pos : pos + size])
            curves.append(("curv", samples))
        elif tsig == b"para":
            func_type = bn.read_u16(body, pos + 8)
            nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[func_type]
            size = 12 + 4 * nparams
            ft, params = decode_para(body[pos : pos + size])
            curves.append(("para", (ft, params)))
        else:
            raise ICCParseError(f"unsupported curve sub-tag type {tsig!r} in mAB/mBA")
        pos += size + bn.pad4(size)
    return curves, pos


def _mab_curve_counts(sig4: bytes, in_ch: int, out_ch: int) -> dict:
    if sig4 == b"mBA ":
        return {"A": out_ch, "M": in_ch, "B": in_ch}
    return {"A": in_ch, "M": out_ch, "B": out_ch}


def decode_lutAtoB_or_BtoA(body: bytes) -> dict:
    """Decode a lutAtoBType ('mAB ') or lutBtoAType ('mBA ') tag body.

    Returns
    -------
    dict
        ``in_ch``, ``out_ch``, ``A_curves``, ``M_curves``, ``B_curves``,
        ``matrix``, ``matrix_offset`` and ``clut``; absent elements are None.
    """
    sig4 = bn.sig(body, 0)
    in_ch = bn.read_u8(body, 8)
    out_ch = bn.read_u8(body, 9)
    off_B = bn.read_u32(body, 12)
    off_matrix = bn.read_u32(body, 16)
    off_M = bn.read_u32(body, 20)
    off_clut = bn.read_u32(body, 24)
    off_A = bn.read_u32(body, 28)
    counts = _mab_curve_counts(sig4, in_ch, out_ch)

    result = {
        "in_ch": in_ch,
        "out_ch": out_ch,
        "A_curves": None,
        "M_curves": None,
        "B_curves": None,
        "matrix": None,
        "matrix_offset": None,
        "clut": None,
    }
    for name, off in (("A", off_A), ("M", off_M), ("B", off_B)):
        if off:
            result[f"{name}_curves"], _ = _decode_subtag_curves(body, off, counts[name])
    if off_matrix:
        vals = [bn.read_s15Fixed16(body, off_matrix + 4 * i) for i in range(12)]
        result["matrix"] = np.array(vals[:9]).reshape(3, 3)
        result["matrix_offset"] = np.array(vals[9:])
    if off_clut:
        grid_pts = list(body[off_clut : off_clut + 16])[:in_ch]
        precision = body[off_clut + 16]
        n = out_ch
        for g in grid_pts:
            n *= g
        data_off = off_clut + 20
        if precision == 1:
            arr = np.frombuffer(body, dtype=np.uint8, count=n, offset=data_off) / 255.0
        elif precision == 2:
            arr = np.frombuffer(body, dtype=">u2", count=n, offset=data_off) / 65535.0
        else:
            raise ICCParseError(f"bad CLUT precision byte {precision} in mAB/mBA")
        result["clut"] = arr.astype(np.float64).reshape(tuple(grid_pts) + (out_ch,))
    return result


def encode_lutAtoB_or_BtoA(
    sig4: bytes,
    in_ch: int,
    out_ch: int,
    A_curves: Sequence[CurveDescriptor] | None = None,
    clut: np.ndarray | None = None,
    M_curves: Sequence[CurveDescriptor] | None = None,
    matrix: np.ndarray | None = None,
    matrix_offset: np.ndarray | None = None,
    B_curves: Sequence[CurveDescriptor] | None = None,
) -> bytes:
    """Encode a lutAtoBType ('mAB ') or lutBtoAType ('mBA ') tag body.

    Parameters
    ----------
    sig4 : bytes
        ``b'mAB '`` or ``b'mBA '``.
    in_ch, out_ch : int
        Input and output channel counts.
    A_curves, M_curves, B_curves : list of tuple, optional
        Curve descriptors, each ``('curv', samples)`` or
        ``('para', (func_type, params))``. `B_curves` is required.
    clut : ndarray, optional
        CLUT samples in [0, 1], shape ``grid_points + (out_ch,)``.
    matrix : ndarray, shape (3, 3), optional
        Matrix element.
    matrix_offset : ndarray, shape (3,), optional
        Matrix offset. Defaults to zeros.

    Returns
    -------
    bytes
    """
    if B_curves is None:
        raise ICCWriteError("B curves are required in mAB/mBA")
    counts = _mab_curve_counts(sig4, in_ch, out_ch)
    for name, curves in (("A", A_curves), ("M", M_curves), ("B", B_curves)):
        if curves is not None and len(curves) != counts[name]:
            raise ICCWriteError(f"{sig4!r} needs {counts[name]} {name} curves, got {len(curves)}")

    def enc_curve(kind: str, payload: Any) -> bytes:
        if kind == "curv":
            return encode_curv(payload)
        if kind == "para":
            ft, params = payload
            return encode_para(ft, params)
        raise ICCWriteError(f"unknown curve kind {kind}")

    blocks = []
    for name, curves in (("A", A_curves), ("M", M_curves), ("B", B_curves)):
        if curves is not None:
            blocks.extend((name, enc_curve(*c)) for c in curves)
    if clut is not None:
        header = bytearray(16)
        for i, g in enumerate(clut.shape[:-1]):
            header[i] = g
        header += bytes([2, 0, 0, 0])
        q = np.clip(np.round(clut.ravel() * 65535.0), 0, 65535).astype(">u2")
        blocks.append(("CLUT", bytes(header) + q.tobytes()))
    if matrix is not None:
        offset = np.zeros(3) if matrix_offset is None else np.asarray(matrix_offset)
        mb = b"".join(bn.write_s15Fixed16(v) for v in np.asarray(matrix).ravel())
        mb += b"".join(bn.write_s15Fixed16(v) for v in offset.ravel())
        blocks.append(("MTX", mb))

    cursor = 32
    offsets = {"A": 0, "CLUT": 0, "M": 0, "MTX": 0, "B": 0}
    body = bytearray()
    for name, payload in blocks:
        if not offsets[name]:
            offsets[name] = cursor
        pad = bn.pad4(len(payload))
        body += payload + b"\x00" * pad
        cursor += len(payload) + pad

    header = bytearray(32)
    header[0:4] = sig4
    header[8] = in_ch
    header[9] = out_ch
    struct.pack_into(">I", header, 12, offsets["B"])
    struct.pack_into(">I", header, 16, offsets["MTX"])
    struct.pack_into(">I", header, 20, offsets["M"])
    struct.pack_into(">I", header, 24, offsets["CLUT"])
    struct.pack_into(">I", header, 28, offsets["A"])
    return bytes(header) + bytes(body)
