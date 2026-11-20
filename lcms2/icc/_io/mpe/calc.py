"""
Calculator elements ('calc'): encoding, decoding and validation.
"""

from __future__ import annotations

import struct

from ...._errors import ICCParseError, ICCWriteError
from .calc_eval import _BINARY, _CONSTANTS, _UNARY, _VARIADIC, _branch_layout
from .elements import _position_table

_FLOAT_OPS = frozenset({"data"})


_SIZE_OPS = frozenset({"if  ", "else"})


def encode_calc_element(in_ch: int, out_ch: int, ops) -> bytes:
    """Encode a calculatorElement ('calc') with one main function.

    Parameters
    ----------
    in_ch, out_ch : int
        Channel counts.
    ops : list of tuple
        ``(sig, arg)`` with a 4-character `sig`; `arg` is a float for
        ``'data'``, an operation count for ``'if  '``/``'else'``, and a
        ``(v1, v2)`` selector otherwise.

    Returns
    -------
    bytes
    """
    words = bytearray()
    for sig, arg in ops:
        if len(sig) != 4:
            raise ICCWriteError(f"calculator operation {sig!r} is not a 4-character signature")
        words += sig.encode("ascii")
        if sig in _FLOAT_OPS:
            words += struct.pack(">f", float(arg))
        elif sig in _SIZE_OPS:
            words += struct.pack(">I", int(arg))
        else:
            v1, v2 = arg
            words += struct.pack(">HH", int(v1), int(v2))
    func = b"func" + b"\x00" * 4 + struct.pack(">I", len(ops)) + bytes(words)
    header = b"calc" + b"\x00" * 4 + struct.pack(">HHI", in_ch, out_ch, 0)
    table, body = _position_table([func], len(header) + 8)
    return header + table + body


def decode_calc_element(buf: bytes) -> dict:
    """Decode a calculatorElement ('calc') body into its main function's operations."""
    in_ch, out_ch, n_sub = struct.unpack_from(">HHI", buf, 8)
    if n_sub:
        raise ICCParseError("calculator sub-elements are not supported")
    off, size = struct.unpack_from(">II", buf, 16)
    func = buf[off : off + size]
    if func[0:4] != b"func":
        raise ICCParseError("calculator main function missing")
    (n_ops,) = struct.unpack_from(">I", func, 8)
    if 12 + 8 * n_ops > len(func):
        raise ICCParseError("calculator function truncated")
    ops = []
    for i in range(n_ops):
        sig = func[12 + 8 * i : 16 + 8 * i].decode("latin-1")
        if sig in _FLOAT_OPS:
            (arg,) = struct.unpack_from(">f", func, 16 + 8 * i)
        elif sig in _SIZE_OPS:
            (arg,) = struct.unpack_from(">I", func, 16 + 8 * i)
        else:
            arg = struct.unpack_from(">HH", func, 16 + 8 * i)
        ops.append((sig, arg))
    return {"kind": "calc", "in_ch": in_ch, "out_ch": out_ch, "ops": ops}


def _stack_effect(sig: str, arg):
    if sig in _FLOAT_OPS or sig in _CONSTANTS:
        return 0, 1
    if sig in _SIZE_OPS:
        return None
    v1, v2 = arg
    if sig in ("in  ", "tget"):
        return 0, v2 + 1
    if sig in ("out ", "tput"):
        return v2 + 1, 0
    if sig == "tsav":
        return v2 + 1, v2 + 1
    if sig in _UNARY:
        return v1 + 1, v1 + 1
    if sig in _BINARY:
        return 2 * (v1 + 1), v1 + 1
    if sig in _VARIADIC:
        return v1 + 2, 1
    if sig == "copy":
        return v1 + 1, (v1 + 1) * (v2 + 2)
    if sig == "pop ":
        return v1 + 1, 0
    if sig in ("rotl", "rotr"):
        return v1 + 1, v1 + 1
    if sig == "flip":
        return v1 + 2, v1 + 2
    return None


def _check_ops(ops, in_ch, out_ch, written, depth):
    i = 0
    while i < len(ops):
        sig, arg = ops[i]
        if sig == "if  ":
            if depth < 1:
                raise ICCParseError("calculator stack underflow")
            depth -= 1
            start, then_n, else_n, has_else = _branch_layout(ops, i)
            if start + then_n + else_n > len(ops):
                raise ICCParseError("calculator branch runs past its function")
            then_depth = _check_ops(ops[start : start + then_n], in_ch, out_ch, written, depth)
            else_ops = ops[start + then_n : start + then_n + else_n]
            else_depth = _check_ops(else_ops, in_ch, out_ch, written, depth) if has_else else depth
            if then_depth != else_depth:
                raise ICCParseError("calculator branches leave different stack depths")
            depth = then_depth
            i = start + then_n + else_n
            continue
        effect = _stack_effect(sig, arg)
        if effect is None:
            raise ICCParseError(f"unsupported calculator operation {sig!r}")
        needed, pushed = effect
        if sig == "in  " and arg[0] + arg[1] >= in_ch:
            raise ICCParseError(f"calculator reads input {arg[0] + arg[1]} of {in_ch}")
        if sig == "out ":
            if arg[0] + arg[1] >= out_ch:
                raise ICCParseError(f"calculator writes output {arg[0] + arg[1]} of {out_ch}")
            written.update(range(arg[0], arg[0] + arg[1] + 1))
        if depth < needed:
            raise ICCParseError("calculator stack underflow")
        depth += pushed - needed
        i += 1
    return depth


def validate_calc_element(decoded: dict, in_ch: int, out_ch: int) -> None:
    """Check a calculator element's channels, operations and stack use.

    Raises
    ------
    ICCParseError
        If the program cannot run on `in_ch` inputs to fill `out_ch` outputs.
    """
    if decoded["in_ch"] != in_ch or decoded["out_ch"] != out_ch:
        raise ICCParseError("calculator channel counts do not match its position")
    written: set = set()
    _check_ops(decoded["ops"], in_ch, out_ch, written, 0)
    if written != set(range(out_ch)):
        raise ICCParseError("calculator program does not write every output")
