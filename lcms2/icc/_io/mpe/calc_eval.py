"""Evaluating calculator programs."""

from __future__ import annotations

import numpy as np

from ...._errors import ICCParseError


def _round_half_away(a):
    return np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))


_UNARY = {
    "neg ": np.negative,
    "sq  ": np.square,
    "sqrt": np.sqrt,
    "cb  ": lambda a: a * a * a,
    "cbrt": np.cbrt,
    "sign": np.sign,
    "abs ": np.abs,
    "flor": np.floor,
    "ceil": np.ceil,
    "trnc": np.trunc,
    "rond": _round_half_away,
    "exp ": np.exp,
    "log ": np.log10,
    "ln  ": np.log,
    "sin ": np.sin,
    "cos ": np.cos,
    "tan ": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "not ": lambda a: (a < 0.5).astype(np.float64),
}


def _fmod(a, b):
    bad = ~np.isfinite(a) | ~np.isfinite(b) | (np.abs(b) < 1e-12)
    return np.where(bad, 0.0, np.fmod(a, np.where(bad, 1.0, b)))


def _divide(a, b):
    return np.where(b != 0, a / np.where(b != 0, b, 1.0), 0.0)


_BINARY = {
    "add ": np.add,
    "sub ": np.subtract,
    "mul ": np.multiply,
    "div ": _divide,
    "mod ": _fmod,
    "pow ": np.power,
    "atn2": lambda a, b: np.arctan2(b, a),
    "lt  ": lambda a, b: (a < b).astype(np.float64),
    "le  ": lambda a, b: (a <= b).astype(np.float64),
    "eq  ": lambda a, b: (a == b).astype(np.float64),
    "neq ": lambda a, b: (a != b).astype(np.float64),
    "ge  ": lambda a, b: (a >= b).astype(np.float64),
    "gt  ": lambda a, b: (a > b).astype(np.float64),
    "vmin": np.minimum,
    "vmax": np.maximum,
    "vand": lambda a, b: ((a >= 0.5) & (b >= 0.5)).astype(np.float64),
    "vor ": lambda a, b: ((a >= 0.5) | (b >= 0.5)).astype(np.float64),
}


_VARIADIC = {
    "sum ": lambda vals: np.sum(vals, axis=0),
    "prod": lambda vals: np.prod(vals, axis=0),
    "min ": lambda vals: np.min(vals, axis=0),
    "max ": lambda vals: np.max(vals, axis=0),
    "and ": lambda vals: np.all(np.stack(vals) >= 0.5, axis=0).astype(np.float64),
    "or  ": lambda vals: np.any(np.stack(vals) >= 0.5, axis=0).astype(np.float64),
}


_CONSTANTS = {"pi  ": np.pi, "+INF": np.inf, "-INF": -np.inf, "NaN ": np.nan}


def _branch_layout(ops, i):
    has_else = i + 1 < len(ops) and ops[i + 1][0] == "else"
    then_n = int(ops[i][1])
    else_n = int(ops[i + 1][1]) if has_else else 0
    start = i + (2 if has_else else 1)
    return start, then_n, else_n, has_else


def _merge(cond, then_state, else_state):
    then_stack, then_temps, then_out = then_state
    else_stack, else_temps, else_out = else_state
    stack = [np.where(cond, a, b) for a, b in zip(then_stack, else_stack, strict=False)]

    def merge_dict(a, b):
        out = dict(b)
        for k, v in a.items():
            out[k] = np.where(cond, v, b[k]) if k in b else v
        return out

    return stack, merge_dict(then_temps, else_temps), merge_dict(then_out, else_out)


def _run(ops, x, state):
    stack, temps, outputs = state
    i = 0
    while i < len(ops):
        sig, arg = ops[i]
        if sig == "if  ":
            cond = stack.pop() >= 0.5
            start, then_n, else_n, _ = _branch_layout(ops, i)
            then_ops = ops[start : start + then_n]
            else_ops = ops[start + then_n : start + then_n + else_n]
            then_state = _run(then_ops, x, (list(stack), dict(temps), dict(outputs)))
            else_state = _run(else_ops, x, (list(stack), dict(temps), dict(outputs)))
            stack, temps, outputs = _merge(cond, then_state, else_state)
            i = start + then_n + else_n
            continue
        if sig == "data":
            stack.append(np.full(x.shape[:-1], float(np.float32(arg))))
        elif sig in _CONSTANTS:
            stack.append(np.full(x.shape[:-1], _CONSTANTS[sig]))
        else:
            v1, v2 = arg
            if sig == "in  ":
                stack.extend(x[..., v1 + k] for k in range(v2 + 1))
            elif sig == "out ":
                vals = stack[len(stack) - (v2 + 1) :]
                del stack[len(stack) - (v2 + 1) :]
                for k, v in enumerate(vals):
                    outputs[v1 + k] = v
            elif sig == "tget":
                stack.extend(temps[v1 + k] for k in range(v2 + 1))
            elif sig in ("tput", "tsav"):
                vals = stack[len(stack) - (v2 + 1) :]
                if sig == "tput":
                    del stack[len(stack) - (v2 + 1) :]
                for k, v in enumerate(vals):
                    temps[v1 + k] = v
            elif sig in _UNARY:
                n = v1 + 1
                stack[-n:] = [_UNARY[sig](a) for a in stack[-n:]]
            elif sig in _BINARY:
                n = v1 + 1
                a, b = stack[-2 * n : -n], stack[-n:]
                del stack[-2 * n :]
                stack.extend(_BINARY[sig](p, q) for p, q in zip(a, b, strict=False))
            elif sig in _VARIADIC:
                n = v1 + 2
                vals = stack[-n:]
                del stack[-n:]
                stack.append(_VARIADIC[sig](vals))
            elif sig == "copy":
                stack.extend(stack[-(v1 + 1) :] * (v2 + 1))
            elif sig == "pop ":
                del stack[-(v1 + 1) :]
            elif sig in ("rotl", "rotr"):
                n = v1 + 1
                shift = (v2 + 1) % n
                vals = stack[-n:]
                stack[-n:] = vals[shift:] + vals[:shift] if sig == "rotl" else vals[-shift:] + vals[:-shift]
            elif sig == "flip":
                n = v1 + 2
                stack[-n:] = stack[-n:][::-1]
            else:
                raise ICCParseError(f"unsupported calculator operation {sig!r}")
        i += 1
    return stack, temps, outputs


def eval_calc_element(decoded: dict, x: np.ndarray, out_ch: int) -> np.ndarray:
    """Run a decoded calculator element on ``(..., in_ch)`` inputs.

    Raises
    ------
    ICCParseError
        If the program uses an unsupported operation or misses an output.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        _, _, outputs = _run(decoded["ops"], x, ([], {}, {}))
    if set(outputs) != set(range(out_ch)):
        raise ICCParseError("calculator program does not write every output")
    return np.stack([np.broadcast_to(outputs[i], x.shape[:-1]) for i in range(out_ch)], axis=-1)
