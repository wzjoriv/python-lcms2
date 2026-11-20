"""
Encode matrix/curve chains exactly as v2/v4 lookup-table tags.
"""

from __future__ import annotations

import itertools

import numpy as np

from .._io import tagtypes as tt
from .._io.constants import TYPE_MAB, TYPE_MBA
from . import curves as cv
from .sampling import grid_points

XYZ_SCALE = 65535.0 / 32768.0

_A2B_SLOTS = ("A", "CLUT", "M", "MATRIX", "B")
_B2A_SLOTS = ("B", "MATRIX", "M", "CLUT", "A")
_LUT16_SLOTS = ("IN", "CLUT", "OUT")
_KIND = {"A": "c", "B": "c", "M": "c", "IN": "c", "OUT": "c", "CLUT": "a", "MATRIX": "a"}


def _identity(t):
    return np.asarray(t, dtype=np.float64)


def _stage_width(stage):
    if stage[0] == "curveset":
        return len(stage[1]), len(stage[1])
    return stage[1].shape


def _slot_width(slot: str, is_b2a: bool, n: int, legacy: bool):
    if legacy:
        dev, pcs = n, 3
        first, last = (pcs, dev) if is_b2a else (dev, pcs)
        return {"IN": (first, first), "CLUT": (first, last), "OUT": (last, last)}[slot]
    if is_b2a:
        return {"B": (3, 3), "MATRIX": (3, 3), "M": (3, 3), "CLUT": (3, n), "A": (n, n)}[slot]
    return {"A": (n, n), "CLUT": (n, 3), "M": (3, 3), "MATRIX": (3, 3), "B": (3, 3)}[slot]


def _assign(chain, slots, is_b2a, n, legacy):
    """Map chain stages onto slots in order, or None if they do not fit."""
    for chosen in itertools.combinations(range(len(slots)), len(chain)):
        ok = True
        for stage, slot_idx in zip(chain, chosen, strict=False):
            slot = slots[slot_idx]
            kind = "c" if stage[0] == "curveset" else "a"
            if _KIND[slot] != kind or _stage_width(stage) != _slot_width(slot, is_b2a, n, legacy):
                ok = False
                break
        if ok:
            return {slots[i]: stage for i, stage in zip(chosen, chain, strict=False)}
    return None


def _full_pipeline(assigned, slots, is_b2a, n, legacy):
    """Every slot's stage, with identities filling unused slots, or None."""
    stages = []
    for slot in slots:
        width_in, width_out = _slot_width(slot, is_b2a, n, legacy)
        if slot in assigned:
            stages.append((slot, assigned[slot], False))
        elif _KIND[slot] == "c":
            stages.append((slot, ("curveset", [_identity] * width_in), True))
        elif width_in == width_out:
            stages.append((slot, ("matrix", np.eye(width_in), np.zeros(width_in)), True))
        else:
            return None
    return stages


def _apply(stage, x: np.ndarray) -> np.ndarray:
    if stage[0] == "matrix":
        return x @ stage[1] + stage[2]
    return np.stack([cv.evaluate(f, x[:, i]) for i, f in enumerate(stage[1])], axis=-1)


def _range(values: np.ndarray, pad: float):
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    span = hi - lo
    lo = lo - pad * span
    span = span * (1 + 2 * pad)
    return lo, np.where(span > 1e-12, span, 1.0)


def _encode(stages, in_box, out_box, samples):
    """Normalize every interface of `stages` into [0, 1] and encode each slot."""
    boxes = [in_box]
    cur = samples
    for idx, (slot, stage, filler) in enumerate(stages):
        out = _apply(stage, cur)
        if idx == len(stages) - 1:
            boxes.append(out_box)
        elif filler:
            boxes.append(boxes[-1])
        elif slot == "CLUT":
            lo, span = boxes[-1]
            corners = grid_points([(a, a + s) for a, s in zip(lo, span, strict=False)], 2)
            corners = corners.reshape(-1, len(lo))
            boxes.append(_range(_apply(stage, corners), 0.0))
        else:
            finite = out[np.all(np.isfinite(out), axis=-1)]
            if finite.size == 0:
                return None
            boxes.append(_range(finite, 0.02))
        cur = out

    encoded = {}
    for idx, (slot, stage, _) in enumerate(stages):
        (lo_i, sp_i), (lo_o, sp_o) = boxes[idx], boxes[idx + 1]
        if stage[0] == "curveset":
            encoded[slot] = [
                lambda t, f=f, a=a, s=s, b=b, r=r: (cv.evaluate(f, a + s * np.asarray(t)) - b) / r
                for f, a, s, b, r in zip(stage[1], lo_i, sp_i, lo_o, sp_o, strict=False)
            ]
        else:
            m, off = stage[1], stage[2]
            m_n = (sp_i[:, None] * m) / sp_o[None, :]
            off_n = (lo_i @ m + off - lo_o) / sp_o
            encoded[slot] = (m_n, off_n)
    return encoded


def _curves(fns):
    return [cv.sampled_curve(lambda t, f=f: np.nan_to_num(f(t))) for f in fns]


def _is_identity(fns) -> bool:
    t = np.linspace(0.0, 1.0, 17)
    return all(np.allclose(cv.evaluate(f, t), t, atol=1e-9) for f in fns)


def _clut_from_affine(m, off):
    corners = grid_points([(0.0, 1.0)] * m.shape[0], 2)
    return np.clip(corners @ m + off, 0.0, 1.0)


def encode_chain(chain, is_b2a: bool, n: int, version_major: int, samples: np.ndarray):
    """Encode a matrix/curve `chain` as an A2B (device to PCS) or B2A tag.

    Parameters
    ----------
    chain : list
        Stages ``('matrix', M, offset)`` with ``y = x @ M + offset`` and
        ``('curveset', [f, ...])``, from normalized device values to real
        XYZ (A2B) or back (B2A).
    is_b2a : bool
        Direction of the tag.
    n : int
        Device channel count.
    version_major : int
        2 writes lut16Type, 4 writes lutAtoBType/lutBtoAType.
    samples : ndarray
        Representative chain inputs.

    Returns
    -------
    bytes or None
        The tag body, or None if the chain does not fit the tag structure.
    """
    legacy = version_major == 2
    slots = _LUT16_SLOTS if legacy else (_B2A_SLOTS if is_b2a else _A2B_SLOTS)
    assigned = _assign(chain, slots, is_b2a, n, legacy)
    if assigned is None:
        return None
    stages = _full_pipeline(assigned, slots, is_b2a, n, legacy)
    if stages is None:
        return None
    pcs_box = (np.zeros(3), np.full(3, XYZ_SCALE))
    dev_box = (np.zeros(n), np.ones(n))
    in_box, out_box = (pcs_box, dev_box) if is_b2a else (dev_box, pcs_box)
    enc = _encode(stages, in_box, out_box, samples)
    if enc is None:
        return None

    if legacy:
        in_ch, out_ch = (3, n) if is_b2a else (n, 3)
        grid = np.linspace(0.0, 1.0, cv.TRC_SAMPLES)
        in_tables = np.array([np.nan_to_num(f(grid)) for f in enc["IN"]])
        out_tables = np.array([np.nan_to_num(f(grid)) for f in enc["OUT"]])
        return tt.encode_lut16(
            in_ch,
            out_ch,
            2,
            np.eye(3),
            np.clip(in_tables, 0.0, 1.0),
            _clut_from_affine(*enc["CLUT"]),
            np.clip(out_tables, 0.0, 1.0),
        )

    parts = {}
    for slot in ("A", "M", "B"):
        parts[f"{slot}_curves"] = _curves(enc[slot])
    use_clut = not (
        _is_identity(enc["A"])
        and enc["CLUT"][0].shape[0] == enc["CLUT"][0].shape[1]
        and np.allclose(enc["CLUT"][0], np.eye(enc["CLUT"][0].shape[0]))
        and np.allclose(enc["CLUT"][1], 0.0)
    )
    use_matrix = not (
        _is_identity(enc["M"])
        and np.allclose(enc["MATRIX"][0], np.eye(3))
        and np.allclose(enc["MATRIX"][1], 0.0)
    )
    return tt.encode_lutAtoB_or_BtoA(
        TYPE_MBA if is_b2a else TYPE_MAB,
        3 if is_b2a else n,
        n if is_b2a else 3,
        A_curves=parts["A_curves"] if use_clut else None,
        clut=_clut_from_affine(*enc["CLUT"]) if use_clut else None,
        M_curves=parts["M_curves"] if use_matrix else None,
        matrix=enc["MATRIX"][0].T if use_matrix else None,
        matrix_offset=enc["MATRIX"][1] if use_matrix else None,
        B_curves=parts["B_curves"],
    )
