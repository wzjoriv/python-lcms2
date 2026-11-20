"""v2/v4 matrix/TRC and gray TRC tags."""

from __future__ import annotations

import numpy as np

from ...colorimetry import D50
from .._io import tagtypes as tt
from . import curves as cv
from .context import Context


def _unit_curves(curves, matrix):
    """Rescale curves to peak at 1 on [0, 1], folding the scale into `matrix`."""
    t = np.linspace(0.0, 1.0, cv.TRC_SAMPLES)
    scaled_curves, scaled_rows = [], []
    for c, row in zip(curves, matrix, strict=False):
        vals = cv.evaluate(c, t)
        if not np.all(np.isfinite(vals)):
            return None
        peak = float(np.max(np.abs(vals)))
        if peak == 0.0 or np.min(vals) < -1e-9 * peak:
            return None
        scaled_curves.append(lambda x, c=c, p=peak: cv.evaluate(c, x) / p)
        scaled_rows.append(np.asarray(row) * peak)
    return scaled_curves, np.array(scaled_rows)


def write_matrix_trc(ctx: Context, tags: dict, notes: list) -> bool:
    """Write rXYZ/gXYZ/bXYZ and TRC tags when the space is curves then a 3x3 matrix."""
    if not ctx.inverts_forward():
        return False
    found = ctx.forward_curves_matrix()
    if found is None or np.any(np.abs(found[2]) > 1e-9):
        return False
    unit = _unit_curves(found[0], found[1])
    if unit is None:
        return False
    curves, matrix = unit
    if not np.isfinite(np.linalg.cond(matrix)) or np.linalg.cond(matrix) > 1e8:
        return False
    for i, (xyz_sig, trc_sig) in enumerate(((b"rXYZ", b"rTRC"), (b"gXYZ", b"gTRC"), (b"bXYZ", b"bTRC"))):
        tags[xyz_sig] = tt.encode_XYZType([tuple(matrix[i])])
        tags[trc_sig] = cv.trc_tag(curves[i], ctx.major)
    notes.append("per-channel curves followed by a 3x3 matrix, written as rXYZ/gXYZ/bXYZ + TRCs")
    return True


def write_gray_trc(ctx: Context, tags: dict, notes: list) -> bool:
    """Write a kTRC tag when the space is one curve along the D50 white."""
    if not ctx.inverts_forward():
        return False
    found = ctx.forward_curves_matrix()
    if found is None or np.any(np.abs(found[2]) > 1e-9):
        return False
    curve, row = found[0][0], np.asarray(found[1][0])
    white = np.asarray(D50)
    k = row[1] / white[1]
    if k <= 0 or not np.allclose(row, k * white, rtol=1e-4, atol=1e-9):
        return False
    t = np.linspace(0.0, 1.0, cv.TRC_SAMPLES)
    vals = cv.evaluate(curve, t) * k
    if not np.all(np.isfinite(vals)) or vals.min() < -1e-9 or vals.max() > 1.0 + 1e-6:
        return False
    tags[b"kTRC"] = cv.trc_tag(lambda x: cv.evaluate(curve, x) * k, ctx.major)
    notes.append("one curve scaled along the D50 white, written as a kTRC gray profile")
    return True
