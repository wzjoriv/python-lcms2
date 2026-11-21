"""A corpus of user-written colour spaces, written to ICC profiles and checked three ways.

Every space is written with ``to_icc_profile``; then

(a) the method must be the most exact one available (recorded per space),
(b) Little CMS loading the bytes must reproduce the NumPy functions, and
(c) the pure-Python reader ``lcms2.icc.ICCSpace`` must evaluate the profile back to the same values.

The same spaces are also written as v5 profiles, which must never need a sampled fallback unless a
space is listed as untraceable.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pytest

import lcms2
from lcms2.icc import ICCSpace
from lcms2.spaces import ColorSpace, SRGBSpace

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

M = SRGBSpace.matrix
M_INV = np.linalg.inv(M)
WHITE = M @ np.ones(3)

#: Tolerances (absolute, on XYZ with Y = 1 or on device values) per representation.
TOL = {
    "exact_matrix_trc": 5e-4,
    "exact_gray_trc": 5e-4,
    "exact_lut": 4e-3,
    "exact_mpet": 1e-6,
    "exact_calc": 1e-6,
    "clut_fallback": 3e-2,
}

TRC, GRAY, LUT, CLUT = "exact_matrix_trc", "exact_gray_trc", "exact_lut", "clut_fallback"


@dataclass
class Case:
    """One user-written space and what the writer is expected to do with it."""

    id: str
    n: int
    to_xyz: object
    from_xyz: object
    method: str
    domain: object = (0.0, 1.0)
    circular: tuple = ()
    v5: tuple = ("exact_mpet", "exact_calc")
    tol: float | None = None
    #: tolerance of the XYZ -> device -> XYZ round trip (defaults to 4x `tol`)
    back_tol: float | None = None
    #: False when several device values give one XYZ, so the inverse is compared directly
    injective: bool = True
    #: Little CMS interpolates hue linearly across the wrap, so its inverse is not compared
    lcms_inverse: bool = True
    #: builds the space itself (for subclasses) instead of wrapping the two callables
    factory: object = None


CASES: list[Case] = []


def case(*args, **kwargs):
    CASES.append(Case(*args, **kwargs))


# ---------------------------------------------------------------------------------------------
# building blocks


def dec(x):
    """sRGB decode written the way users write it: np.where over both branches."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def enc(y):
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


def rgb(id, curve, inverse, method, **kw):
    """Device RGB through per-channel curves and the sRGB primaries."""
    case(id, 3, lambda x: curve(x) @ M.T, lambda xyz: inverse(xyz @ M_INV.T), method, **kw)


# ---------------------------------------------------------------------------------------------
# matrix + curves

rgb("gamma22", lambda x: x**2.2, lambda y: y ** (1 / 2.2), TRC)
rgb("gamma_np_power", lambda x: np.power(x, 1.8), lambda y: np.power(y, 1 / 1.8), TRC)
rgb("gamma_int_exponent", lambda x: x**2, lambda y: np.sqrt(y), TRC)
rgb("square_sqrt", np.square, np.sqrt, TRC)
rgb("srgb_where", dec, enc, TRC)
rgb(
    "srgb_sign_abs",
    lambda x: np.sign(x) * dec(np.abs(x)),
    lambda y: np.sign(y) * enc(np.abs(y)),
    TRC,
)
rgb(
    "clip_both_sides",
    lambda x: dec(np.clip(x, 0.0, 1.0)),
    lambda y: np.clip(enc(np.clip(y, 0.0, 1.0)), 0.0, 1.0),
    TRC,
)
rgb(
    "maximum_minimum",
    lambda x: np.minimum(np.maximum(x, 0.0), 1.0) ** 2.4,
    lambda y: np.minimum(np.maximum(y, 0.0), 1.0) ** (1 / 2.4),
    TRC,
)
rgb(
    "per_channel_gamma",
    lambda x: x ** np.array([1.8, 2.2, 2.6]),
    lambda y: y ** (1 / np.array([1.8, 2.2, 2.6])),
    TRC,
)
rgb(
    "exp_log_curve",
    lambda x: np.exp(2.2 * np.log(0.05 + 0.95 * x)),
    lambda y: (np.exp(np.log(y) / 2.2) - 0.05) / 0.95,
    TRC,
)
rgb("cbrt_cube", lambda x: np.cbrt(x), lambda y: y**3, TRC)
rgb("fractional_exponent", lambda x: x**0.45, lambda y: y ** (1 / 0.45), TRC)
rgb(
    "float32_constants",
    lambda x: x.astype(np.float64) ** np.float32(2.2),
    lambda y: y ** np.float32(1 / 2.2),
    TRC,
    tol=2e-3,
)
rgb(
    "dtype_cast_asarray",
    lambda x: np.asarray(x, dtype=np.float64) ** 2.4,
    lambda y: np.asarray(y, dtype=float) ** (1 / 2.4),
    TRC,
)
rgb(
    "rpow_exp2",
    lambda x: 2.0**x - 1.0,
    lambda y: np.log2(y + 1.0),
    TRC,
)
rgb(
    "log10_curve",
    lambda x: (10.0**x - 1.0) / 9.0,
    lambda y: np.log10(1.0 + 9.0 * y),
    TRC,
)
rgb(
    "log1p_expm1",
    lambda x: np.expm1(x * np.log(3.0)) / 2.0,
    lambda y: np.log1p(2.0 * y) / np.log(3.0),
    TRC,
)
rgb(
    "tanh_arctanh",
    lambda x: np.tanh(2.0 * x) / np.tanh(2.0),
    lambda y: np.arctanh(np.clip(y, 0.0, 1.0) * np.tanh(2.0)) / 2.0,
    TRC,
    tol=2e-3,
)
rgb(
    "np_interp_table",
    lambda x: np.interp(x, [0.0, 0.25, 0.5, 1.0], [0.0, 0.1, 0.4, 1.0]),
    lambda y: np.interp(y, [0.0, 0.1, 0.4, 1.0], [0.0, 0.25, 0.5, 1.0]),
    TRC,
)
rgb(
    "np_select_piecewise",
    lambda x: np.select([x < 0.2, x < 0.6], [x * 0.5, 0.1 + (x - 0.2) * 1.25], 0.6 + (x - 0.6) * 1.0),
    lambda y: np.select([y < 0.1, y < 0.6], [y * 2.0, 0.2 + (y - 0.1) * 0.8], 0.6 + (y - 0.6) * 1.0),
    TRC,
)
rgb(
    "nested_where",
    lambda x: np.where(
        x < 0.3, x * 0.5, np.where(x < 0.7, 0.15 + (x - 0.3) * 1.5, 0.75 + (x - 0.7) * 0.8333333333333334)
    ),
    lambda y: np.where(y < 0.15, y * 2.0, np.where(y < 0.75, 0.3 + (y - 0.15) / 1.5, 0.7 + (y - 0.75) * 1.2)),
    TRC,
)
rgb(
    "limited_range_levels",
    lambda x: dec((x * 219.0 + 16.0) / 255.0),
    lambda y: (enc(y) * 255.0 - 16.0) / 219.0,
    TRC,
)
rgb(
    "int_and_bool_scalars",
    lambda x: (x * 1 + 0) ** 2.0 * True,
    lambda y: y**0.5,
    TRC,
)

# matrix forms -------------------------------------------------------------------------------

case("matmul_operator", 3, lambda x: x @ M.T, lambda xyz: xyz @ M_INV.T, TRC)
case(
    "matrix_as_lists",
    3,
    lambda x: x @ np.array(M.tolist()).T,
    lambda xyz: xyz @ np.array(M_INV.tolist()).T,
    TRC,
)
case("np_dot", 3, lambda x: np.dot(x**2.2, M.T), lambda xyz: np.dot(xyz, M_INV.T) ** (1 / 2.2), TRC)
case(
    "np_matmul_func",
    3,
    lambda x: np.matmul(x**2.2, M.T),
    lambda xyz: np.matmul(xyz, M_INV.T) ** (1 / 2.2),
    TRC,
)
case(
    "einsum_ij",
    3,
    lambda x: np.einsum("...j,ij->...i", x**2.2, M),
    lambda xyz: np.einsum("...j,ij->...i", xyz, M_INV) ** (1 / 2.2),
    TRC,
)
case(
    "einsum_leading_matrix",
    3,
    lambda x: np.einsum("ij,...j->...i", M, x**2.2),
    lambda xyz: np.einsum("ij,...j->...i", M_INV, xyz) ** (1 / 2.2),
    TRC,
)
case(
    "matrix_left_multiply_transposed",
    3,
    lambda x: (x**2.2) @ M.T,
    lambda xyz: (M_INV @ xyz.T).T ** (1 / 2.2) if xyz.ndim == 2 else (M_INV @ xyz) ** (1 / 2.2),
    TRC,
)

# assembling channels ------------------------------------------------------------------------

case(
    "np_stack_channels",
    3,
    lambda x: np.stack([dec(x[..., 0]), dec(x[..., 1]), dec(x[..., 2])], axis=-1) @ M.T,
    lambda xyz: np.stack([enc(c) for c in np.moveaxis(xyz @ M_INV.T, -1, 0)], axis=-1),
    TRC,
)
case(
    "concatenate_slices",
    3,
    lambda x: np.concatenate([dec(x[..., :2]), dec(x[..., 2:])], axis=-1) @ M.T,
    lambda xyz: np.concatenate([enc((xyz @ M_INV.T)[..., :2]), enc((xyz @ M_INV.T)[..., 2:])], -1),
    TRC,
)
case(
    "reversed_channels_bgr",
    3,
    lambda x: dec(x[..., ::-1]) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T)[..., ::-1],
    TRC,
)
case(
    "fancy_index_permutation",
    3,
    lambda x: dec(x[..., [2, 0, 1]]) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T)[..., [1, 2, 0]],
    TRC,
)
case(
    "negative_indices",
    3,
    lambda x: np.stack([dec(x[..., -3]), dec(x[..., -2]), dec(x[..., -1])], -1) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T),
    TRC,
)
case(
    "column_stack_per_channel",
    3,
    lambda x: np.column_stack([dec(x[..., i]) for i in range(3)]) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T),
    TRC,
)


def _inplace_to(x):
    lin = x.copy()
    lin[..., 0] = dec(x[..., 0])
    lin[..., 1] = dec(x[..., 1])
    lin[..., 2] = dec(x[..., 2])
    return lin @ M.T


def _inplace_from(xyz):
    out = (xyz @ M_INV.T).copy()
    out[...] = enc(out)
    out[..., 0] *= 1.0
    return out


case("inplace_on_copy", 3, _inplace_to, _inplace_from, TRC)


def _inplace_slices_to(x):
    y = np.array(x, dtype=np.float64)
    y[..., :2] = y[..., :2] ** 2.2
    y[..., 2] = y[..., 2] ** 2.2
    y += 0.0
    return y @ M.T


case(
    "inplace_slice_assign_array_ctor",
    3,
    _inplace_slices_to,
    lambda xyz: (xyz @ M_INV.T) ** (1 / 2.2),
    TRC,
)

# helper functions, lambdas and classes --------------------------------------------------------


_CURVES = {0: lambda v: v**1.8, 1: lambda v: v**2.2, 2: lambda v: dec(v)}
_INVERSES = {0: lambda v: v ** (1 / 1.8), 1: lambda v: v ** (1 / 2.2), 2: lambda v: enc(v)}


def _helper_decode(v, gamma):
    return np.where(v <= 0.05, v / 8.0, ((v + 0.05) / 1.05) ** gamma)


def _helper_encode(v, gamma):
    return np.where(v <= 0.05 / 8.0, v * 8.0, 1.05 * v ** (1 / gamma) - 0.05)


def _helper_to(x):
    return _linear_to_xyz(_helper_decode(x, 2.2))


def _linear_to_xyz(rgb_lin):
    return rgb_lin @ M.T


case(
    "nested_helper_functions",
    3,
    _helper_to,
    lambda xyz: _helper_encode(xyz @ M_INV.T, 2.2),
    TRC,
)
case(
    "functools_partial",
    3,
    lambda x: functools.partial(_helper_decode, gamma=2.4)(x) @ M.T,
    lambda xyz: functools.partial(_helper_encode, gamma=2.4)(xyz @ M_INV.T),
    TRC,
)


class _ClassSpace(ColorSpace):
    """A subclass keeping its parameters on ``self`` and splitting the work into helpers."""

    name = "class-corpus"
    n_channels = 3

    def __init__(self):
        super().__init__()
        self.gamma = 2.3
        self.mat = M * 1.0

    def _lin(self, x):
        return np.clip(x, 0.0, 1.0) ** self.gamma

    def _enc(self, y):
        return np.clip(y, 0.0, 1.0) ** (1.0 / self.gamma)

    def to_xyz(self, x):
        return self._lin(x) @ self.mat.T

    def from_xyz(self, xyz):
        return self._enc(xyz @ np.linalg.inv(self.mat).T)


# ---------------------------------------------------------------------------------------------
# non-identity domains and circular channels

case(
    "domain_0_255",
    3,
    lambda x: (x / 255.0) ** 2.2 @ M.T,
    lambda xyz: (xyz @ M_INV.T) ** (1 / 2.2) * 255.0,
    TRC,
    domain=(0.0, 255.0),
)
case(
    "domain_per_channel_signed",
    3,
    lambda x: (x * np.array([1.0, 0.5, 0.25]) + np.array([0.5, 0.5, 0.5])) @ M.T,
    lambda xyz: ((xyz @ M_INV.T) - 0.5) / np.array([1.0, 0.5, 0.25]),
    TRC,
    domain=[(-0.5, 0.5), (-1.0, 1.0), (-2.0, 2.0)],
)
case(
    "domain_gamma_video_levels",
    3,
    lambda x: dec(x / 1023.0) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T) * 1023.0,
    TRC,
    domain=(0.0, 1023.0),
)


def _hsv_to_xyz(x):
    h, s, v = x[..., 0], x[..., 1], x[..., 2]

    def ch(k):
        return np.clip(np.abs((h * 6.0 + k) % 6.0 - 3.0) - 1.0, 0.0, 1.0)

    r, g, b = (v * (1 - s + s * ch(k)) for k in (0.0, 4.0, 2.0))
    return dec(np.stack([r, g, b], axis=-1)) @ M.T


def _xyz_to_hsv(xyz):
    rgb_ = np.clip(enc(xyz @ M_INV.T), 0.0, 1.0)
    mx, mn = rgb_.max(axis=-1), rgb_.min(axis=-1)
    d = mx - mn
    safe = np.where(d == 0, 1.0, d)
    r, g, b = rgb_[..., 0], rgb_[..., 1], rgb_[..., 2]
    h = np.select(
        [mx == r, mx == g],
        [((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        (r - g) / safe + 4.0,
    )
    h = np.where(d == 0, 0.0, h) / 6.0
    s = np.where(mx == 0, 0.0, d / np.where(mx == 0, 1.0, mx))
    return np.stack([h, s, mx], axis=-1)


case(
    "hsv_circular_hue",
    3,
    _hsv_to_xyz,
    _xyz_to_hsv,
    CLUT,
    lcms_inverse=False,
    circular=(0,),
    v5=("exact_calc",),
)


def _lch_to_xyz(x):
    L, c, h = x[..., 0], x[..., 1], x[..., 2]
    a, b = c * np.cos(np.deg2rad(h)), c * np.sin(np.deg2rad(h))
    return _lab_to_xyz(np.stack([L, a, b], axis=-1))


def _xyz_to_lch(xyz):
    lab = _xyz_to_lab(xyz)
    c = np.hypot(lab[..., 1], lab[..., 2])
    h = np.rad2deg(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], c, h], axis=-1)


_EPS, _KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


def _lab_to_xyz(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    return np.where(f**3 > _EPS, f**3, (116.0 * f - 16.0) / _KAPPA) * WHITE


def _xyz_to_lab(xyz):
    t = xyz / WHITE
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1
    )


case(
    "cielab_where_cbrt",
    3,
    _lab_to_xyz,
    _xyz_to_lab,
    CLUT,
    domain=[(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)],
    v5=("exact_calc",),
)
case(
    "lch_polar_circular_degrees",
    3,
    _lch_to_xyz,
    _xyz_to_lch,
    CLUT,
    domain=[(0.0, 100.0), (0.0, 150.0), (0.0, 360.0)],
    lcms_inverse=False,
    circular=(2,),
    v5=("exact_calc",),
)

_M1 = np.array([[0.8189, 0.3619, -0.1288], [0.0330, 0.9293, 0.0361], [0.0482, 0.2643, 0.6338]])
_M2 = np.array([[0.2105, 0.7936, -0.0041], [1.9780, -2.4286, 0.4506], [0.0259, 0.7828, -0.8087]])

_M1_INV, _M2_INV = np.linalg.inv(_M1), np.linalg.inv(_M2)
case(
    "oklab_like_matrix_cbrt_matrix",
    3,
    lambda x: ((x @ _M2_INV.T) ** 3) @ _M1_INV.T,
    lambda xyz: np.cbrt(xyz @ _M1.T) @ _M2.T,
    LUT,
    domain=[(0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)],
)

# ---------------------------------------------------------------------------------------------
# transfer functions

_M1P, _M2P, _C1, _C2, _C3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875


def _pq_decode(x):
    p = x ** (1 / _M2P)
    return (np.maximum(p - _C1, 0.0) / (_C2 - _C3 * p)) ** (1 / _M1P)


def _pq_encode(y):
    p = y**_M1P
    return ((_C1 + _C2 * p) / (1 + _C3 * p)) ** _M2P


rgb("pq_like", _pq_decode, _pq_encode, TRC, tol=2e-3, back_tol=5e-3)

_A, _B, _C = 0.17883277, 0.28466892, 0.55991073


def _hlg_decode(x):
    return np.where(x <= 0.5, x * x / 3.0, (np.exp((x - _C) / _A) + _B) / 12.0)


def _hlg_encode(y):
    y12 = 12.0 * y
    return np.where(y12 <= 1.0, np.sqrt(3.0 * y), _A * np.log(np.maximum(y12 - _B, 1e-9)) + _C)


rgb("hlg_like", _hlg_decode, _hlg_encode, TRC, tol=2e-3, back_tol=5e-3)

# ---------------------------------------------------------------------------------------------
# one-channel spaces


def _gray(id, curve, inverse, method, to_xyz=None, from_xyz=None, **kw):
    to = to_xyz or (lambda x: curve(x) * WHITE)
    frm = from_xyz or (lambda xyz: inverse(xyz[..., 1:2]))
    case(id, 1, to, frm, method, **kw)


_gray("gray_gamma", lambda x: x**2.2, lambda y: y ** (1 / 2.2), GRAY)
_gray("gray_srgb_where", dec, enc, GRAY)
_gray(
    "gray_log_curve",
    lambda x: (10.0**x - 1.0) / 9.0,
    lambda y: np.log10(1.0 + 9.0 * y),
    GRAY,
)
_gray(
    "gray_squeezed_index",
    None,
    None,
    GRAY,
    to_xyz=lambda x: np.stack([x[..., 0] ** 2.2 * w for w in WHITE], axis=-1),
    from_xyz=lambda xyz: (xyz[..., 1] ** (1 / 2.2))[..., None],
)
_gray(
    "gray_concatenate_copies",
    None,
    None,
    GRAY,
    to_xyz=lambda x: np.concatenate([x**2.2 * WHITE[0], x**2.2 * WHITE[1], x**2.2 * WHITE[2]], axis=-1),
    from_xyz=lambda xyz: xyz[..., 1:2] ** (1 / 2.2),
)
_gray(
    "gray_np_repeat",
    None,
    None,
    GRAY,
    to_xyz=lambda x: np.repeat(x**2.2, 3, axis=-1) * WHITE,
    from_xyz=lambda xyz: xyz[..., 1:2] ** (1 / 2.2),
)
_gray(
    "gray_domain_percent",
    None,
    None,
    GRAY,
    to_xyz=lambda x: (x / 100.0) ** 2.2 * WHITE,
    from_xyz=lambda xyz: xyz[..., 1:2] ** (1 / 2.2) * 100.0,
    domain=(0.0, 100.0),
)
_gray(
    "gray_clip_sign_abs",
    lambda x: np.sign(x) * np.clip(np.abs(x), 0.0, 1.0) ** 2.4,
    lambda y: np.sign(y) * np.abs(y) ** (1 / 2.4),
    GRAY,
)

# ---------------------------------------------------------------------------------------------
# two and four channels

_A4 = np.array([[0.4, 0.2, 0.0], [0.1, 0.5, 0.1], [0.0, 0.1, 0.6], [0.2, 0.2, 0.2]])
_A4_PINV = np.linalg.pinv(_A4)
_A2 = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.9]])
_A2_PINV = np.linalg.pinv(_A2)

case("two_channel_linear", 2, lambda x: x @ _A2, lambda xyz: xyz @ _A2_PINV, LUT, tol=None)
case(
    "two_channel_gamma",
    2,
    lambda x: x**2.0 @ _A2,
    lambda xyz: np.sqrt(np.clip(xyz @ _A2_PINV, 0.0, None)),
    LUT,
)
case(
    "four_channel_linear",
    4,
    lambda x: x @ _A4,
    lambda xyz: xyz @ _A4_PINV,
    LUT,
    injective=False,
)
case(
    "four_channel_gamma_matrix",
    4,
    lambda x: np.maximum(x, 0.0) ** 1.8 @ _A4,
    lambda xyz: np.maximum(xyz @ _A4_PINV, 0.0) ** (1 / 1.8),
    LUT,
    injective=False,
)


def _cmyk_to_xyz(x):
    c, m, y, k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    rgb_ = np.stack([(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)], axis=-1)
    return dec(rgb_) @ M.T


def _xyz_to_cmyk(xyz):
    rgb_ = np.clip(enc(xyz @ M_INV.T), 0.0, 1.0)
    k = 1.0 - rgb_.max(axis=-1)
    d = 1.0 - k
    safe = np.where(d == 0, 1.0, d)
    return np.stack([1 - rgb_[..., 0] / safe, 1 - rgb_[..., 1] / safe, 1 - rgb_[..., 2] / safe, k], -1)


case(
    "cmyk_multiplicative",
    4,
    _cmyk_to_xyz,
    _xyz_to_cmyk,
    CLUT,
    v5=("exact_calc",),
    tol=3e-2,
)

# ---------------------------------------------------------------------------------------------
# other nonlinear three-channel spaces

_YXY_DOMAIN = [(0.0, 1.0), (0.2, 0.8), (0.2, 0.8)]


def _yxy_to_xyz(c):
    Y, x, y = c[..., 0], c[..., 1], c[..., 2]
    return np.stack([x * Y / y, Y, (1.0 - x - y) * Y / y], axis=-1)


def _xyz_to_yxy(xyz):
    s = xyz.sum(axis=-1)
    safe = np.where(s == 0, 1.0, s)
    return np.stack([xyz[..., 1], xyz[..., 0] / safe, xyz[..., 1] / safe], axis=-1)


case(
    "xyy_ratio_of_channels",
    3,
    _yxy_to_xyz,
    _xyz_to_yxy,
    CLUT,
    domain=_YXY_DOMAIN,
    v5=("exact_calc",),
)
case(
    "reduce_max_min_identity",
    3,
    lambda x: np.minimum(x, x.max(axis=-1, keepdims=True)) @ M.T,
    lambda xyz: np.maximum(xyz @ M_INV.T, (xyz @ M_INV.T).min(axis=-1, keepdims=True)),
    TRC,
)
case(
    "sum_normalised_luma",
    3,
    lambda x: (x + 0.0 * np.sum(x, axis=-1, keepdims=True) + 0.0 * np.mean(x, axis=-1, keepdims=True)) @ M.T,
    lambda xyz: xyz @ M_INV.T,
    TRC,
)
case(
    "scalar_affine_broadcast",
    3,
    lambda x: (x * 0.9 + np.array([0.05, 0.02, 0.01])) @ M.T,
    lambda xyz: ((xyz @ M_INV.T) - np.array([0.05, 0.02, 0.01])) / 0.9,
    LUT,
)
case(
    "channel_affine_stack",
    3,
    lambda x: np.stack([x[..., 0], x[..., 1] * 0.5 + 0.25, x[..., 2] * 0.5 + 0.25], -1) @ M.T,
    lambda xyz: np.stack(
        [(xyz @ M_INV.T)[..., 0], ((xyz @ M_INV.T)[..., 1] - 0.25) * 2, ((xyz @ M_INV.T)[..., 2] - 0.25) * 2],
        -1,
    ),
    LUT,
)


def _soft_to(x):
    return np.clip(x, np.array([0.0, 0.05, 0.1]), np.array([1.0, 0.95, 0.9])) @ M.T


case(
    "clip_with_array_bounds",
    3,
    _soft_to,
    lambda xyz: xyz @ M_INV.T,
    TRC,
)


# more ways users spell the same things ------------------------------------------------------

case(
    "unpack_channels_by_indexing",
    3,
    lambda x: np.stack([dec(x[..., 0]), dec(x[..., 1]), dec(x[..., 2])], axis=-1) @ M.T,
    lambda xyz: enc(xyz @ M_INV.T),
    TRC,
)


def _moveaxis_to(x):
    r, g, b = np.moveaxis(x, -1, 0)
    return np.stack([dec(r), dec(g), dec(b)], axis=-1) @ M.T


case("moveaxis_unpack", 3, _moveaxis_to, lambda xyz: enc(xyz @ M_INV.T), TRC)


def _transpose_to(x):
    r, g, b = x.T
    return np.stack([r**2.2, g**2.2, b**2.2], axis=-1) @ M.T


case("transpose_unpack", 3, _transpose_to, lambda xyz: (xyz @ M_INV.T) ** (1 / 2.2), TRC)
case(
    "numpy_scalar_left_operand",
    3,
    lambda x: (np.float64(2.0) ** x - 1.0) @ M.T,
    lambda xyz: np.log2(xyz @ M_INV.T + np.float64(1.0)),
    TRC,
)
case(
    "clip_method_and_keywords",
    3,
    lambda x: np.clip(x.clip(0.0, 1.0), a_min=0.0, a_max=1.0) ** 2.2 @ M.T,
    lambda xyz: np.clip(xyz @ M_INV.T, a_min=0.0, a_max=1.0) ** (1 / 2.2),
    TRC,
)
case(
    "tensordot_matrix",
    3,
    lambda x: np.tensordot(x**2.2, M, axes=([-1], [1])),
    lambda xyz: np.tensordot(xyz, M_INV, axes=([-1], [1])) ** (1 / 2.2),
    TRC,
)
case(
    "float32_astype_roundtrip",
    3,
    lambda x: (x.astype(np.float32) ** 2.2).astype(np.float64) @ M.T,
    lambda xyz: (xyz @ M_INV.T) ** (1 / 2.2),
    TRC,
    tol=2e-3,
)
case(
    "dict_of_lambdas_per_channel",
    3,
    lambda x: np.stack([_CURVES[i](x[..., i]) for i in range(3)], axis=-1) @ M.T,
    lambda xyz: np.stack([_INVERSES[i]((xyz @ M_INV.T)[..., i]) for i in range(3)], axis=-1),
    TRC,
)
case(
    "where_condition_on_one_channel",
    3,
    lambda x: np.where(x[..., :1] > 2.0, x, x**2.2) @ M.T,
    lambda xyz: (xyz @ M_INV.T) ** (1 / 2.2),
    TRC,
)
case(
    "exp_log_power_through_zero",
    3,
    lambda x: np.exp(2.0 * np.log(x)) @ M.T,
    lambda xyz: np.exp(np.log(np.maximum(xyz @ M_INV.T, 1e-30)) / 2.0),
    TRC,
)
case(
    "division_by_channel_sum",
    3,
    lambda x: (x / (1.0 + x.sum(axis=-1, keepdims=True))) @ M.T,
    lambda xyz: (lambda y: y * (1.0 + y.sum(axis=-1, keepdims=True) / (1.0 - y.sum(axis=-1, keepdims=True))))(
        xyz @ M_INV.T
    ),
    CLUT,
    v5=("exact_calc",),
)
case(
    "channel_ratio_product",
    3,
    lambda x: np.stack([x[..., 0] * x[..., 2], x[..., 1], x[..., 2]], axis=-1),
    lambda xyz: np.stack([xyz[..., 0] / xyz[..., 2], xyz[..., 1], xyz[..., 2]], axis=-1),
    CLUT,
    domain=[(0.0, 1.0), (0.0, 1.0), (0.2, 1.0)],
    v5=("exact_calc",),
)
_gray(
    "gray_weighted_sum",
    None,
    None,
    GRAY,
    to_xyz=lambda x: (x**2.2) * WHITE,
    from_xyz=lambda xyz: np.sum(xyz * np.array([0.0, 1.0, 0.0]), axis=-1, keepdims=True) ** (1 / 2.2),
)
case(
    "average_weights_identity",
    3,
    lambda x: (x + 0.0 * np.average(x, weights=[1.0, 2.0, 3.0], axis=-1)[..., None]) @ M.T,
    lambda xyz: xyz @ M_INV.T,
    TRC,
)


# a subclass that keeps its parameters on ``self`` and splits the work into helper methods
case("subclass_with_helper_methods", 3, None, None, TRC, factory=_ClassSpace)


# ---------------------------------------------------------------------------------------------
# the tests

GRID_POINTS = {1: 33, 2: 9, 3: 6, 4: 4}

#: PCS values a v2/v4 profile can encode (u1Fixed15).
XYZ_MAX = 1.99


def pairs(c):
    d = c.domain
    if isinstance(d, tuple) and not isinstance(d[0], (tuple, list)):
        return [tuple(d)] * c.n
    return [tuple(p) for p in d]


def make_space(c):
    if c.factory is not None:
        return c.factory()
    space = ColorSpace(
        n_channels=c.n, to_xyz=c.to_xyz, from_xyz=c.from_xyz, name=c.id, device_domain=c.domain
    )
    if c.circular:
        space.circular_device_channels = frozenset(c.circular)
    return space


def write(c, version, **kwargs):
    space = make_space(c)
    return space, space.to_icc_profile(version=version, **kwargs)


def samples(c, space):
    """``(normalized device, device, XYZ)`` for a grid of device values the profile can encode."""
    axis = np.linspace(0.02, 0.98, GRID_POINTS[c.n])
    g01 = np.array(list(itertools.product(axis, repeat=c.n)))
    lo = np.array([p[0] for p in pairs(c)])
    span = np.array([p[1] - p[0] for p in pairs(c)])
    dev = lo + g01 * span
    with np.errstate(all="ignore"):
        xyz = np.asarray(space.to_xyz(dev), dtype=np.float64)
    keep = np.all(np.isfinite(xyz) & (xyz >= 0.0) & (xyz <= XYZ_MAX), axis=-1)
    assert keep.sum() >= 0.3 * len(keep), "the grid is mostly outside what a profile can encode"
    return g01[keep], dev[keep], xyz[keep]


def back_error(c, space, xyz, got_dev):
    """How far device values `got_dev` found for `xyz` are from what the NumPy functions give."""
    if c.injective:
        with np.errstate(all="ignore"):
            again = np.asarray(space.to_xyz(got_dev), dtype=np.float64)
        return float(np.max(np.abs(again - xyz)))
    lo = np.array([p[0] for p in pairs(c)])
    span = np.array([p[1] - p[0] for p in pairs(c)])
    with np.errstate(all="ignore"):
        want = (np.asarray(space.from_xyz(xyz), dtype=np.float64) - lo) / span
    inside = np.all((want >= 0.0) & (want <= 1.0), axis=-1)
    assert inside.sum() >= 4, "no sample has an inverse inside the device domain"
    return float(np.max(np.abs((got_dev[inside] - lo) / span - want[inside])))


def ids(cases):
    return [c.id for c in cases]


def test_corpus_is_large_enough():
    assert len(CASES) >= 40
    assert len({c.id for c in CASES}) == len(CASES)
    assert {1, 2, 3, 4} <= {c.n for c in CASES}


@pytest.mark.parametrize("c", CASES, ids=ids(CASES))
def test_method_is_the_most_exact_available(c):
    _, result = write(c, 4)
    assert result["method"] == c.method, result["notes"]


@pytest.mark.parametrize("c", CASES, ids=ids(CASES))
def test_lcms_reproduces_numpy_forward(c):
    space, result = write(c, 4)
    _, dev, want = samples(c, space)
    profile = lcms2.Profile(result["bytes"])
    got = lcms2.convert(dev, profile, "XYZ", intent="relative", bpc=False).reshape(want.shape)
    assert np.max(np.abs(got - want)) <= (c.tol or TOL[result["method"]])


@pytest.mark.parametrize("c", CASES, ids=ids(CASES))
def test_lcms_inverse_matches_numpy(c):
    if not c.lcms_inverse:
        pytest.skip("circular channel: Little CMS does not wrap hue when interpolating")
    space, result = write(c, 4)
    _, dev, xyz = samples(c, space)
    profile = lcms2.Profile(result["bytes"])
    back = lcms2.convert(xyz, "XYZ", profile, intent="relative", bpc=False).reshape(dev.shape)
    tol = c.back_tol or 4 * (c.tol or TOL[result["method"]])
    assert back_error(c, space, xyz, back) <= tol


@pytest.mark.parametrize("c", CASES, ids=ids(CASES))
def test_icc_space_evaluates_the_profile_back(c):
    space, result = write(c, 4)
    _, dev, want = samples(c, space)
    back = ICCSpace(result["bytes"])
    assert back.n_channels == c.n
    tol = c.tol or TOL[result["method"]]
    assert np.max(np.abs(back.to_xyz(dev) - want)) <= tol
    assert back_error(c, space, want, back.from_xyz(want)) <= (c.back_tol or 4 * tol)


@pytest.mark.parametrize("c", CASES, ids=ids(CASES))
def test_v5_profile_is_exact_and_reads_back(c):
    space, result = write(c, 5)
    assert result["method"] in c.v5, result["notes"]
    _, dev, want = samples(c, space)
    back = ICCSpace(result["bytes"])
    assert np.max(np.abs(back.to_xyz(dev) - want)) <= (c.tol or TOL[result["method"]])
    tol = c.back_tol or 4 * (c.tol or TOL[result["method"]])
    assert back_error(c, space, want, back.from_xyz(want)) <= tol


# ---------------------------------------------------------------------------------------------
# functions that cannot be traced: sampled fallback with a reason, or an error when strict


def _inverse(xyz):
    return enc(xyz @ M_INV.T)


def _finish(lin):
    """Linear RGB to XYZ plus a cross-channel term, so no curve-and-matrix chain can describe it."""
    lin = np.asarray(lin)
    return lin @ M.T + 0.05 * lin[..., 0:1] * lin[..., 1:2] * np.array([0.0, 1.0, 0.0])


def _branch_on_batch_max(x):
    if x.max() > 1.0:
        x = x / 255.0
    return _finish(dec(x))


def _float_of_channel(x):
    return _finish(np.array([dec(float(x[..., 0])), dec(float(x[..., 1])), dec(float(x[..., 2]))]))


def _math_module(x):
    return _finish(np.array([math.pow(x[..., 0], 2.2), math.pow(x[..., 1], 2.2), math.pow(x[..., 2], 2.2)]))


def _item_call(x):
    return _finish(np.array([x[..., 0].item() ** 2.2, x[..., 1].item() ** 2.2, x[..., 2].item() ** 2.2]))


def _tolist_call(x):
    return _finish(np.array(x.tolist()) ** 2.2)


def _unpack_by_iteration(x):
    r, g, b = x
    return _finish(np.array([r**2.2, g**2.2, b**2.2]))


def _comprehension_over_array(x):
    return _finish(np.array([c**2.2 for c in x]))


def _random_numbers(x):
    return x**2.2 @ M.T + 1e-4 * np.random.random(3)


def _random_generator(x):
    return x**2.2 @ M.T + 1e-4 * np.random.default_rng().random(3)


def _colorsys_library(x):
    import colorsys

    h, s, v = colorsys.rgb_to_hsv(x[..., 0], x[..., 1], x[..., 2])
    return _finish(dec(np.array(colorsys.hsv_to_rgb(h, s, v))))


def _mask_assignment(x):
    y = x.copy()
    y[y < 0.0] = 0.0
    return _finish(y**2.2)


def _any_branch(x):
    if np.any(x < 0.0):
        x = np.abs(x)
    return _finish(x**2.2)


def _while_loop(x):
    y = x * 0.5
    while y.max() > 1.0:
        y = y / 2.0
    return _finish(y**2.2)


def _sorts_channels(x):
    return _finish(np.sort(x, axis=-1))


def _unsupported_ufunc(x):
    return _finish(np.logaddexp(x, 0.5))


def _quantize(x):
    return _finish((x * 255.0).astype(np.uint8) / 255.0)


def _out_parameter(x):
    y = np.empty_like(x)
    np.multiply(x, 0.5, out=y)
    return _finish(y**2.2)


def _vectorized_python(x):
    return _finish(np.vectorize(lambda v: v**2.2)(x))


def _from_xyz_branches(xyz):
    if xyz.sum() < 0:
        return xyz
    return enc(xyz @ M_INV.T)


#: (id, to_xyz, from_xyz, text the reason must contain)
UNTRACEABLE = [
    ("if_on_batch_value", _branch_on_batch_max, _inverse, "over all axes"),
    ("float_of_channel", _float_of_channel, _inverse, "float()"),
    ("math_module_pow", _math_module, _inverse, "math.*"),
    ("item_call", _item_call, _inverse, ".item()"),
    ("tolist_call", _tolist_call, _inverse, ".tolist()"),
    ("unpack_by_iteration", _unpack_by_iteration, _inverse, "iterating"),
    ("comprehension_over_array", _comprehension_over_array, _inverse, "iterating"),
    ("random_module_function", _random_numbers, _inverse, "np.random.random"),
    ("random_generator_state", _random_generator, _inverse, "different results"),
    ("colorsys_library", _colorsys_library, _inverse, "branching"),
    ("boolean_mask_assignment", _mask_assignment, _inverse, "boolean-mask"),
    ("np_any_condition", _any_branch, _inverse, "numpy function any"),
    ("while_loop_on_values", _while_loop, _inverse, "over all axes"),
    ("np_sort_channels", _sorts_channels, _inverse, "numpy function sort"),
    ("unsupported_ufunc", _unsupported_ufunc, _inverse, "logaddexp"),
    ("astype_integer_quantize", _quantize, _inverse, "astype(uint8)"),
    ("ufunc_out_parameter", _out_parameter, _inverse, "out="),
    ("np_vectorize", _vectorized_python, _inverse, "np.vectorize"),
    ("from_xyz_branches", lambda x: _finish(dec(x)), _from_xyz_branches, "from_xyz is not traceable"),
]


def untraceable_space(to_xyz, from_xyz):
    return ColorSpace(n_channels=3, to_xyz=to_xyz, from_xyz=from_xyz, name="untraceable")


@pytest.mark.parametrize("entry", UNTRACEABLE, ids=[e[0] for e in UNTRACEABLE])
@pytest.mark.parametrize("version", [4, 5])
def test_untraceable_falls_back_to_clut_and_says_why(entry, version):
    _, to_xyz, from_xyz, needle = entry
    space = untraceable_space(to_xyz, from_xyz)
    result = space.to_icc_profile(version=version)
    assert result["method"] == CLUT
    reasons = [n for n in result["notes"] if n.startswith("sampled fallback")]
    assert reasons and needle in " ".join(reasons), result["notes"]
    # the fallback is a faithful sample of what the functions compute
    dev = np.linspace(0.05, 0.95, 5)[:, None] * np.array([1.0, 0.8, 0.6])
    back = ICCSpace(result["bytes"])
    with np.errstate(all="ignore"):
        want = np.array([np.asarray(to_xyz(row)) for row in dev])
    got = back.to_xyz(dev)
    noise = 2e-3 if entry[0].startswith("random") else 0.0
    assert np.max(np.abs(got - want)) <= 3e-2 + noise


@pytest.mark.parametrize("entry", UNTRACEABLE, ids=[e[0] for e in UNTRACEABLE])
def test_strict_raises_naming_the_operation(entry):
    _, to_xyz, from_xyz, needle = entry
    space = untraceable_space(to_xyz, from_xyz)
    for call in (
        lambda: space.to_icc_profile(strict=True),
        lambda: space.to_icc_profile(strict=True, version=5),
        lambda: lcms2.Profile.from_space(space, strict=True),
    ):
        with pytest.raises(lcms2.icc.ICCWriteError) as info:
            call()
        assert needle in str(info.value)
    # without strict the profile is still made, from the samples
    assert lcms2.Profile(space).color_space == "RGB"
    assert lcms2.Profile.from_space(space).color_space == "RGB"


def test_errors_are_cms_errors():
    assert issubclass(lcms2.ICCWriteError, lcms2.CMSError)
    assert issubclass(lcms2.ICCParseError, lcms2.CMSError)
    assert lcms2.icc.ICCWriteError is lcms2.ICCWriteError
    assert lcms2.icc.ICCParseError is lcms2.ICCParseError


def test_strict_accepts_exact_spaces():
    srgb = untraceable_space(lambda x: dec(x) @ M.T, _inverse)
    result = srgb.to_icc_profile(strict=True)
    assert result["method"] == TRC
    assert not any(n.startswith("sampled fallback") for n in result["notes"])
    assert lcms2.Profile.from_space(srgb, strict=True).color_space == "RGB"


@pytest.mark.parametrize("c", [c for c in CASES if c.method != CLUT], ids=lambda c: c.id)
def test_strict_accepts_every_exact_corpus_space(c):
    space = make_space(c)
    assert space.to_icc_profile(strict=True)["method"] == c.method


def test_strict_names_the_version_limit_for_traceable_spaces():
    hsv = make_space(next(c for c in CASES if c.id == "hsv_circular_hue"))
    with pytest.raises(lcms2.ICCWriteError, match="version=5"):
        hsv.to_icc_profile(strict=True)
    assert hsv.to_icc_profile(strict=True, version=5)["method"] == "exact_calc"


def test_force_clut_with_strict_still_writes_the_sampled_profile():
    space = untraceable_space(lambda x: dec(x) @ M.T, _inverse)
    assert space.to_icc_profile(force_clut=True, strict=True)["method"] == CLUT


def test_pixelwise_python_functions_are_sampled_pixel_by_pixel():
    calls = []

    def one_pixel_only(x):
        calls.append(np.shape(x))
        return _finish(np.array([float(x[0]) ** 2.2, float(x[1]) ** 2.2, float(x[2]) ** 2.2]))

    space = untraceable_space(one_pixel_only, _inverse)
    assert space.to_icc_profile()["method"] == CLUT
    assert (3,) in calls


def test_a_function_that_always_raises_is_a_clear_error():
    def broken(x):
        raise RuntimeError("no luck")

    space = untraceable_space(broken, _inverse)
    with pytest.raises(lcms2.ICCWriteError, match="to_xyz failed.*no luck"):
        space.to_icc_profile()


def test_a_function_without_finite_values_is_a_clear_error():
    space = untraceable_space(lambda x: x * np.nan, _inverse)
    with pytest.raises(lcms2.ICCWriteError, match="finite"):
        space.to_icc_profile()


def test_wrong_output_width_is_a_clear_error():
    space = ColorSpace(n_channels=3, to_xyz=lambda x: x[..., :2], from_xyz=_inverse, name="two-wide")
    with pytest.raises(ValueError):
        space.to_icc_profile()


def test_tracing_does_not_leave_numpy_patched():
    before = (np.asarray, np.array, np.random.random)
    untraceable_space(_random_numbers, _inverse).to_icc_profile()
    assert (np.asarray, np.array, np.random.random) == before
    assert isinstance(np.random.random(2), np.ndarray)
