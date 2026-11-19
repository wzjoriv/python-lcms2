"""Build ``to_xyz``/``from_xyz`` callables from a parsed ICC profile."""

from __future__ import annotations

import struct

import numpy as np

from ..spaces import _colorimetry as col
from ._errors import ICCParseError
from ._interp import sample_clut
from ._io import binary as bn
from ._io import mpe
from ._io import tagtypes as tt
from ._io.constants import (
    CS_GRAY,
    CS_LAB,
    CS_XYZ,
    TAG_BTRC,
    TAG_BXYZ,
    TAG_GTRC,
    TAG_GXYZ,
    TAG_KTRC,
    TAG_RTRC,
    TAG_RXYZ,
    TYPE_CURV,
    TYPE_MAB,
    TYPE_MBA,
    TYPE_MFT1,
    TYPE_MFT2,
    TYPE_MPET,
    TYPE_PARA,
)

XYZ_U1F15_SCALE = 65535.0 / 32768.0

_A2B_PREFERENCE = (b"A2B1", b"A2B0", b"A2B2")
_B2A_PREFERENCE = (b"B2A1", b"B2A0", b"B2A2")


class BuiltTransform:
    """Device <-> XYZ callables derived from a profile.

    Attributes
    ----------
    to_xyz, from_xyz : callable
        Device values to PCS XYZ, and back.
    kind : str
        ``'matrix_trc'``, ``'gray_trc'`` or ``'lut'``.
    n_channels : int
        Number of device channels.
    """

    def __init__(self, to_xyz, from_xyz, kind: str, n_channels: int):
        self.to_xyz = to_xyz
        self.from_xyz = from_xyz
        self.kind = kind
        self.n_channels = n_channels


def build_transform(profile) -> BuiltTransform:
    """Derive ``to_xyz``/``from_xyz`` from a parsed profile.

    Parameters
    ----------
    profile : ICCProfile

    Returns
    -------
    BuiltTransform

    Raises
    ------
    ICCParseError
        If the profile has no LUT, matrix/TRC or gray TRC model, or a tag
        of that model is malformed.
    """
    try:
        built = _build(profile)
    except ICCParseError:
        raise
    except _DECODE_ERRORS as e:
        raise ICCParseError(f"malformed profile tag: {e}") from e
    built.to_xyz = _guarded(built.to_xyz)
    built.from_xyz = _guarded(built.from_xyz)
    return built


_DECODE_ERRORS = (ValueError, IndexError, KeyError, struct.error, np.linalg.LinAlgError)


def _guarded(fn):
    def call(values):
        try:
            return fn(values)
        except ICCParseError:
            raise
        except _DECODE_ERRORS as e:
            raise ICCParseError(f"malformed profile data: {e}") from e

    return call


def _build(profile) -> BuiltTransform:
    if _first_tag(profile, _A2B_PREFERENCE + _B2A_PREFERENCE) is not None:
        return _build_lut(profile)
    if all(
        profile.has_tag(t) for t in (TAG_RXYZ, TAG_GXYZ, TAG_BXYZ, TAG_RTRC, TAG_GTRC, TAG_BTRC)
    ):
        return _build_matrix_trc(profile)
    if profile.has_tag(TAG_KTRC) and profile.header.data_colour_space == CS_GRAY:
        return _build_gray_trc(profile)
    raise ICCParseError(
        "profile has no A2B/B2A LUT, matrix/TRC or gray TRC model "
        f"(device_class={profile.header.device_class!r}, "
        f"space={profile.header.data_colour_space!r})"
    )


def _first_tag(profile, sigs):
    for sig in sigs:
        raw = profile.get_tag(sig)
        if raw is not None:
            return raw
    return None


def _curve_from_tag_bytes(raw: bytes):
    tsig = raw[0:4]
    if tsig == TYPE_CURV:
        return ("curv", tt.decode_curv(raw))
    if tsig == TYPE_PARA:
        return ("para", tt.decode_para(raw))
    raise ICCParseError(f"unsupported TRC tag type {tsig!r}")


def _apply_curves(curves, x: np.ndarray) -> np.ndarray:
    return np.stack([tt.eval_curve(c, x[..., i]) for i, c in enumerate(curves)], axis=-1)


def _build_matrix_trc(profile) -> BuiltTransform:
    columns = [tt.decode_XYZType(profile.get_tag(t))[0] for t in (TAG_RXYZ, TAG_GXYZ, TAG_BXYZ)]
    matrix = np.array(columns).T
    matrix_inv = np.linalg.inv(matrix)
    curves = [_curve_from_tag_bytes(profile.get_tag(t)) for t in (TAG_RTRC, TAG_GTRC, TAG_BTRC)]

    def to_xyz(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return _apply_curves(curves, rgb) @ matrix.T

    def from_xyz(xyz: np.ndarray) -> np.ndarray:
        lin = np.asarray(xyz, dtype=np.float64) @ matrix_inv.T
        return np.stack(
            [tt.invert_curve_numeric(c, lin[..., i]) for i, c in enumerate(curves)], axis=-1
        )

    return BuiltTransform(to_xyz, from_xyz, kind="matrix_trc", n_channels=3)


def _build_gray_trc(profile) -> BuiltTransform:
    curve = _curve_from_tag_bytes(profile.get_tag(TAG_KTRC))
    white = np.asarray(col.D50)

    def to_xyz(gray: np.ndarray) -> np.ndarray:
        gray = np.asarray(gray, dtype=np.float64)
        return tt.eval_curve(curve, gray[..., 0])[..., None] * white

    def from_xyz(xyz: np.ndarray) -> np.ndarray:
        y = np.asarray(xyz, dtype=np.float64)[..., 1] / white[1]
        return tt.invert_curve_numeric(curve, y)[..., None]

    return BuiltTransform(to_xyz, from_xyz, kind="gray_trc", n_channels=1)


def _classic_lut_apply(d, x: np.ndarray, use_matrix: bool, circular) -> np.ndarray:
    if use_matrix:
        x = np.clip(x @ d["matrix"].T, 0.0, 1.0)
    grid_in = np.linspace(0.0, 1.0, d["in_tables"].shape[1])
    stage = np.stack(
        [np.interp(x[..., i], grid_in, d["in_tables"][i]) for i in range(d["in_ch"])], axis=-1
    )
    out = sample_clut(d["clut"], stage, circular_channels=circular)
    grid_out = np.linspace(0.0, 1.0, d["out_tables"].shape[1])
    return np.stack(
        [np.interp(out[..., i], grid_out, d["out_tables"][i]) for i in range(d["out_ch"])],
        axis=-1,
    )


def _mab_apply(d, x: np.ndarray, is_b2a: bool, circular) -> np.ndarray:
    order = ("B", "MATRIX", "M", "CLUT", "A") if is_b2a else ("A", "CLUT", "M", "MATRIX", "B")
    cur = x
    for stage in order:
        if stage == "CLUT":
            if d["clut"] is not None:
                cur = sample_clut(d["clut"], cur, circular_channels=circular)
        elif stage == "MATRIX":
            if d["matrix"] is not None:
                cur = cur @ d["matrix"].T + d["matrix_offset"]
        else:
            curves = d[f"{stage}_curves"]
            if curves is not None:
                cur = _apply_curves(curves, cur)
    return cur


def _mpet_apply(d, x: np.ndarray, circular) -> np.ndarray:
    cur = x
    elems = d["elements"]
    for idx, elem in enumerate(elems):
        kind = elem["kind"]
        if kind == "matrix":
            cur = cur @ elem["matrix"].T + elem["offset"]
        elif kind == "curveset":
            cur = np.stack(
                [
                    mpe.eval_curveset_channel(elem["curves"][i], cur[..., i])
                    for i in range(cur.shape[-1])
                ],
                axis=-1,
            )
        elif kind == "clut":
            cc = circular if idx == len(elems) - 1 else frozenset()
            cur = sample_clut(elem["clut"], cur, circular_channels=cc)
        elif kind == "calc":
            cur = mpe.eval_calc_element(elem, cur, elem.get("out_ch", cur.shape[-1]))
        else:
            raise ICCParseError(f"unknown mpet element kind {kind}")
    return cur


class LutStage:
    """One decoded A2B*/B2A* tag, evaluated in its own encoded value space."""

    def __init__(self, raw: bytes, pcs: bytes, is_b2a: bool, circular):
        tsig = raw[0:4]
        self.is_b2a = is_b2a
        self.circular = circular if is_b2a else frozenset()
        if tsig in (TYPE_MFT1, TYPE_MFT2):
            self.kind = "classic"
            self.data = tt.decode_lut8(raw) if tsig == TYPE_MFT1 else tt.decode_lut16(raw)
        elif tsig in (TYPE_MAB, TYPE_MBA):
            self.kind = "mab"
            self.data = tt.decode_lutAtoB_or_BtoA(raw)
        elif tsig == TYPE_MPET:
            self.kind = "mpet"
            self.data = mpe.decode_mpet(raw)
        else:
            raise ICCParseError(f"unsupported LUT tag type {tsig!r}")
        self.in_ch = self.data["in_ch"]
        self.out_ch = self.data["out_ch"]
        self._validate()
        self.pcs = pcs
        self.legacy_lab = tsig == TYPE_MFT2
        self.real_valued = self.kind == "mpet"
        self.use_matrix = self.kind == "classic" and is_b2a and pcs == CS_XYZ

    def _validate(self) -> None:
        pcs_side = self.in_ch if self.is_b2a else self.out_ch
        if pcs_side != 3:
            raise ICCParseError(f"LUT has {pcs_side} PCS channels, expected 3")
        if self.in_ch < 1 or self.out_ch < 1:
            raise ICCParseError("LUT has no channels")
        if self.kind == "mab" and self.data["matrix"] is not None:
            width = self.in_ch if self.is_b2a else self.out_ch
            if width != 3:
                raise ICCParseError("mAB/mBA matrix element on a non-3-channel stage")
        if self.kind == "mpet":
            mpe.validate_mpet(self.data)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "classic":
            return _classic_lut_apply(self.data, x, self.use_matrix, self.circular)
        if self.kind == "mab":
            return _mab_apply(self.data, x, self.is_b2a, self.circular)
        return _mpet_apply(self.data, x, self.circular)

    def decode_pcs(self, pcs_values: np.ndarray) -> np.ndarray:
        if self.pcs == CS_XYZ:
            return pcs_values if self.real_valued else pcs_values * XYZ_U1F15_SCALE
        if self.pcs == CS_LAB:
            lab = (
                pcs_values
                if self.real_valued
                else col.decode_lab_pcs(pcs_values, legacy16=self.legacy_lab)
            )
            return col.lab_to_xyz(lab, white=col.D50)
        raise ICCParseError(f"unsupported PCS signature {self.pcs!r}")

    def encode_pcs(self, xyz: np.ndarray) -> np.ndarray:
        if self.pcs == CS_XYZ:
            return xyz if self.real_valued else xyz / XYZ_U1F15_SCALE
        if self.pcs == CS_LAB:
            lab = col.xyz_to_lab(xyz, white=col.D50)
            return lab if self.real_valued else col.encode_lab_pcs(lab, legacy16=self.legacy_lab)
        raise ICCParseError(f"unsupported PCS signature {self.pcs!r}")


def read_circular_channels(profile) -> frozenset:
    """Return the device channels listed in the profile's ``crsC`` tag.

    Parameters
    ----------
    profile : ICCProfile
        A parsed profile.

    Returns
    -------
    frozenset of int
        Empty when the tag is absent.

    Raises
    ------
    ICCParseError
        If the tag is malformed.
    """
    tag = profile.get_tag(b"crsC")
    if tag is None:
        return frozenset()
    count = bn.read_u32(tag, 0) if len(tag) >= 4 else -1
    if count < 0 or len(tag) < 4 + 4 * count:
        raise ICCParseError("malformed 'crsC' circular channel tag")
    return frozenset(bn.read_u32(tag, 4 + 4 * i) for i in range(count))


def read_device_domain(profile, n_channels: int):
    """Return the ``(lo, hi)`` pairs of the profile's ``crsD`` tag.

    Parameters
    ----------
    profile : ICCProfile
        A parsed profile.
    n_channels : int
        Expected channel count.

    Returns
    -------
    list of tuple of float, or None
        One pair per channel; None when the tag is absent.

    Raises
    ------
    ICCParseError
        If the tag is malformed or lists another channel count.
    """
    tag = profile.get_tag(b"crsD")
    if tag is None:
        return None
    count = bn.read_u32(tag, 0) if len(tag) >= 4 else -1
    if count != n_channels or len(tag) < 4 + 8 * count:
        raise ICCParseError("malformed 'crsD' device domain tag")
    return [struct.unpack_from(">ff", tag, 4 + 8 * i) for i in range(count)]


def _build_lut(profile) -> BuiltTransform:
    pcs = profile.header.pcs
    circular = read_circular_channels(profile)
    a2b_raw = _first_tag(profile, _A2B_PREFERENCE)
    b2a_raw = _first_tag(profile, _B2A_PREFERENCE)
    a2b = LutStage(a2b_raw, pcs, False, circular) if a2b_raw is not None else None
    b2a = LutStage(b2a_raw, pcs, True, circular) if b2a_raw is not None else None
    if a2b is not None and b2a is not None and a2b.in_ch != b2a.out_ch:
        raise ICCParseError(f"A2B has {a2b.in_ch} device channels, B2A {b2a.out_ch}")

    def to_xyz(device: np.ndarray) -> np.ndarray:
        if a2b is None:
            raise ICCParseError("profile has no A2B tag")
        return a2b.decode_pcs(a2b(np.asarray(device, dtype=np.float64)))

    def from_xyz(xyz: np.ndarray) -> np.ndarray:
        if b2a is None:
            raise ICCParseError("profile has no B2A tag")
        return b2a(b2a.encode_pcs(np.asarray(xyz, dtype=np.float64)))

    n_channels = a2b.in_ch if a2b is not None else b2a.out_ch
    return BuiltTransform(to_xyz, from_xyz, kind="lut", n_channels=n_channels)
