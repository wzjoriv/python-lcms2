"""
Colour transforms between profiles.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from . import _lcms2
from ._constants import ColorSpaceSignature, DeviceClass, Flag, Intent
from ._formats import as_array, dtype_spec, format_info, format_name, is_format_name, parse_format
from ._parallel import parallel_worthwhile, run_chunks
from ._profile import Profile

__all__ = ["Transform", "get_alarm_codes", "set_adaptation_state", "set_alarm_codes"]

_PRECISION = {"auto": Flag.NONE, "high": Flag.HIGHRESPRECALC, "low": Flag.LOWRESPRECALC}
_FLOAT_SPACES = (ColorSpaceSignature.LAB, ColorSpaceSignature.XYZ)
_COLORSPACE_BITS = 0x1F << 16


def _run(handle: Any, arr: np.ndarray, out: np.ndarray, fin: Any, fout: Any) -> None:
    """Transform ``arr`` into ``out``, in parallel pixel chunks when the image is large."""
    npix = arr.size // fin.channels
    if fin.planar or fout.planar or not parallel_worthwhile(npix):
        _lcms2.transform_apply(handle, arr, out)
        return
    src = arr.reshape(npix, fin.channels)
    dst = out.reshape(npix, fout.channels)
    run_chunks(npix, lambda a, b: _lcms2.transform_apply(handle, src[a:b], dst[a:b]))


def _device_domain(profile: Profile) -> tuple[np.ndarray, np.ndarray] | None:
    """``(lo, span)`` from a generated profile's ``crsD`` tag; such profiles take plain [0, 1] device values."""
    if not profile.has_tag("crsD"):
        return None
    import struct

    raw = profile.read_tag_raw("crsD")
    n = struct.unpack_from(">I", raw, 0)[0]
    lo, hi = np.array([struct.unpack_from(">ff", raw, 4 + 8 * i) for i in range(n)], dtype=np.float64).T
    return lo, hi - lo


def _scale_device(v: np.ndarray, lo: np.ndarray, span: np.ndarray, inverse: bool) -> np.ndarray:
    """In place: [0, 1] -> device range, or back with ``inverse``; gray arrays may lack a channel axis."""
    view = v[..., : lo.size] if lo.size > 1 else v
    if inverse:
        view -= lo
        view /= span
    else:
        view *= span
        view += lo
    return v


def _private(profile: Profile) -> Profile:
    """A copy for Little CMS to read when building a transform would rewrite the profile's v2 Lab tables."""
    if profile.version >= 4 or str(profile.pcs) != "Lab":
        return profile
    return Profile._from_handle(_lcms2.profile_open(_lcms2.profile_to_bytes(profile._handle)))


def as_profile(obj: Any) -> Profile:
    """``obj`` itself if it is a Profile, else ``Profile(obj)`` (name, path, bytes, colour space)."""
    return obj if isinstance(obj, Profile) else Profile(obj)


def _output_side_is_pcs(p: Profile) -> bool:
    return p.device_class in (DeviceClass.LINK, DeviceClass.ABSTRACT)


class Transform:
    """A colour transform from one profile to another (or through a chain of profiles).

    Pixel formats are inferred from each input array (dtype, channel count, alpha) and one
    LittleCMS transform is cached per format pair. The explicit form with format strings,
    ``Transform(src, "RGB_8", dst, "RGB_16")``, fixes the formats instead.

    Parameters
    ----------
    src : Profile, str, path or list of those
        Source profile; a list makes a multi-profile transform (``dst`` omitted).
    dst : Profile, str or path
        Destination profile.
    intent : Intent, str or int, or one per profile
        Rendering intent: ``"relative"`` (default; in-gamut colors unchanged), ``"perceptual"``,
        ``"saturation"`` or ``"absolute"``.
    bpc : bool or one per profile
        Black point compensation.
    proof : Profile, optional
        Proofing profile: simulate ``proof`` on ``dst`` (soft proofing).
    proof_intent : intent
        Intent from the source to the proofing device (0-3).
    gamut_check : bool
        Mark colours outside the ``proof`` gamut with the alarm colour (see :func:`set_alarm_codes`).
    optimize : bool
        Allow LittleCMS to precalculate (False adds ``NOOPTIMIZE``).
    precision : {"auto", "high", "low"}
        Precalculation table size.
    flags : Flag, int or str
        Any other ``cmsFLAGS_*``.
    adaptation : sequence of float, optional
        Per-profile adaptation states (makes an extended transform).

    Raises
    ------
    CMSError
        When LittleCMS cannot build the transform (e.g. incompatible profiles).

    Examples
    --------
    >>> import numpy as np, lcms2
    >>> t = lcms2.Transform("sRGB", "Lab", intent="relative")
    >>> t(np.array([255, 0, 0], np.uint8)).round(2)
    array([54.29, 80.81, 69.89])
    >>> t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("Lab"), "Lab_DBL")
    >>> t.apply([255, 255, 255]).round(3) + 0.0
    array([100.,   0.,   0.])
    """

    def __init__(
        self,
        src: Any,
        *args: Any,
        intent: Any = Intent.RELATIVE_COLORIMETRIC,
        flags: Any = None,
        proof: Any = None,
        proof_intent: Any = Intent.RELATIVE_COLORIMETRIC,
        gamut_check: bool = False,
        bpc: bool | Sequence[bool] = False,
        optimize: bool = True,
        precision: str = "auto",
        adaptation: Sequence[float] | None = None,
    ) -> None:
        self._formats: tuple[int, int] | None = None
        if len(args) >= 3 and is_format_name(args[0]):
            if len(args) > 5:
                raise TypeError("Transform(src, src_format, dst, dst_format[, intent[, flags]])")
            dst = args[1]
            self._formats = (parse_format(args[0]), parse_format(args[2]))
            if len(args) > 3:
                intent = args[3]
            if len(args) > 4:
                flags = args[4]
            profiles = [as_profile(src), as_profile(dst)]
        elif len(args) == 1:
            profiles = [as_profile(src), as_profile(args[0])]
        elif not args and not isinstance(src, (list, tuple)):
            link = as_profile(src)
            if link.device_class != DeviceClass.LINK:
                raise TypeError("Transform(profile) needs a device link; use Transform(src, dst)")
            profiles = [link]
        elif not args:
            profiles = [as_profile(p) for p in src]
            if len(profiles) < 1:
                raise ValueError("a multi-profile transform needs at least one profile")
        else:
            raise TypeError(
                "use Transform(src, dst, ...), Transform([p1, p2, ...], ...) or "
                "Transform(src, 'RGB_8', dst, 'RGB_16', intent, flags)"
            )
        self.profiles: tuple[Profile, ...] = tuple(profiles)
        n = len(profiles)

        per_profile = (
            not isinstance(intent, (str, int)) or not isinstance(bpc, bool) or adaptation is not None
        )
        if per_profile:
            self._intents = _per(intent, n, "intent", Intent.parse)
            self._bpc = _per(bpc, n, "bpc", bool)
            self._adaptation = (
                _per(adaptation, n, "adaptation", float) if adaptation is not None else [1.0] * n
            )
            self.intent = self._intents[-1]
        else:
            self.intent = Intent.parse(intent)

        f = Flag.parse(flags) | Flag.NOCACHE
        if per_profile is False and bpc:
            f |= Flag.BLACKPOINTCOMPENSATION
        if not optimize:
            f |= Flag.NOOPTIMIZE
        if precision not in _PRECISION:
            raise ValueError(f"precision must be 'auto', 'high' or 'low', got {precision!r}")
        f |= _PRECISION[precision]
        self.proof = None if proof is None else as_profile(proof)
        self.proof_intent = Intent.parse(proof_intent)
        if self.proof_intent > 3:
            raise ValueError(
                "proof_intent must be an ICC intent (perceptual, relative, saturation, absolute)"
            )
        if gamut_check:
            if self.proof is None:
                raise ValueError("gamut_check=True needs a proof= profile to check against")
            f |= Flag.GAMUTCHECK
        if self.proof is not None:
            if n != 2:
                raise ValueError("proof= works with exactly two profiles (src, dst)")
            f |= Flag.SOFTPROOFING
        self._per_profile = per_profile
        self.flags = f
        self._cache: dict[tuple[int, int], Any] = {}
        self._formats_for: dict[tuple[str, int | None, str | None], tuple[int, int, np.dtype]] = {}
        self._lock = threading.Lock()
        self._in_domain = _device_domain(self.profiles[0])
        last = self.profiles[-1]
        self._out_domain = None if _output_side_is_pcs(last) else _device_domain(last)
        self._proof_chain: tuple[Transform, Transform] | None = None
        if self.proof is not None and not gamut_check and not per_profile and self.proof.is_matrix_shaper:
            self._proof_chain = (
                Transform(
                    profiles[0], self.proof, intent=self.intent, flags=flags, bpc=bpc, precision=precision
                ),
                Transform(
                    self.proof, profiles[1], intent=self.proof_intent, flags=flags, precision=precision
                ),
            )
        self._exact_out: tuple[Transform, Any] | None = None
        out_profile = profiles[-1]
        if (
            self.proof is None
            and not per_profile
            and len(profiles) >= 2
            and not _output_side_is_pcs(out_profile)
        ):
            space = getattr(out_profile, "_source_space", None)
            if space is None and out_profile.has_tag("crsC"):
                from .icc import ICCSpace

                space = ICCSpace(out_profile.to_bytes())
            if space is not None:
                to_xyz = Transform([*profiles[:-1], Profile.xyz()], intent=self.intent, flags=flags, bpc=bpc)
                self._exact_out = (to_xyz, space)
        if self._formats is not None:
            self._get(*self._formats)

    # ---- Creation ----
    def _create(self, in_fmt: int, out_fmt: int) -> Any:
        flags = int(self.flags)
        if format_info(in_fmt).extra:
            flags |= Flag.COPY_ALPHA
        p = [_private(x) for x in self.profiles]
        proof = None if self.proof is None else _private(self.proof)
        if self._per_profile:
            gamut = proof if self.flags & Flag.GAMUTCHECK else None
            return _lcms2.transform_create_extended(
                list(p),
                self._bpc,
                [int(i) for i in self._intents],
                self._adaptation,
                gamut,
                len(p) - 1 if gamut else 0,
                in_fmt,
                out_fmt,
                flags,
            )
        if self.proof is not None:
            return _lcms2.transform_create_proofing(
                p[0], p[1], proof, in_fmt, out_fmt, int(self.intent), int(self.proof_intent), flags
            )
        if len(p) == 2:
            return _lcms2.transform_create(p[0], p[1], in_fmt, out_fmt, int(self.intent), flags)
        return _lcms2.transform_create_multi(list(p), in_fmt, out_fmt, int(self.intent), flags)

    def _get(self, in_fmt: int, out_fmt: int) -> Any:
        key = (in_fmt, out_fmt)
        h = self._cache.get(key)
        if h is None:
            with self._lock:
                h = self._cache.get(key)
                if h is None:
                    h = self._cache[key] = self._create(in_fmt, out_fmt)
        return h

    # ---- Format resolution ----
    @property
    def input_profile(self) -> Profile:
        """The first profile."""
        return self.profiles[0]

    @property
    def output_profile(self) -> Profile:
        """The last profile."""
        return self.profiles[-1]

    @property
    def input_format(self) -> str | None:
        """Explicit input format name, or None when formats are inferred."""
        return None if self._formats is None else format_name(self._formats[0])

    @property
    def output_format(self) -> str | None:
        """Explicit output format name, or None when formats are inferred."""
        return None if self._formats is None else format_name(self._formats[1])

    def _out_space(self) -> Any:
        last = self.profiles[-1]
        return last.pcs if _output_side_is_pcs(last) else last.color_space

    def _infer(self, arr: np.ndarray, out_dtype: DTypeLike) -> tuple[int, int, np.dtype]:
        key = (
            arr.dtype.str,
            arr.shape[-1] if arr.ndim and arr.shape[-1] <= 16 else None,
            None if out_dtype is None else np.dtype(out_dtype).str,
        )
        hit = self._formats_for.get(key)
        if hit is None:
            hit = self._formats_for[key] = self._resolve(arr, out_dtype)
        return hit

    def _resolve(self, arr: np.ndarray, out_dtype: DTypeLike) -> tuple[int, int, np.dtype]:
        nbytes, is_float = dtype_spec(arr.dtype)
        first, last = self.profiles[0], self.profiles[-1]
        base = format_info(_lcms2.format_for_profile(first, nbytes, is_float, 0)).channels
        extra = 0
        if base != 1:
            got = arr.shape[-1] if arr.ndim else None
            if got == base + 1:
                extra = 1
            elif got != base:
                raise ValueError(
                    f"{first.description!r} is {first.color_space} with {base} channels: the last axis must be "
                    f"{base} (or {base + 1} with alpha), got shape {arr.shape}"
                )
        if out_dtype is None:
            out_dtype = arr.dtype
            if arr.dtype.kind == "u" and self._out_space() in _FLOAT_SPACES:
                out_dtype = np.dtype(np.float64)
        out_dtype = np.dtype(out_dtype)
        obytes, ofloat = dtype_spec(out_dtype)
        in_fmt = _lcms2.format_for_profile(first, nbytes, is_float, extra)
        out_fmt = _lcms2.format_for_profile(last, obytes, ofloat, extra, _output_side_is_pcs(last))
        if self._in_domain is not None:
            in_fmt &= ~_COLORSPACE_BITS
        if self._out_domain is not None:
            out_fmt &= ~_COLORSPACE_BITS
        return in_fmt, out_fmt, out_dtype

    # ---- Apply ----
    def apply(self, data: Any, out: np.ndarray | None = None, *, out_dtype: DTypeLike = None) -> np.ndarray:
        """Transform pixel data; the last axis holds the channels.

        Parameters
        ----------
        data : array-like
            ``(..., channels)`` pixels (``(H, W)`` works for gray). Lists of ints become uint8/uint16,
            floats float64. Non-contiguous input is copied.
        out : numpy.ndarray, optional
            Preallocated C-contiguous output of the right shape and dtype.
        out_dtype : dtype, optional
            Output dtype (inferred transforms only). Default: the input dtype, except integer input into
            Lab/XYZ which gives float64.

        Returns
        -------
        numpy.ndarray
            The transformed pixels (``out`` when given).

        Examples
        --------
        >>> import lcms2
        >>> t = lcms2.Transform("sRGB", "sRGB")
        >>> t.apply([[0, 128, 255]])
        array([[  0, 128, 255]], dtype=uint8)
        """
        if self._formats is not None:
            if out_dtype is not None:
                raise ValueError(
                    "out_dtype cannot be used with explicit formats; pick the output format instead"
                )
            in_fmt, out_fmt = self._formats
            in_info = format_info(in_fmt)
            arr = as_array(data, None if isinstance(data, np.ndarray) else in_info.dtype)
            if arr.dtype != in_info.dtype:
                raise TypeError(
                    f"input dtype {arr.dtype} does not match format {self.input_format} ({in_info.dtype}); "
                    f"use .astype(np.{in_info.dtype})"
                )
            odt = format_info(out_fmt).dtype
        else:
            arr = as_array(data)
            in_fmt, out_fmt, odt = self._infer(arr, out_dtype)
            if self._proof_chain is not None:
                to_proof, from_proof = self._proof_chain
                mid = to_proof.apply(arr, out_dtype=np.float64)
                np.clip(mid, 0.0, 1.0, out=mid)
                return from_proof.apply(mid, out=out, out_dtype=odt)
            in_info = format_info(in_fmt)
            if self._exact_out is not None and in_info.extra == 0:
                to_xyz, space = self._exact_out
                result = space.from_xyz(to_xyz.apply(arr, out_dtype=np.float64))
                if odt.kind == "f":
                    result = result.astype(odt, copy=False)
                else:
                    lo, hi = np.array(space.device_domain, dtype=np.float64).T
                    scaled = np.clip((result - lo) / (hi - lo), 0.0, 1.0) * np.iinfo(odt).max
                    result = np.rint(scaled).astype(odt)
                result = result.reshape(_out_shape(arr, in_info, format_info(out_fmt)))
                if out is None:
                    return result
                out[...] = result
                return out
            if self._in_domain is not None and arr.dtype.kind == "f":
                arr = _scale_device(arr.astype(np.float64, copy=True), *self._in_domain, inverse=True)
                arr = arr.astype(in_info.dtype, copy=False)
        shape = _out_shape(arr, in_info, format_info(out_fmt))
        if out is None:
            out = np.empty(shape, odt)
        else:
            if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != odt:
                got = (out.shape, out.dtype) if isinstance(out, np.ndarray) else type(out).__name__
                raise ValueError(f"out must be an array of shape {shape} and dtype {odt}, got {got}")
            if not out.flags.c_contiguous or not out.flags.writeable:
                raise ValueError("out must be C-contiguous and writable")
        if arr.size:
            _run(self._get(in_fmt, out_fmt), arr, out, in_info, format_info(out_fmt))
            if self._formats is None and self._out_domain is not None and out.dtype.kind == "f":
                _scale_device(out, *self._out_domain, inverse=False)
        return out

    __call__ = apply

    def to_device_link(self, version: float = 4.3, flags: Any = None) -> Profile:
        """A device link profile equivalent to this transform.

        Examples
        --------
        >>> import lcms2
        >>> link = lcms2.Transform("sRGB", "Lab").to_device_link()
        >>> str(link.device_class)
        'link'
        """
        if self._formats is not None:
            h = self._get(*self._formats)
        else:
            first, last = self.profiles[0], self.profiles[-1]
            h = self._get(
                _lcms2.format_for_profile(first, 2, False),
                _lcms2.format_for_profile(last, 2, False, 0, _output_side_is_pcs(last)),
            )
        return Profile._from_handle(
            _lcms2.transform_to_device_link(h, float(version), int(Flag.parse(flags)))
        )

    def __repr__(self) -> str:
        chain = " -> ".join(repr(p.description) for p in self.profiles)
        fmt = f" {self.input_format} -> {self.output_format}" if self._formats else ""
        extra = f" proof={self.proof.description!r}" if self.proof is not None else ""
        flags = self.flags & ~Flag.NOCACHE
        shown = f" flags={flags!r}" if flags else ""
        return f"<Transform {chain}{fmt} {self.intent.name.lower()}{extra}{shown}>"


def _per(value: Any, n: int, what: str, conv: Any) -> list[Any]:
    if isinstance(value, (str, int, bool)) or value is None:
        return [conv(value)] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"{what} needs one value per profile ({n}), got {len(values)}")
    return [conv(v) for v in values]


def _out_shape(arr: np.ndarray, fin: Any, fout: Any) -> tuple[int, ...]:
    if fin.planar or fout.planar:
        if fin.planar and arr.ndim and arr.shape[0] == fin.channels:
            lead = arr.shape[1:]
        elif not fin.planar and arr.ndim and arr.shape[-1] == fin.channels:
            lead = arr.shape[:-1]
        else:
            raise ValueError(
                f"expected {fin.channels} channels ({'first' if fin.planar else 'last'} axis), "
                f"got shape {arr.shape}"
            )
        return (fout.channels, *lead) if fout.planar else (*lead, fout.channels)
    keep = False
    if fin.channels == 1:
        keep = arr.ndim > 0 and arr.shape[-1] == 1
        lead = arr.shape[:-1] if keep else arr.shape
    else:
        if arr.ndim == 0 or arr.shape[-1] != fin.channels:
            raise ValueError(f"the last axis must hold {fin.channels} channels, got shape {arr.shape}")
        lead = arr.shape[:-1]
    return (*lead, fout.channels) if fout.channels > 1 or keep else tuple(lead)


def supported_intents() -> dict[Intent | int, str]:
    """Every rendering intent the LittleCMS engine implements, with its description.

    Includes intents added by plugins; :attr:`Profile.supported_intents` lists the ones a
    given profile supports.

    Examples
    --------
    >>> import lcms2
    >>> lcms2.supported_intents()[lcms2.Intent.PERCEPTUAL]
    'Perceptual'
    """
    out: dict[Intent | int, str] = {}
    for code, desc in _lcms2.get_supported_intents():
        try:
            out[Intent(code)] = desc
        except ValueError:
            out[code] = desc
    return out


def get_alarm_codes() -> tuple[int, ...]:
    """The 16 gamut-check alarm values (process-wide), in 16-bit output encoding."""
    return _lcms2.get_alarm_codes()


def set_alarm_codes(codes: Sequence[int]) -> None:
    """Set the gamut-check alarm values (process-wide; affects transforms created afterwards)."""
    _lcms2.set_alarm_codes(list(codes))


def set_adaptation_state(state: float) -> float:
    """Set the process-wide absolute-colorimetric adaptation state (0..1); returns the previous one."""
    return _lcms2.set_adaptation_state(float(state))
