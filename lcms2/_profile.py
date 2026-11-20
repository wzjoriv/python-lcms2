"""
ICC profiles.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from . import _lcms2
from ._builtins import builtin_names, create_builtin, lookup_builtin, resolve_white_point
from ._constants import ColorSpaceSignature, DeviceClass, Intent
from ._profile_data import ProfileData

__all__ = ["Profile"]

_INFO = {"description": 0, "manufacturer": 1, "model": 2, "copyright": 3}


def _info_property(name: str, doc: str) -> property:
    which = _INFO[name]

    def fget(self: Profile) -> str:
        return _lcms2.profile_get_info(self._handle, which)

    def fset(self: Profile, text: str) -> None:
        _lcms2.profile_set_info(self._handle, which, str(text))

    return property(fget, fset, doc=doc)


class Profile(ProfileData):
    """An ICC profile.

    Parameters
    ----------
    source : str, os.PathLike, bytes, Profile, colour space or None
        A built-in name (case-insensitive, lenient: ``"adobe rgb"``, ``"display p3"``, ``"prophoto"``;
        see :meth:`builtins`), a path to an ``.icc``/``.icm`` file, raw profile bytes, an object with
        ``to_icc_profile()`` (a ``ColorSpace``) or None for the NULL profile.
    white_point : str or sequence of float, optional
        For ``"Lab"`` only: illuminant name (``"D65"``) or xyY of the Lab white (D50 by default).
    degrees : {2, 10}
        Observer for illuminant names.
    temperature : float, optional
        For ``"Lab"`` only: white point as a daylight colour temperature in kelvin.

    Raises
    ------
    FileNotFoundError
        For a string that is neither a built-in name nor an existing file.
    CMSError
        When the data is not a valid ICC profile.

    Examples
    --------
    >>> import lcms2
    >>> p = lcms2.Profile("sRGB")
    >>> p.color_space, p.device_class
    (<ColorSpaceSignature.RGB: 'RGB'>, <DeviceClass.DISPLAY: 'display'>)
    >>> lcms2.Profile("display p3").description
    'Display P3'
    """

    __slots__ = ("__weakref__", "_handle")

    def __init__(
        self,
        source: Any = None,
        *,
        white_point: Any = None,
        degrees: int = 2,
        temperature: float | None = None,
    ) -> None:
        self._handle = _open_source(source, white_point, degrees, temperature)

    # Construction -------------------------------------------------------------------------------

    @classmethod
    def _from_handle(cls, handle: Any, description: str | None = None) -> Profile:
        self = cls.__new__(cls)
        self._handle = handle
        if description is not None:
            self.description = description
        return self

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a profile file (any path, including non-ASCII ones on Windows)."""
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Profile:
        """Parse a profile from memory.

        Examples
        --------
        >>> p = Profile.from_bytes(Profile("sRGB").to_bytes())
        >>> p.description
        'sRGB built-in'
        """
        return cls._from_handle(_lcms2.profile_open(data))

    @classmethod
    def from_space(cls, space: Any, strict: bool = False) -> Profile:
        """An exact profile of a colour space object (anything with ``to_icc_profile()``).

        Parameters
        ----------
        space : ColorSpace
            The space; its NumPy ``to_xyz``/``from_xyz`` are traced into an exact profile.
        strict : bool, optional
            Raise :class:`~lcms2.ICCWriteError` instead of falling back to sampled lookup
            tables when the functions cannot be traced (data-dependent branching, ``float()``,
            random numbers, non-NumPy libraries). Default False.

        Raises
        ------
        ICCWriteError
            With ``strict=True``, naming the operation that prevents an exact profile.
        """
        make = getattr(space, "to_icc_profile", None)
        if make is None:
            raise TypeError(f"{type(space).__name__} has no to_icc_profile(); pass a ColorSpace")
        data = make(strict=True) if strict else make()
        if isinstance(data, Profile):
            return data
        if isinstance(data, dict):  # ColorSpace.to_icc_profile() -> {"bytes", "method", "notes"}
            data = data["bytes"]
        return cls.from_bytes(data)

    @classmethod
    def builtin(cls, name: str) -> Profile:
        """A built-in profile by (lenient) name; see :meth:`builtins`."""
        canonical = lookup_builtin(name)
        if canonical is None:
            raise ValueError(f"unknown built-in profile {name!r}; available: {', '.join(builtin_names())}")
        desc = None if canonical in ("sRGB", "Lab", "Lab v2", "XYZ", "NULL") else canonical
        return cls._from_handle(create_builtin(canonical), desc)

    @staticmethod
    def builtins() -> list[str]:
        """Names of the built-in profiles.

        Examples
        --------
        >>> "Display P3" in Profile.builtins()
        True
        """
        return builtin_names()

    @classmethod
    def srgb(cls) -> Profile:
        """LittleCMS's built-in sRGB profile."""
        return cls._from_handle(_lcms2.profile_create_srgb())

    @classmethod
    def lab(
        cls, white_point: Any = None, *, version: int = 4, degrees: int = 2, temperature: float | None = None
    ) -> Profile:
        """Lab identity profile (v4 or v2) relative to a white point (D50 by default).

        Examples
        --------
        >>> Profile.lab("D65").color_space
        <ColorSpaceSignature.LAB: 'Lab'>
        """
        wp = resolve_white_point(white_point, degrees, temperature)
        return cls._from_handle(_lcms2.profile_create_lab(version, wp))

    @classmethod
    def xyz(cls) -> Profile:
        """XYZ identity profile (D50 PCS, Y = 1 for the white)."""
        return cls._from_handle(_lcms2.profile_create_xyz())

    @classmethod
    def rgb(
        cls,
        white_point: Any = "D65",
        primaries: Sequence[Sequence[float]] = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
        gamma: Any = 2.2,
        *,
        degrees: int = 2,
        description: str | None = None,
    ) -> Profile:
        """Matrix-shaper RGB profile.

        Parameters
        ----------
        white_point : str or sequence of float
            Illuminant name or xyY.
        primaries : three (x, y) or (x, y, Y) sequences
            Red, green and blue chromaticities.
        gamma : float, ToneCurve, (type, params) or a sequence of 3 of those
            Transfer function(s) from encoded to linear values.
        description : str, optional
            Profile description text.

        Examples
        --------
        >>> p = Profile.rgb("D65", gamma=2.2, description="my RGB")
        >>> p.description, p.is_matrix_shaper
        ('my RGB', True)
        """
        wp = resolve_white_point(white_point, degrees)
        prims = tuple(tuple(float(v) for v in p) for p in primaries)
        if len(prims) != 3:
            raise ValueError(f"primaries must be (red, green, blue), got {len(prims)} entries")
        is_parametric = (
            isinstance(gamma, tuple)
            and len(gamma) == 2
            and isinstance(gamma[0], int)
            and not isinstance(gamma[1], (int, float))
        )
        curves = list(gamma) if isinstance(gamma, (list, tuple)) and not is_parametric else [gamma]
        return cls._from_handle(_lcms2.profile_create_rgb(wp, prims, curves), description)

    @classmethod
    def gray(cls, white_point: Any = "D50", gamma: Any = 2.2, *, degrees: int = 2) -> Profile:
        """Gray profile from a white point and a transfer function (gamma, ToneCurve or (type, params))."""
        return cls._from_handle(_lcms2.profile_create_gray(resolve_white_point(white_point, degrees), gamma))

    @classmethod
    def abstract(
        cls,
        brightness: float = 0.0,
        contrast: float = 1.0,
        hue: float = 0.0,
        saturation: float = 0.0,
        *,
        temp_src: int = 0,
        temp_dst: int = 0,
        lut_points: int = 17,
    ) -> Profile:
        """Abstract Lab profile adjusting brightness, contrast, hue, saturation and white temperature."""
        return cls._from_handle(
            _lcms2.profile_create_bchsw(lut_points, brightness, contrast, hue, saturation, temp_src, temp_dst)
        )

    @classmethod
    def ink_limiting(cls, limit: float = 300.0, color_space: str = "CMYK") -> Profile:
        """CMYK device link limiting total ink coverage to ``limit`` percent."""
        return cls._from_handle(_lcms2.profile_create_ink_limiting(color_space, limit))

    @classmethod
    def linearization(cls, curves: Sequence[Any], color_space: str = "RGB") -> Profile:
        """Device link applying one curve (gamma, ToneCurve or (type, params)) per channel."""
        return cls._from_handle(_lcms2.profile_create_linearization(color_space, list(curves)))

    # Serialisation ---------------------------------------------------------------------------------

    @property
    def space(self):
        """A :class:`lcms2.spaces.ColorSpace` view of this profile (pure NumPy evaluation).

        Examples
        --------
        >>> import lcms2
        >>> lcms2.Profile("sRGB").space.n_channels
        3
        """
        from .icc import ICCSpace

        return ICCSpace(self.to_bytes())

    def to_bytes(self) -> bytes:
        """The profile as ICC bytes."""
        return _lcms2.profile_to_bytes(self._handle)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the profile to a file."""
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)

    def __reduce__(self) -> tuple[Any, ...]:
        return (Profile.from_bytes, (self.to_bytes(),))

    def __copy__(self) -> Profile:
        return Profile.from_bytes(self.to_bytes())

    # Text and header -------------------------------------------------------------------------------

    description = _info_property("description", "Profile description text (settable).")
    manufacturer = _info_property("manufacturer", "Device manufacturer text (settable).")
    model = _info_property("model", "Device model text (settable).")
    copyright = _info_property("copyright", "Copyright text (settable).")

    @property
    def header(self) -> dict[str, Any]:
        """Raw header fields (signatures as 4-character strings)."""
        return _lcms2.profile_get_header(self._handle)

    def set_header(self, **fields: Any) -> None:
        """Change header fields: device_class, color_space, pcs, version, rendering_intent, flags,
        manufacturer, model, attributes, profile_id."""
        _lcms2.profile_set_header(self._handle, **fields)

    @property
    def color_space(self) -> ColorSpaceSignature | str:
        """Data colour space (``"RGB"``, ``"CMYK"``, ``"Lab"`` ...)."""
        return ColorSpaceSignature.from_signature(self.header["color_space"])

    @property
    def pcs(self) -> ColorSpaceSignature | str:
        """Profile connection space (``"XYZ"`` or ``"Lab"``)."""
        return ColorSpaceSignature.from_signature(self.header["pcs"])

    @property
    def device_class(self) -> DeviceClass | str:
        """Profile class (``display``, ``output``, ``link`` ...)."""
        return DeviceClass.from_signature(self.header["device_class"])

    @device_class.setter
    def device_class(self, value: DeviceClass | str) -> None:
        sig = value.signature if isinstance(value, DeviceClass) else DeviceClass(value).signature
        self.set_header(device_class=sig)

    @property
    def version(self) -> float:
        """ICC version, e.g. 4.3 or 2.1 (settable)."""
        return round(self.header["version"], 2)

    @version.setter
    def version(self, value: float) -> None:
        self.set_header(version=float(value))

    @property
    def channels(self) -> int:
        """Number of channels of the data colour space."""
        return self.header["channels"]

    @property
    def rendering_intent(self) -> Intent | int:
        """Header rendering intent (settable)."""
        v = self.header["rendering_intent"]
        try:
            return Intent(v)
        except ValueError:
            return v

    @rendering_intent.setter
    def rendering_intent(self, value: Intent | int | str) -> None:
        self.set_header(rendering_intent=int(Intent.parse(value)))

    @property
    def creation_date(self) -> datetime.datetime | None:
        """Creation date from the header."""
        d = self.header["creation_date"]
        try:
            return datetime.datetime(*d) if d else None
        except ValueError:
            return None

    @property
    def profile_id(self) -> bytes:
        """The 16-byte MD5 profile ID (zeros when unset); see :meth:`compute_id`."""
        return self.header["profile_id"]

    def compute_id(self) -> bytes:
        """Compute the MD5 profile ID, store it in the header and return it."""
        return _lcms2.profile_compute_id(self._handle)

    # Conversion -------------------------------------------------------------------------------------

    def transform(
        self,
        data: Any,
        to: Any,
        *,
        intent: Any = "perceptual",
        bpc: bool = False,
        proof: Any = None,
        gamut_check: bool = False,
        optimize: bool = True,
        precision: str = "auto",
        flags: Any = None,
        out: np.ndarray | None = None,
        out_dtype: DTypeLike = None,
    ) -> np.ndarray:
        """Convert ``data`` from this profile to ``to``; same keywords as :func:`lcms2.convert`.

        Examples
        --------
        >>> Profile("sRGB").transform([255, 255, 255], "Lab").round(2) + 0.0
        array([100.,   0.,   0.])
        """
        from ._convert import convert

        return convert(
            data,
            self,
            to,
            intent=intent,
            bpc=bpc,
            proof=proof,
            gamut_check=gamut_check,
            optimize=optimize,
            precision=precision,
            flags=flags,
            out=out,
            out_dtype=out_dtype,
        )

    def __repr__(self) -> str:
        if self._handle is None:
            return "<Profile closed>"
        return f"<Profile {self.description!r} {self.color_space} {self.device_class} v{self.version:g}>"


def _open_source(source: Any, white_point: Any, degrees: int, temperature: float | None) -> Any:
    if (white_point is not None or temperature is not None) and not (
        isinstance(source, str) and lookup_builtin(source) in ("Lab", "Lab v2")
    ):
        raise ValueError(
            "white_point/temperature only apply to Profile('Lab'); use Profile.rgb/gray otherwise"
        )
    if source is None:
        return _lcms2.profile_create_null()
    if isinstance(source, Profile):
        return _lcms2.profile_open(source.to_bytes())
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _lcms2.profile_open(source)
    if isinstance(source, str):
        name = lookup_builtin(source)
        if name in ("Lab", "Lab v2"):
            wp = resolve_white_point(white_point, degrees, temperature)
            return _lcms2.profile_create_lab(4 if name == "Lab" else 2, wp)
        if name is not None:
            return Profile.builtin(name)._handle
        if os.path.isfile(source):
            return Profile.open(source)._handle
        raise FileNotFoundError(
            f"{source!r} is neither a built-in profile name nor an existing file; "
            f"built-ins: {', '.join(builtin_names())}"
        )
    if isinstance(source, os.PathLike):
        return Profile.open(source)._handle
    if hasattr(source, "to_icc_profile"):
        return Profile.from_space(source)._handle
    raise TypeError(
        f"cannot make a Profile from {type(source).__name__}; pass a built-in name, a path, bytes or a ColorSpace"
    )
