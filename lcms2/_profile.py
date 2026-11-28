"""
ICC profiles.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike

from . import _lcms2
from ._builtins import builtin_names, create_builtin, lookup_builtin, resolve_white_point
from ._constants import ColorSpaceSignature, DeviceClass, Intent
from ._profile_data import ProfileData

if TYPE_CHECKING:
    from .icc import ICCSpace

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
    degrees : {2, 10}, optional
        Observer for illuminant names. Defaults to 2.
    temperature : float, optional
        For ``"Lab"`` only: white point as a daylight colour temperature in kelvin.

    Raises
    ------
    FileNotFoundError
        For a string that is neither a built-in name nor an existing file.
    CMSError
        When the data is not a valid ICC profile.
    ValueError
        If ``white_point`` or ``temperature`` is given for a profile other than Lab.
    TypeError
        If ``source`` is of an unsupported type.

    See Also
    --------
    Profile.builtins : Names of the built-in profiles.
    Profile.open : Read a profile from a file.

    Examples
    --------
    >>> import lcms2
    >>> p = lcms2.Profile("sRGB")
    >>> p.color_space, p.device_class
    (<ColorSpaceSignature.RGB: 'RGB'>, <DeviceClass.DISPLAY: 'display'>)
    >>> lcms2.Profile("display p3").description
    'Display P3'
    """

    __slots__ = ("__weakref__", "_handle", "_source_space")

    def __init__(
        self,
        source: Any = None,
        *,
        white_point: Any = None,
        degrees: int = 2,
        temperature: float | None = None,
    ) -> None:
        self._handle = _open_source(source, white_point, degrees, temperature)
        self._source_space: Any = None

    # ---- Construction ----
    @classmethod
    def _from_handle(cls, handle: Any, description: str | None = None) -> Profile:
        self = cls.__new__(cls)
        self._handle = handle
        self._source_space = None
        if description is not None:
            self.description = description
        return self

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a profile file (any path, including non-ASCII ones on Windows).

        Parameters
        ----------
        path : str or os.PathLike
            Path of the ``.icc``/``.icm`` file.

        Returns
        -------
        Profile
            The loaded profile.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        CMSError
            If the file is not a valid ICC profile.

        See Also
        --------
        Profile.save : Write a profile to a file.

        Examples
        --------
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "p.icc")
        >>> Profile("sRGB").save(path)
        >>> Profile.open(path).color_space
        <ColorSpaceSignature.RGB: 'RGB'>
        """
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Profile:
        """Parse a profile from memory.

        Parameters
        ----------
        data : bytes, bytearray or memoryview
            The raw ICC profile.

        Returns
        -------
        Profile
            The parsed profile.

        Raises
        ------
        CMSError
            If the data is not a valid ICC profile.

        See Also
        --------
        Profile.to_bytes : The inverse.

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
            random numbers, non-NumPy libraries). Defaults to False.

        Returns
        -------
        Profile
            The profile of the colour space.

        Raises
        ------
        ICCWriteError
            With ``strict=True``, naming the operation that prevents an exact profile.
        TypeError
            If ``space`` has no ``to_icc_profile()``.

        See Also
        --------
        Profile.space : The opposite direction, a colour space view of a profile.

        Examples
        --------
        >>> import lcms2
        >>> Profile.from_space(lcms2.spaces.get("srgb")).color_space
        <ColorSpaceSignature.RGB: 'RGB'>
        """
        make = getattr(space, "to_icc_profile", None)
        if make is None:
            raise TypeError(f"{type(space).__name__} has no to_icc_profile(); pass a ColorSpace")
        data = make(strict=True) if strict else make()
        if isinstance(data, Profile):
            return data
        if isinstance(data, dict):  # ColorSpace.to_icc_profile() -> {"bytes", "method", "notes"}
            data = data["bytes"]
        profile = cls.from_bytes(data)
        profile._source_space = space
        return profile

    @classmethod
    def builtin(cls, name: str) -> Profile:
        """A built-in profile by (lenient) name; see :meth:`builtins`.

        Parameters
        ----------
        name : str
            Built-in profile name, case-insensitive (``"adobe rgb"``, ``"display p3"``).

        Returns
        -------
        Profile
            The built-in profile.

        Raises
        ------
        ValueError
            If the name is unknown, listing the available names.

        See Also
        --------
        Profile.builtins : List the available names.

        Examples
        --------
        >>> Profile.builtin("display p3").description
        'Display P3'
        """
        canonical = lookup_builtin(name)
        if canonical is None:
            raise ValueError(f"unknown built-in profile {name!r}; available: {', '.join(builtin_names())}")
        desc = None if canonical in ("sRGB", "Lab", "Lab v2", "XYZ", "NULL") else canonical
        return cls._from_handle(create_builtin(canonical), desc)

    @staticmethod
    def builtins() -> list[str]:
        """Names of the built-in profiles.

        Returns
        -------
        list of str
            The canonical built-in names.

        See Also
        --------
        Profile.builtin : Create one by name.

        Examples
        --------
        >>> "Display P3" in Profile.builtins()
        True
        """
        return builtin_names()

    @classmethod
    def srgb(cls) -> Profile:
        """LittleCMS's built-in sRGB profile.

        Returns
        -------
        Profile
            An sRGB display profile.

        See Also
        --------
        Profile.rgb : A custom matrix-shaper RGB profile.

        Examples
        --------
        >>> Profile.srgb().color_space
        <ColorSpaceSignature.RGB: 'RGB'>
        """
        return cls._from_handle(_lcms2.profile_create_srgb())

    @classmethod
    def lab(
        cls, white_point: Any = None, *, version: int = 4, degrees: int = 2, temperature: float | None = None
    ) -> Profile:
        """Lab identity profile (v4 or v2) relative to a white point (D50 by default).

        Parameters
        ----------
        white_point : str or sequence of float, optional
            Illuminant name or xyY of the Lab white. Defaults to D50.
        version : int, optional
            ICC major version, 4 or 2. Defaults to 4.
        degrees : {2, 10}, optional
            Observer for illuminant names. Defaults to 2.
        temperature : float, optional
            White point as a daylight colour temperature in kelvin.

        Returns
        -------
        Profile
            The Lab profile.

        See Also
        --------
        Profile.xyz : The XYZ identity profile.

        Examples
        --------
        >>> Profile.lab("D65").color_space
        <ColorSpaceSignature.LAB: 'Lab'>
        """
        wp = resolve_white_point(white_point, degrees, temperature)
        return cls._from_handle(_lcms2.profile_create_lab(version, wp))

    @classmethod
    def xyz(cls) -> Profile:
        """XYZ identity profile (D50 PCS, Y = 1 for the white).

        Returns
        -------
        Profile
            The XYZ profile.

        See Also
        --------
        Profile.lab : The Lab identity profile.

        Examples
        --------
        >>> Profile.xyz().color_space
        <ColorSpaceSignature.XYZ: 'XYZ'>
        """
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
        degrees : {2, 10}, optional
            Observer for illuminant names. Defaults to 2.
        description : str, optional
            Profile description text. Defaults to one naming the white point and gamma,
            such as ``"RGB (D65, gamma 2.2)"``.

        Returns
        -------
        Profile
            The RGB profile.

        Raises
        ------
        ValueError
            If ``primaries`` does not have three entries.

        See Also
        --------
        Profile.srgb : The built-in sRGB profile.
        Profile.gray : A gray profile.

        Examples
        --------
        >>> Profile.rgb("D65", gamma=2.2).description
        'RGB (D65, gamma 2.2)'
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
        if description is None:
            wp_name = white_point if isinstance(white_point, str) else "custom white"
            g = f"gamma {gamma:g}" if isinstance(gamma, (int, float)) else "custom curves"
            description = f"RGB ({wp_name}, {g})"
        return cls._from_handle(_lcms2.profile_create_rgb(wp, prims, curves), description)

    @classmethod
    def gray(cls, white_point: Any = "D50", gamma: Any = 2.2, *, degrees: int = 2) -> Profile:
        """Gray profile from a white point and a transfer function (gamma, ToneCurve or (type, params)).

        Parameters
        ----------
        white_point : str or sequence of float, optional
            Illuminant name or xyY. Defaults to ``"D50"``.
        gamma : float, ToneCurve or (type, params), optional
            Transfer function from encoded to linear values. Defaults to 2.2.
        degrees : {2, 10}, optional
            Observer for illuminant names. Defaults to 2.

        Returns
        -------
        Profile
            The gray profile.

        See Also
        --------
        Profile.rgb : A matrix-shaper RGB profile.

        Examples
        --------
        >>> Profile.gray(gamma=2.2).channels
        1
        """
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
        """Abstract Lab profile adjusting brightness, contrast, hue, saturation and white temperature.

        Parameters
        ----------
        brightness : float, optional
            Brightness offset. Defaults to 0.0.
        contrast : float, optional
            Contrast multiplier. Defaults to 1.0.
        hue : float, optional
            Hue rotation in degrees. Defaults to 0.0.
        saturation : float, optional
            Saturation offset. Defaults to 0.0.
        temp_src : int, optional
            Source white temperature in kelvin; 0 disables the adjustment. Defaults to 0.
        temp_dst : int, optional
            Destination white temperature in kelvin; 0 disables the adjustment. Defaults to 0.
        lut_points : int, optional
            Lookup table grid points per dimension. Defaults to 17.

        Returns
        -------
        Profile
            The abstract profile.

        See Also
        --------
        Profile.linearization : A per-channel curve device link.

        Examples
        --------
        >>> Profile.abstract(contrast=1.2).device_class
        <DeviceClass.ABSTRACT: 'abstract'>
        """
        return cls._from_handle(
            _lcms2.profile_create_bchsw(lut_points, brightness, contrast, hue, saturation, temp_src, temp_dst)
        )

    @classmethod
    def ink_limiting(cls, limit: float = 300.0, color_space: str = "CMYK") -> Profile:
        """CMYK device link limiting total ink coverage to ``limit`` percent.

        Parameters
        ----------
        limit : float, optional
            Maximum total ink coverage in percent. Defaults to 300.0.
        color_space : str, optional
            ``"CMYK"`` or ``"CMY"``. Defaults to ``"CMYK"``.

        Returns
        -------
        Profile
            The device link profile.

        Raises
        ------
        CMSError
            If the colour space is not supported.

        See Also
        --------
        Profile.linearization : A per-channel curve device link.

        Examples
        --------
        >>> Profile.ink_limiting(280).device_class
        <DeviceClass.LINK: 'link'>
        """
        return cls._from_handle(_lcms2.profile_create_ink_limiting(color_space, limit))

    @classmethod
    def linearization(cls, curves: Sequence[Any], color_space: str = "RGB") -> Profile:
        """Device link applying one curve (gamma, ToneCurve or (type, params)) per channel.

        Parameters
        ----------
        curves : sequence of float, ToneCurve or (type, params)
            One transfer function per channel.
        color_space : str, optional
            Colour space of the link. Defaults to ``"RGB"``.

        Returns
        -------
        Profile
            The device link profile.

        Raises
        ------
        CMSError
            If the colour space or curves are not supported.

        See Also
        --------
        Profile.ink_limiting : A CMYK ink limiting device link.

        Examples
        --------
        >>> Profile.linearization([2.2, 2.2, 2.2]).device_class
        <DeviceClass.LINK: 'link'>
        """
        return cls._from_handle(_lcms2.profile_create_linearization(color_space, list(curves)))

    # ---- Serialisation ----
    @property
    def space(self) -> ICCSpace:
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
        """The profile as ICC bytes.

        Returns
        -------
        bytes
            The serialized profile.

        See Also
        --------
        Profile.from_bytes : The inverse.

        Examples
        --------
        >>> Profile("sRGB").to_bytes()[36:40]
        b'acsp'
        """
        return _lcms2.profile_to_bytes(self._handle)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the profile to a file.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.

        See Also
        --------
        Profile.open : Read a profile file.

        Examples
        --------
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "srgb.icc")
        >>> Profile("sRGB").save(path)
        >>> os.path.getsize(path) > 0
        True
        """
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)

    def __reduce__(self) -> tuple[Any, ...]:
        return (Profile.from_bytes, (self.to_bytes(),))

    def __copy__(self) -> Profile:
        return Profile.from_bytes(self.to_bytes())

    # ---- Text and header ----
    description = _info_property("description", "Profile description text (settable).")
    manufacturer = _info_property("manufacturer", "Device manufacturer text (settable).")
    model = _info_property("model", "Device model text (settable).")
    copyright = _info_property("copyright", "Copyright text (settable).")

    @property
    def header(self) -> dict[str, Any]:
        """Raw header fields (signatures as 4-character strings)."""
        return _lcms2.profile_get_header(self._handle)

    def set_header(self, **fields: Any) -> None:
        """Change header fields in place.

        Parameters
        ----------
        **fields
            Any of ``device_class``, ``color_space``, ``pcs``, ``version``, ``rendering_intent``,
            ``flags``, ``manufacturer``, ``model``, ``attributes`` and ``profile_id``.

        Raises
        ------
        CMSError
            For an unknown field or invalid value.

        See Also
        --------
        Profile.header : The current header fields.

        Examples
        --------
        >>> p = Profile("sRGB")
        >>> p.set_header(version=2.1)
        >>> p.version
        2.1
        """
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
        """Compute the MD5 profile ID, store it in the header and return it.

        Returns
        -------
        bytes
            The 16-byte profile ID.

        See Also
        --------
        Profile.profile_id : The stored ID.

        Examples
        --------
        >>> len(Profile("sRGB").compute_id())
        16
        """
        return _lcms2.profile_compute_id(self._handle)

    # ---- Conversion ----
    def transform(
        self,
        data: Any,
        to: Any,
        *,
        intent: Any = "relative",
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

        Parameters
        ----------
        data : array_like
            Values in this profile's colour space, shaped ``(..., channels)``.
        to : str, Profile or colour space
            Destination profile or colour space.
        intent : Intent or str, optional
            Rendering intent. Defaults to ``"relative"``.
        bpc : bool, optional
            Apply black point compensation. Defaults to False.
        proof : str, Profile or colour space, optional
            Soft-proofing profile. Defaults to None.
        gamut_check : bool, optional
            Mark out-of-gamut colours when proofing. Defaults to False.
        optimize : bool, optional
            Allow transform optimization. Defaults to True.
        precision : str, optional
            Evaluation precision, as in :func:`lcms2.convert`. Defaults to ``"auto"``.
        flags : Flag, int or str, optional
            Extra transform flags. Defaults to None.
        out : numpy.ndarray, optional
            Array to write the result to. Defaults to None.
        out_dtype : dtype, optional
            Data type of the result. Defaults to None.

        Returns
        -------
        numpy.ndarray
            The converted values.

        Raises
        ------
        CMSError
            If LittleCMS cannot build the transform.

        See Also
        --------
        lcms2.convert : The equivalent function taking a source argument.

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
