"""
The `ColorSpace` base class and its registry.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike

from .._parallel import parallel_worthwhile, run_chunks
from ._domain import check_circular_channels, resolve_channels_and_domain

if TYPE_CHECKING:
    from .._profile import Profile

__all__ = ["ColorSpace"]

_REGISTRY: dict[str, type[ColorSpace]] = {}


def _shape(values: Any) -> tuple:
    shape = getattr(values, "shape", None)
    return tuple(shape) if shape is not None else np.shape(values)


def _validated(fn: Callable, n_in: int, n_out: int, method: str) -> Callable:
    def call(values: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(values, "shape") or (isinstance(values, np.ndarray) and values.dtype.kind != "f"):
            values = np.asarray(values, dtype=np.float64)
        in_shape = _shape(values)
        if len(in_shape) == 0 or in_shape[-1] != n_in:
            raise ValueError(f"{method}: expected last dimension {n_in}, got shape {in_shape}")
        out = fn(values, *args, **kwargs)
        out_shape = _shape(out)
        if out_shape != in_shape[:-1] + (n_out,):
            raise ValueError(
                f"{method}: returned shape {out_shape} for input shape {in_shape}, "
                f"expected {in_shape[:-1] + (n_out,)}"
            )
        return out

    return call


class ColorSpace:
    """A color space defined by conversions between device values and CIE XYZ (D50).

    Define one by subclassing and overriding `to_xyz`/`from_xyz`, or by
    passing `to_xyz`/`from_xyz` callables.

    Parameters
    ----------
    n_channels : int, optional
        Number of device channels. Defaults to the class attribute, and to the
        length of `device_domain` when that lists one pair per channel.
    to_xyz : callable, optional
        Maps ``(..., n_channels)`` device values to ``(..., 3)`` XYZ.
        Required unless a subclass overrides `to_xyz`.
    from_xyz : callable, optional
        Maps ``(..., 3)`` XYZ to ``(..., n_channels)`` device values.
        Required unless a subclass overrides `from_xyz`.
    name : str, optional
        Display name. Defaults to the registered name.
    device_domain : tuple or list of tuple, optional
        Device value range written to ICC profiles: one ``(lo, hi)`` pair
        shared by every channel, or one pair per channel. Defaults to the
        class attribute, ``(0.0, 1.0)``. Values outside it are clamped in a
        saved profile. Each end must be finite and ``lo < hi``.

    Attributes
    ----------
    n_channels : int
        Number of device channels.
    device_domain : list of tuple
        One ``(lo, hi)`` pair per channel, whichever form was given.
    circular_device_channels : frozenset of int
        Device channels holding an angle; they wrap over their `device_domain` range (0..360 for degrees).
    icc_colour_space_signature : bytes or None
        ICC data colour space written to profiles; inferred when None.

    Raises
    ------
    TypeError
        If a direction has no implementation, or has both an override and a
        callable, or the channel count is neither declared nor implied by
        `device_domain`.
    ValueError
        If `n_channels` disagrees with `device_domain`, a range is invalid, or
        `circular_device_channels` names a channel the space does not have.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.spaces import ColorSpace
    >>> gray = ColorSpace(
    ...     n_channels=1,
    ...     to_xyz=lambda v: np.repeat(v, 3, axis=-1),
    ...     from_xyz=lambda xyz: xyz[..., 1:2],
    ...     name="flat-gray",
    ... )
    >>> gray.to_xyz([0.5]).tolist()
    [0.5, 0.5, 0.5]
    """

    n_channels: int | None = None
    icc_colour_space_signature: bytes | None = None
    circular_device_channels: frozenset = frozenset()
    device_domain: tuple | list = (0.0, 1.0)
    _registry_name: str | None = None

    def __init__(
        self,
        n_channels: int | None = None,
        to_xyz: Callable | None = None,
        from_xyz: Callable | None = None,
        name: str | None = None,
        device_domain: tuple | list | None = None,
    ):
        cls = type(self)
        declared = n_channels if n_channels is not None else self.n_channels
        domain = device_domain if device_domain is not None else self.device_domain
        try:
            self.n_channels, self.device_domain = resolve_channels_and_domain(declared, domain)
        except TypeError as e:
            raise TypeError(f"{cls.__name__}: {e}") from e
        self.circular_device_channels = check_circular_channels(
            self.circular_device_channels, self.n_channels
        )

        for direction, given, n_in, n_out in (
            ("to_xyz", to_xyz, self.n_channels, 3),
            ("from_xyz", from_xyz, 3, self.n_channels),
        ):
            overridden = getattr(cls, direction) is not getattr(ColorSpace, direction)
            if overridden and given is not None:
                raise TypeError(
                    f"{cls.__name__} overrides {direction}() and was also given a {direction}= callable"
                )
            if not overridden and given is None:
                raise TypeError(f"{cls.__name__} has no {direction} implementation")
            impl = given if given is not None else getattr(cls, direction).__get__(self, cls)
            setattr(self, direction, _validated(impl, n_in, n_out, f"{cls.__name__}.{direction}"))

        self.name = name or cls._registry_name or cls.__name__

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register the subclass under its ``name`` class attribute or class
        name, lowercased, unless it sets ``_abstract = True``."""
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract", False):
            return
        key = cls.__dict__.get("name")
        key = _key(key if isinstance(key, str) else cls.__name__)
        cls._registry_name = key
        existing = _REGISTRY.get(key)
        redefined = existing is not None and (existing.__module__, existing.__qualname__) == (
            cls.__module__,
            cls.__qualname__,
        )
        if existing is not None and existing is not cls and not redefined:
            warnings.warn(
                f"ColorSpace {cls.__module__}.{cls.__qualname__} replaces "
                f"{existing.__module__}.{existing.__qualname__} as {key!r}; "
                "set a distinct `name` class attribute to keep both",
                stacklevel=2,
            )
        _REGISTRY[key] = cls

    @classmethod
    def list(cls) -> list:
        """Return the sorted names of all registered color spaces.

        Returns
        -------
        list of str
        """
        return sorted(_REGISTRY)

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
        """Convert ``data`` from this space to ``to``, exactly like :meth:`lcms2.Profile.transform`.

        ``to`` may be a colour space, a profile, a built-in profile name or an ICC file; see
        :func:`lcms2.convert` for the keywords (they apply when a profile is involved).

        Examples
        --------
        >>> from lcms2.spaces import oklab
        >>> oklab.transform([1.0, 0.0, 0.0], "sRGB").round(3)
        array([1., 1., 1.])
        """
        from .._convert import convert

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

    def to(self, target: str | ColorSpace, array: Any) -> np.ndarray:
        """Convert values from this space into `target`.

        Parameters
        ----------
        target : str or ColorSpace
            A registered name (see `ColorSpace.list`) or a color space instance.
        array : array_like, shape (..., n_channels)
            Values in this space.

        Returns
        -------
        ndarray, shape (..., target.n_channels)

        Raises
        ------
        KeyError
            If `target` is an unregistered name.
        TypeError
            If `target` names a space that needs constructor arguments.

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2.spaces import srgb, xyz_d50
        >>> srgb.to(xyz_d50, np.array([1.0, 1.0, 1.0])).round(4)
        array([0.9642, 1.    , 0.8249])
        """
        from .._convert import _is_color_text

        if _is_color_text(array):
            from .._parse import parse_color, parse_colors

            array = parse_color(array) if isinstance(array, str) else parse_colors(array)
        if (isinstance(target, str) and _key(target) not in _REGISTRY) or (
            not isinstance(target, (str, ColorSpace))
        ):
            from .._convert import convert

            return convert(array, self, target)
        if isinstance(target, str):
            target_cls = _REGISTRY[_key(target)]
            try:
                target = get(target)
            except TypeError as e:
                raise TypeError(f"{target_cls.__name__} needs constructor arguments; pass an instance") from e
        arr = np.asarray(array)
        if target is self:
            return np.array(arr, dtype=np.float64)
        n = arr.size // self.n_channels
        if arr.ndim < 2 or arr.shape[-1] != self.n_channels or not parallel_worthwhile(n):
            return target.from_xyz(self.to_xyz(arr))
        src = arr.reshape(n, self.n_channels)
        out = np.empty((n, target.n_channels))

        def work(a: int, b: int) -> None:
            out[a:b] = target.from_xyz(self.to_xyz(src[a:b]))

        run_chunks(n, work)
        return out.reshape(*arr.shape[:-1], target.n_channels)

    def to_xyz(self, values: Any) -> np.ndarray:
        """Convert device values to CIE XYZ (D50).

        Parameters
        ----------
        values : array_like, shape (..., n_channels)

        Returns
        -------
        ndarray, shape (..., 3)

        Raises
        ------
        ValueError
            If the input's last dimension is not `n_channels`, or the result
            does not have shape ``(..., 3)``.
        """
        raise NotImplementedError

    def from_xyz(self, xyz: Any) -> np.ndarray:
        """Convert CIE XYZ (D50) to device values.

        Parameters
        ----------
        xyz : array_like, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., n_channels)

        Raises
        ------
        ValueError
            If the input's last dimension is not 3, or the result does not
            have shape ``(..., n_channels)``.
        """
        raise NotImplementedError

    @property
    def profile(self) -> Profile:
        """This space as an :class:`lcms2.Profile` (built once, exactly when possible).

        Examples
        --------
        >>> from lcms2.spaces import display_p3
        >>> display_p3.profile.color_space == "RGB"
        True
        """
        cached = self.__dict__.get("_profile")
        if cached is None:
            from .._profile import Profile

            cached = self.__dict__["_profile"] = Profile.from_space(self)
        return cached

    def to_icc_profile(
        self,
        path: str | None = None,
        *,
        version: int | tuple = 4,
        rendering_intent: int = 1,
        grid_size: int | None = None,
        description: str | None = None,
        copyright: str = "No copyright, generated by python-lcms2",
        force_clut: bool = False,
        strict: bool = False,
    ) -> dict:
        """Write this space as an ICC profile.

        Parameters
        ----------
        path : str, optional
            File to write the profile to.
        version : int or tuple of int, optional
            ``2``, ``4``, ``5`` or a published ``(major, minor, bugfix)``
            revision such as ``(4, 2, 0)``. Default 4.
        rendering_intent : int, optional
            Header rendering intent. Default 1 (relative colorimetric).
        grid_size : int, optional
            Grid points per axis of sampled lookup tables.
        description : str, optional
            Profile description text.
        copyright : str, optional
            Profile copyright text.
        force_clut : bool, optional
            Always write sampled lookup tables.
        strict : bool, optional
            Raise :class:`~lcms2.ICCWriteError` instead of falling back to sampled lookup
            tables when `to_xyz` or `from_xyz` has no exact representation, for example
            because it branches on values or calls a non-NumPy library. The message names
            the operation that could not be traced.

        Returns
        -------
        dict
            ``{'bytes': bytes, 'method': str, 'notes': list of str}`` where
            `method` is the representation written:

            - ``'exact_matrix_trc'``: v2/v4 rXYZ/gXYZ/bXYZ matrix and TRC curves.
            - ``'exact_gray_trc'``: v2/v4 kTRC gray curve.
            - ``'exact_lut'``: v2/v4 A2B0/B2A0 lookup tables holding the
              curves and linear map exactly.
            - ``'exact_mpet'``: v5 chain of matrix and curve elements.
            - ``'exact_calc'``: v5 calculator elements.
            - ``'clut_fallback'``: sampled lookup tables; `notes` says why.

        Raises
        ------
        ICCWriteError
            If the version or `device_domain` is invalid, if `to_xyz` or `from_xyz`
            fails on every input, or with ``strict=True`` if there is no exact
            representation.
        """
        from ..icc import normalize_icc_version, write_icc_bytes

        result = write_icc_bytes(
            self,
            version=normalize_icc_version(version),
            rendering_intent=rendering_intent,
            grid_size=grid_size,
            description=description,
            copyright=copyright,
            force_clut=force_clut,
            strict=strict,
        )
        if path is not None:
            with open(path, "wb") as f:
                f.write(result["bytes"])
        return result

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} n_channels={self.n_channels}>"


_INSTANCES: dict[type, ColorSpace] = {}


def _key(name: object) -> str:
    return str(name).lower().replace("_", "-")


def get(name: str) -> ColorSpace:
    """The registered color space called `name` (case-insensitive; ``_`` and ``-`` match), as an instance.

    The instance is shared: repeated calls return the same object until another class is
    registered under that name.

    Raises
    ------
    KeyError
        If no space is registered under `name`.

    Examples
    --------
    >>> from lcms2.spaces import get
    >>> get("OKLab").n_channels
    3
    """
    cls = _REGISTRY.get(_key(name))
    if cls is None:
        raise KeyError(f"no color space registered as {name!r}; registered: {ColorSpace.list()}")
    space = _INSTANCES.get(cls)
    if space is None:
        space = _INSTANCES[cls] = cls()
    return space
