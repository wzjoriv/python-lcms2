"""``lcms2.convert``: one-call conversions between profiles and colour spaces."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ._builtins import lookup_builtin
from ._profile import Profile
from ._transform import Transform

__all__ = ["convert"]

_CACHE_SIZE = 64
_cache: OrderedDict[tuple, tuple[tuple, Transform]] = OrderedDict()
_builtin_profiles: dict[str, Profile] = {}
_lock = threading.Lock()


def _is_space(obj: Any) -> bool:
    return callable(getattr(obj, "to_xyz", None)) and callable(getattr(obj, "from_xyz", None))


def _lookup_space_name(name: str) -> Any:
    """A registered ColorSpace by name, when the ``lcms2.spaces`` package provides one."""
    try:
        from . import spaces  # type: ignore[attr-defined]
    except ImportError:
        return None
    for getter in ("get", "get_space", "lookup"):
        fn = getattr(spaces, getter, None)
        if callable(fn):
            try:
                space = fn(name)
            except (KeyError, LookupError, ValueError, TypeError):
                return None
            return space if _is_space(space) else None
    return None


def _builtin(name: str) -> Profile:
    with _lock:
        p = _builtin_profiles.get(name)
        if p is None:
            p = _builtin_profiles[name] = Profile.builtin(name)
        return p


def _resolve_space(obj: Any) -> Profile | Any:
    """A Profile, or a colour space object (``to_xyz``/``from_xyz``) for analytic conversions.

    Built-in profile names win over registered colour space names; strings that are neither are
    treated as file paths.
    """
    if isinstance(obj, Profile):
        return obj
    if isinstance(obj, str):
        name = lookup_builtin(obj)
        if name is not None:
            return _builtin(name)
        if not os.path.isfile(obj):
            space = _lookup_space_name(obj)
            if space is not None:
                return space
        return Profile(obj)
    if _is_space(obj):
        return obj
    return Profile(obj)


def _hashable(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, Profile):
        return ("profile", id(v))
    return v


def _transform(src: Profile, dst: Profile, options: dict[str, Any]) -> Transform:
    key = (id(src), id(dst), tuple(sorted((k, _hashable(v)) for k, v in options.items())))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit[1]
    t = Transform(src, dst, **options)
    with _lock:
        # keep the profiles (and proof) alive so their ids stay unique while cached
        _cache[key] = ((src, dst, options.get("proof")), t)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return t


def _xyz_profile() -> Profile:
    return _builtin("XYZ")


def convert(
    data: Any,
    src: Any,
    dst: Any,
    *,
    out: np.ndarray | None = None,
    out_dtype: Any = None,
    **options: Any,
) -> np.ndarray:
    """Convert colours from ``src`` to ``dst``.

    ``src`` and ``dst`` may be :class:`Profile` objects, built-in profile names, paths, or colour
    space objects (anything with ``to_xyz``/``from_xyz`` working in XYZ D50, Y = 1). Mixed
    conversions go through XYZ D50. Transforms are cached (LRU) per profile pair and options.

    Parameters
    ----------
    data : array-like
        ``(..., channels)`` pixels.
    src, dst : Profile, str, path or colour space
        Source and destination.
    out, out_dtype
        See :meth:`Transform.apply`.
    **options
        :class:`Transform` keywords: ``intent``, ``bpc``, ``proof``, ``gamut_check``, ``optimize``,
        ``precision``, ``flags``.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import lcms2
    >>> lcms2.convert([255, 255, 255], "sRGB", "Lab").round(2) + 0.0
    array([100.,   0.,   0.])
    >>> lcms2.convert([[1.0, 0.0, 0.0]], "sRGB", "Display P3").round(3)
    array([[0.917, 0.2  , 0.139]])
    """
    s, d = _resolve_space(src), _resolve_space(dst)
    s_profile, d_profile = isinstance(s, Profile), isinstance(d, Profile)
    if s_profile and d_profile:
        return _transform(s, d, options).apply(data, out, out_dtype=out_dtype)
    if s_profile:
        xyz = _transform(s, _xyz_profile(), options).apply(data, out_dtype=np.float64)
    else:
        xyz = np.asarray(s.to_xyz(np.asarray(data, dtype=np.float64)), dtype=np.float64)
    if d_profile:
        return _transform(_xyz_profile(), d, options).apply(xyz, out, out_dtype=out_dtype)
    result = np.asarray(d.from_xyz(xyz))
    if out_dtype is not None:
        result = result.astype(out_dtype)
    if out is not None:
        out[...] = result
        return out
    return result
