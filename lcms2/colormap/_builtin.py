"""
A small set of named colormaps made by this package's own generators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SPECS: dict[str, Callable[[Any], Any]] = {
    "glacier": lambda c: c.sequential(250, name="glacier"),
    "ember": lambda c: c.sequential(35, 70, lightness=(0.14, 0.97), chroma=0.15, name="ember"),
    "meadow": lambda c: c.sequential(165, 110, chroma=0.13, name="meadow"),
    "orchid": lambda c: c.sequential(330, name="orchid"),
    "lagoon": lambda c: c.sequential(260, 170, chroma=0.12, name="lagoon"),
    "sunrise": lambda c: c.sequential(300, 80, lightness=(0.2, 0.95), chroma=0.15, name="sunrise"),
    "polar": lambda c: c.diverging(250, 40, name="polar"),
    "verdant": lambda c: c.diverging(160, 330, name="verdant"),
    "halo": lambda c: c.cyclic(30, name="halo"),
    "parade": lambda c: c.qualitative(8, name="parade"),
}


def names() -> list[str]:
    """Sorted names of the built-in colormaps."""
    return sorted(_SPECS)


def get(name: str, cls: type | None = None) -> Any:
    """Build the named colormap (a ``_r`` suffix reverses it).

    Examples
    --------
    >>> from lcms2.colormap import get
    >>> get("glacier").name
    'glacier'
    """
    if cls is None:
        from ._colormap import Colormap as cls
    key = name.lower()
    reverse = key.endswith("_r") and key not in _SPECS
    base = key[:-2] if reverse else key
    if base not in _SPECS:
        raise KeyError(f"no built-in colormap {name!r}; available: {names()}")
    cm = _SPECS[base](cls)
    return cm.reversed() if reverse else cm
