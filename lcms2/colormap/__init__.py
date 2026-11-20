"""
Perceptually designed colormaps, gamut-mapped to sRGB.

Get a map in one line and use it in matplotlib in two::

    >>> from lcms2.colormap import Colormap
    >>> cm = Colormap.get("glacier")
    >>> cm(0.5)                                    # doctest: +SKIP
    >>> plt.imshow(data, cmap=cm.to_matplotlib())  # doctest: +SKIP

`Colormap.from_colors` interpolates your own colours in OKLab (or any colour space),
the generators (`Colormap.sequential`, `Colormap.diverging`, `Colormap.cyclic`,
`Colormap.qualitative`) design maps from lightness and hue, and `list` names the
built-in set.
"""

from __future__ import annotations

from ._builtin import get  # noqa: A004
from ._builtin import names as list
from ._colormap import Colormap

__all__ = ["Colormap", "get", "list"]
