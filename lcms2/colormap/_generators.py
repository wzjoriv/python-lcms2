"""
Generators for sequential, diverging, cyclic and qualitative colormaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._parse import parse_color
from ..spaces import oklab, srgb
from ._build import oklab_to_lch, oklch_to_srgb

if TYPE_CHECKING:
    from ._colormap import Colormap


def _hue_chroma(value: Any) -> tuple[float, float | None, float | None]:
    """Hue (degrees) of a number or colour, plus its OKLab chroma and lightness if a colour."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value), None, None
    lch = oklab_to_lch(srgb.to(oklab, parse_color(value)[:3]))
    return float(lch[2] * 360.0), float(lch[1]), float(lch[0])


def _check_n(n: int) -> None:
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")


class GeneratorsMixin:
    """Perceptually designed colormap generators for `Colormap`."""

    __slots__ = ()

    @classmethod
    def sequential(
        cls,
        hue_start: Any,
        hue_end: Any = None,
        *,
        lightness: tuple[float, float] = (0.15, 0.97),
        chroma: float = 0.12,
        n: int = 256,
        name: str | None = None,
    ) -> Colormap:
        """A lightness ramp in OKLCh with monotonic lightness.

        Parameters
        ----------
        hue_start : float or colour
            Starting hue in degrees (0 red, 90 yellow, 145 green, 250 blue), or a colour
            whose hue is used.
        hue_end : float or colour, optional
            Final hue; the hue sweeps linearly from `hue_start` to `hue_end` (the numbers
            are used as given, so ``260`` to ``-100`` goes the other way round). Default:
            a single hue.
        lightness : (float, float), default (0.15, 0.97)
            OKLab lightness at the two ends; ``(0.97, 0.15)`` gives a dark-high map.
        chroma : float, default 0.12
            Peak OKLCh chroma, reduced where sRGB cannot reach it.
        n : int, default 256
        name : str, optional

        Returns
        -------
        Colormap

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.sequential(250)
        >>> cm.is_lightness_monotonic(), len(cm)
        (True, 256)
        """
        _check_n(n)
        h0 = _hue_chroma(hue_start)[0]
        h1 = h0 if hue_end is None else _hue_chroma(hue_end)[0]
        t = np.linspace(0.0, 1.0, n)
        L = lightness[0] + (lightness[1] - lightness[0]) * t
        C = chroma * (0.25 + 0.75 * np.sin(np.pi * t))
        h = (h0 + (h1 - h0) * t) / 360.0
        rgb = oklch_to_srgb(np.stack([L, C, np.mod(h, 1.0)], axis=-1))
        return cls(rgb, name=name or f"sequential-{h0:g}", space="oklch")

    @classmethod
    def diverging(
        cls,
        low: Any,
        high: Any,
        *,
        center: Any = "#f2f2f2",
        n: int = 256,
        name: str | None = None,
    ) -> Colormap:
        """A two-armed map with a neutral centre and lightness symmetric about it.

        Parameters
        ----------
        low, high : float or colour
            Hue in degrees, or a colour, for each arm. Both arms end at the same
            lightness (the mean of the two colours' lightness, or 0.4 for hues).
        center : colour or None, default "#f2f2f2"
            Midpoint colour; ``None`` picks a neutral light grey automatically.
        n : int, default 256
        name : str, optional

        Returns
        -------
        Colormap

        Examples
        --------
        >>> import numpy as np
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.diverging(250, 40)
        >>> bool(np.allclose(cm.lightness(), cm.lightness()[::-1], atol=1e-6))
        True
        """
        _check_n(n)
        (hl, cl, ll), (hh, ch, lh) = _hue_chroma(low), _hue_chroma(high)
        known = [x for x in (ll, lh) if x is not None]
        end_l = float(np.mean(known)) if known else 0.4
        cl = 0.14 if cl is None else max(cl, 0.04)
        ch = 0.14 if ch is None else max(ch, 0.04)
        ends = oklch_to_srgb(np.array([[end_l, cl, hl / 360 % 1.0], [end_l, ch, hh / 360 % 1.0]]))
        mid = parse_color("#f2f2f2" if center is None else center)[:3]
        return cls.from_colors(
            [ends[0], mid, ends[1]], space="oklab", n=n, name=name or "diverging", gamut="chroma"
        )

    @classmethod
    def cyclic(
        cls,
        hue_start: float = 0.0,
        *,
        lightness: float = 0.7,
        chroma: float = 0.1,
        n: int = 256,
        name: str | None = None,
    ) -> Colormap:
        """A constant-lightness hue wheel whose last colour wraps to the first.

        Parameters
        ----------
        hue_start : float, default 0
            Hue in degrees of the first colour; the hue then rotates a full turn
            (the end point is not repeated, so the map wraps seamlessly).
        lightness, chroma : float
            OKLCh lightness and chroma; chroma is reduced where sRGB cannot reach it.
        n : int, default 256
        name : str, optional

        Returns
        -------
        Colormap

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.cyclic(30)
        >>> float(cm.lightness().std()) < 0.01
        True
        """
        _check_n(n)
        h = (hue_start / 360.0 + np.arange(n) / n) % 1.0
        rgb = oklch_to_srgb(np.stack([np.full(n, lightness), np.full(n, chroma), h], axis=-1))
        return cls(rgb, name=name or "cyclic", space="oklch")

    @classmethod
    def qualitative(
        cls,
        k: int,
        *,
        lightness: float = 0.7,
        chroma: float = 0.12,
        hue_start: float = 25.0,
        name: str | None = None,
    ) -> Colormap:
        """``k`` well-separated categorical colours.

        Hues are evenly spaced around OKLCh; beyond six colours the lightness alternates
        slightly so neighbours differ in more than hue. The result has exactly ``k``
        entries (use ``cm.colors[i]`` or ``to_matplotlib()``).

        Parameters
        ----------
        k : int
            Number of colours (at least 1).
        lightness, chroma : float
        hue_start : float, default 25
        name : str, optional

        Returns
        -------
        Colormap

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.qualitative(5).hex()
        ['#df7f78', '#b49e3b', '#3db691', '#50a7e2', '#b887d3']
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        h = (hue_start / 360.0 + np.arange(k) / k) % 1.0
        L = np.full(k, lightness)
        if k > 6:
            L = L + np.where(np.arange(k) % 2 == 0, 0.07, -0.07)
        rgb = oklch_to_srgb(np.stack([L, np.full(k, chroma), h], axis=-1))
        return cls(rgb, name=name or f"qualitative-{k}", space="oklch")
