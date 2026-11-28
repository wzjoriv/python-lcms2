"""
Perceptual diagnostics and colour-vision-deficiency simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..spaces import oklab, srgb, srgb_linear
from ._cvd_data import MATRICES

if TYPE_CHECKING:
    from ._colormap import Colormap

_ALIASES = {
    "protan": "protan",
    "protanopia": "protan",
    "protanomaly": "protan",
    "deutan": "deutan",
    "deuteranopia": "deutan",
    "deuteranomaly": "deutan",
    "tritan": "tritan",
    "tritanopia": "tritan",
    "tritanomaly": "tritan",
}


def cvd_matrix(deficiency: str, severity: float) -> np.ndarray:
    """The 3x3 linear-sRGB matrix for ``deficiency`` at ``severity`` (0..1), interpolated."""
    key = _ALIASES.get(str(deficiency).lower())
    if key is None:
        raise ValueError(f"deficiency must be 'protan', 'deutan' or 'tritan', got {deficiency!r}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in 0..1, got {severity}")
    x = severity * 10.0
    lo = min(int(np.floor(x + 1e-9)), 10)
    hi = min(lo + 1, 10)
    w = x - lo
    return MATRICES[key][lo] * (1 - w) + MATRICES[key][hi] * w


class DiagnosticsMixin:
    """Lightness, step-size and colour-vision diagnostics for `Colormap`."""

    __slots__ = ()

    def lightness(self) -> np.ndarray:
        """OKLab lightness ``L`` of each entry.

        Returns
        -------
        ndarray, shape (n,)
            Lightness per entry, 0..1.

        See Also
        --------
        Colormap.is_lightness_monotonic : Whether it never reverses.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> L = Colormap.from_colors(["black", "white"], n=3).lightness()
        >>> L.round(2).tolist()
        [0.0, 0.5, 1.0]
        """
        return srgb.to(oklab, self.colors)[:, 0]

    def perceptual_deltas(self) -> np.ndarray:
        """Euclidean OKLab distance between consecutive entries.

        Returns
        -------
        ndarray, shape (n - 1,)
            Distances between neighbouring entries.

        See Also
        --------
        Colormap.uniformity : A single summary number.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["black", "white"], n=3).perceptual_deltas().round(2).tolist()
        [0.5, 0.5]
        """
        lab = srgb.to(oklab, np.clip(self.colors, 0.0, 1.0))
        return np.linalg.norm(np.diff(lab, axis=0), axis=1)

    def uniformity(self) -> float:
        """Coefficient of variation of the perceptual step sizes (lower is more uniform).

        Meant for continuous maps (sequential, diverging, cyclic). A qualitative map's
        colors are categories, so equal steps between them say nothing about its quality.

        Returns
        -------
        float
            ``std / mean`` of `perceptual_deltas`; ``0.0`` for fewer than two steps.

        See Also
        --------
        Colormap.perceptual_deltas : The step sizes.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["black", "white"], n=64).uniformity() < 0.01
        True
        """
        d = self.perceptual_deltas()
        if len(d) < 2 or d.mean() == 0:
            return 0.0
        return float(d.std() / d.mean())

    def is_lightness_monotonic(self, tol: float = 1e-6) -> bool:
        """Whether OKLab lightness never reverses direction along the map.

        A map of constant lightness (most cyclic maps) counts as monotonic; check
        `lightness` too when that matters.

        Parameters
        ----------
        tol : float, optional
            Largest reversal still ignored. Defaults to 1e-6.

        Returns
        -------
        bool
            True when lightness is non-decreasing or non-increasing.

        See Also
        --------
        Colormap.lightness : The values checked.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["black", "white"], n=8).is_lightness_monotonic()
        True
        >>> Colormap.from_colors(["black", "white", "black"], n=8).is_lightness_monotonic()
        False
        """
        d = np.diff(self.lightness())
        return bool(np.all(d >= -tol) or np.all(d <= tol))

    def simulate(self, deficiency: str, severity: float = 1.0) -> Colormap:
        """Simulate colour-vision deficiency (Machado, Oliveira and Fernandes 2009).

        Parameters
        ----------
        deficiency : {"protan", "deutan", "tritan"}
            Missing or weakened cone class (``"protanopia"`` style spellings also work).
        severity : float, optional
            0 (normal vision) to 1 (dichromacy); matrices are tabulated every 0.1 and
            linearly interpolated. Applied in linear sRGB. Defaults to 1.0.

        Returns
        -------
        Colormap
            A new colormap showing what the original looks like.

        Raises
        ------
        ValueError
            If `deficiency` is unknown or `severity` is outside 0..1.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.from_colors(["red", "green"], n=2)
        >>> cm.simulate("deutan", 0.0) == cm
        True
        >>> cm.simulate("deutan").hex()
        ['#a39000', '#776a18']
        """
        m = cvd_matrix(deficiency, float(severity))
        if severity == 0:
            return type(self)(self.colors, name=self.name, space=self.space)
        lin = srgb.to(srgb_linear, np.clip(self.colors, 0.0, 1.0))
        out = np.clip(srgb_linear.to(srgb, np.clip(lin @ m.T, 0.0, 1.0)), 0.0, 1.0)
        return type(self)(out, name=f"{self.name} ({deficiency} {severity:g})", space=self.space)
