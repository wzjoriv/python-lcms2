"""
The immutable `Colormap` class: construction, sampling and slicing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._parallel import parallel_worthwhile, run_chunks
from .._parse import parse_color, parse_colors, to_hex
from ._build import interpolate, resolve_space
from ._diagnostics import DiagnosticsMixin
from ._export import ExportMixin
from ._generators import GeneratorsMixin

__all__ = ["Colormap"]


class Colormap(GeneratorsMixin, DiagnosticsMixin, ExportMixin):
    """An immutable colormap: sampled sRGB colours from 0 to 1.

    Build one with `Colormap.from_colors`, a generator (`Colormap.sequential`,
    `Colormap.diverging`, `Colormap.cyclic`, `Colormap.qualitative`) or by name with
    `Colormap.get`. Call it on data to get colours.

    Parameters
    ----------
    colors : array_like, shape (n, 3)
        sRGB floats in 0..1 (values outside are kept only with ``gamut="none"``).
    name : str, optional
        Display name.
    space : str, optional
        Name of the interpolation space the colours were built in (informational).

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.colormap import Colormap
    >>> cm = Colormap.from_colors(["black", "white"], n=5, name="ramp")
    >>> len(cm), cm.name
    (5, 'ramp')
    >>> cm(np.array([0.0, 0.5, 1.0])).round(2)
    array([[0.  , 0.  , 0.  ],
           [0.39, 0.39, 0.39],
           [1.  , 1.  , 1.  ]])
    >>> cm(np.array([0, 5, 10]), vmin=0, vmax=10, bytes=True)
    array([[  0,   0,   0],
           [ 99,  99,  99],
           [255, 255, 255]], dtype=uint8)
    """

    __slots__ = ("_colors", "_name", "_space")

    def __init__(self, colors: Any, name: str | None = None, space: Any = "oklab") -> None:
        arr = np.array(colors, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            raise ValueError(f"colors must have shape (n, 3) with n >= 1, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("colors must be finite")
        arr.setflags(write=False)
        object.__setattr__(self, "_colors", arr)
        object.__setattr__(self, "_name", name or "colormap")
        label = space if isinstance(space, str) else getattr(space, "name", "custom")
        object.__setattr__(self, "_space", label)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Colormap is immutable")

    @classmethod
    def from_colors(
        cls,
        colors: Any,
        *,
        space: Any = "oklab",
        n: int = 256,
        positions: Any = None,
        name: str | None = None,
        gamut: str = "chroma",
    ) -> Colormap:
        """Interpolate control colours into a colormap.

        Parameters
        ----------
        colors : sequence
            Two or more colours in any form `lcms2._parse.parse_color` accepts.
        space : str or ColorSpace, default "oklab"
            Space the colours are interpolated in. Cylindrical spaces (``oklch``,
            ``lch``, ``hsv``) interpolate hue along the shorter arc.
        n : int, default 256
            Number of samples.
        positions : sequence of float or "uniform", optional
            Increasing positions of the control colours; rescaled to span 0..1.
            Default: evenly spaced (stops at 0, 1/(k-1), ..., 1). ``"uniform"`` places
            each stop at its cumulative OKLab distance from the first, so the perceived
            colour change per step is the same along the whole map.
        name : str, optional
        gamut : {"chroma", "clip", "none"}, default "chroma"
            ``"chroma"`` reduces OKLCh chroma at constant lightness and hue until the
            colour fits sRGB; ``"clip"`` clips channels; ``"none"`` leaves them.

        Returns
        -------
        Colormap

        Raises
        ------
        ValueError
            For fewer than two colours, bad positions, ``n < 2`` or an unknown gamut mode.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.from_colors(["#000", "#f80", "#fff"], n=3)
        >>> cm.hex()
        ['#000000', '#ff8800', '#ffffff']
        >>> cm = Colormap.from_colors(["#000", "#f80", "#fff"], positions="uniform")
        """
        ctrl = parse_colors(colors)[:, :3]
        if len(ctrl) < 2:
            raise ValueError("need at least two colours to interpolate")
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if positions is None:
            pos = np.linspace(0.0, 1.0, len(ctrl))
        elif isinstance(positions, str):
            if positions != "uniform":
                raise ValueError(f'positions must be a sequence or "uniform", got {positions!r}')
            from ..spaces import oklab, srgb

            steps = np.linalg.norm(np.diff(srgb.to(oklab, ctrl), axis=0), axis=1)
            if np.any(steps <= 0):
                raise ValueError('positions="uniform" needs consecutive colours to differ')
            pos = np.concatenate([[0.0], np.cumsum(steps)]) / steps.sum()
        else:
            pos = np.asarray(positions, dtype=np.float64)
            if pos.shape != (len(ctrl),):
                raise ValueError(f"positions must have {len(ctrl)} entries, got shape {pos.shape}")
            if np.any(np.diff(pos) <= 0):
                raise ValueError("positions must be strictly increasing")
            pos = (pos - pos[0]) / (pos[-1] - pos[0])
        sp = resolve_space(space)
        rgb = interpolate(ctrl, sp, pos, n, gamut)
        return cls(rgb, name=name or "custom", space=sp.name)

    @classmethod
    def get(cls, name: str) -> Colormap:
        """A built-in colormap by name (append ``_r`` to reverse); see `lcms2.colormap.list`.

        Raises
        ------
        KeyError
            For an unknown name; the message lists the available ones.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.get("glacier_r").name
        'glacier_r'
        """
        from . import _builtin

        return _builtin.get(name, cls)

    # ---- Basic protocol ----
    @property
    def colors(self) -> np.ndarray:
        """Read-only ``(n, 3)`` array of sRGB colours."""
        return self._colors

    @property
    def name(self) -> str:
        """The display name."""
        return self._name

    @property
    def space(self) -> str:
        """Name of the interpolation space."""
        return self._space

    def __len__(self) -> int:
        return self._colors.shape[0]

    def __repr__(self) -> str:
        c = to_hex(self._colors[[0, -1]])
        return f"Colormap({self._name!r}, n={len(self)}, space={self._space!r}, {c[0]} -> {c[1]})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Colormap):
            return NotImplemented
        return self._colors.shape == other._colors.shape and bool(np.array_equal(self._colors, other._colors))

    def __hash__(self) -> int:
        return hash(self._colors.tobytes())

    def hex(self) -> list[str]:
        """The colours as ``"#rrggbb"`` strings.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["red", "blue"], n=2).hex()
        ['#ff0000', '#0000ff']
        """
        return to_hex(self._colors)

    def reversed(self) -> Colormap:
        """A copy with the colours in reverse order.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["red", "blue"], n=2).reversed().hex()
        ['#0000ff', '#ff0000']
        """
        return Colormap(self._colors[::-1], name=self._name + "_r", space=self._space)

    def resampled(self, n: int) -> Colormap:
        """Linearly resample to ``n`` entries (endpoints preserved).

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> len(Colormap.from_colors(["red", "blue"], n=2).resampled(10))
        10
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
        return Colormap(self(t), name=self._name, space=self._space)

    def __getitem__(self, key: Any) -> Colormap:
        """Slice into a sub-colormap, e.g. ``cm[64:192]`` or ``cm[::-1]``."""
        if not isinstance(key, slice):
            raise TypeError("index a Colormap with a slice (cm[a:b]); use cm.colors[i] for one colour")
        sub = self._colors[key]
        if len(sub) == 0:
            raise ValueError("slice selects no colours")
        return Colormap(sub, name=self._name, space=self._space)

    # ---- Sampling ----
    def _lookup(self, v: np.ndarray, nan: np.ndarray, lo: float, hi: float, fill: np.ndarray) -> np.ndarray:
        """Linearly interpolated colours of `v` scaled from ``lo..hi``; NaN entries get `fill`."""
        t = (v - lo) / (hi - lo) if hi != lo else np.zeros_like(v)
        t = np.clip(np.where(nan, 0.0, t), 0.0, 1.0) * (len(self) - 1)
        i0 = np.minimum(np.floor(t).astype(int), max(len(self) - 2, 0))
        frac = (t - i0)[..., None]
        i1 = np.minimum(i0 + 1, len(self) - 1)
        out = self._colors[i0] * (1 - frac) + self._colors[i1] * frac
        return np.where(nan[..., None], fill, out)

    def __call__(
        self,
        values: Any,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        bytes: bool = False,  # noqa: A002
        alpha: Any = None,
        nan_color: Any = None,
    ) -> np.ndarray:
        """Map data to colours.

        Parameters
        ----------
        values : array_like
            Data of any shape. With neither `vmin` nor `vmax` the values are taken as
            already in 0..1; otherwise a missing limit defaults to the data minimum or
            maximum. Values outside the range are clamped.
        vmin, vmax : float, optional
        bytes : bool, default False
            Return ``uint8`` 0..255 instead of floats 0..1.
        alpha : float or array_like, optional
            Append an alpha channel (broadcast against `values`).
        nan_color : colour, optional
            Colour for NaN (default mid grey; fully transparent when `alpha` is given).

        Returns
        -------
        ndarray, shape (..., 3) or (..., 4)

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.from_colors(["black", "white"], n=256)
        >>> cm([0.0, 1.0]).tolist()
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        >>> cm([float("nan")], alpha=1.0, bytes=True).tolist()
        [[128, 128, 128, 0]]
        """
        v = np.asarray(values, dtype=np.float64)
        nan = np.isnan(v)
        if vmin is None and vmax is None:
            lo, hi = 0.0, 1.0
        else:
            finite = v[~nan]
            lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
            hi = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)
        fill = np.array([0.5, 0.5, 0.5]) if nan_color is None else parse_color(nan_color)[:3]
        if parallel_worthwhile(v.size):
            flat_v, flat_nan = v.reshape(-1), nan.reshape(-1)
            flat_out = np.empty((v.size, 3))

            def work(start: int, stop: int) -> None:
                flat_out[start:stop] = self._lookup(flat_v[start:stop], flat_nan[start:stop], lo, hi, fill)

            run_chunks(v.size, work)
            out = flat_out.reshape(v.shape + (3,))
        else:
            out = self._lookup(v, nan, lo, hi, fill)
        if alpha is not None:
            a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), v.shape).copy()
            if nan_color is None:
                a[nan] = 0.0
            out = np.concatenate([out, a[..., None]], axis=-1)
        if bytes:
            return np.rint(np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
        return out
