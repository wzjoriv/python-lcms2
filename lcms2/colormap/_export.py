"""Export to matplotlib, images, notebooks and files; load from files."""

from __future__ import annotations

import csv
import html
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .._parse import parse_colors, to_hex

_PLOT_HINT = "matplotlib is required for this: pip install python-lcms2[plot]"


def _matplotlib():
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(_PLOT_HINT) from e
    return matplotlib


def png_bytes(image: np.ndarray, use_pillow: bool = True) -> bytes:
    """Encode a ``(H, W, 3|4)`` uint8 array as PNG (Pillow if present, else pure zlib)."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if use_pillow:
        try:
            import io

            from PIL import Image
        except ImportError:
            pass
        else:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            return buf.getvalue()
    h, w, c = image.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))
    header = struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class ExportMixin:
    """Interop, image and file I/O for `Colormap`."""

    __slots__ = ()

    def to_matplotlib(self):
        """Convert to a matplotlib ``ListedColormap``.

        Raises
        ------
        ImportError
            If matplotlib is missing (``pip install python-lcms2[plot]``).

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> cmap = Colormap.from_colors(["black", "white"], n=16).to_matplotlib()  # doctest: +SKIP
        >>> cmap.N  # doctest: +SKIP
        16
        """
        _matplotlib()
        from matplotlib.colors import ListedColormap

        return ListedColormap(np.clip(self.colors, 0.0, 1.0), name=self.name)

    def register(self, name: str | None = None):
        """Register with matplotlib so ``cmap="name"`` works (and ``"name_r"``).

        Parameters
        ----------
        name : str, optional
            Registry name; defaults to the colormap's own name. Re-registering replaces.

        Returns
        -------
        matplotlib.colors.ListedColormap
        """
        mpl = _matplotlib()
        cmap = self.to_matplotlib()
        if name is not None:
            cmap.name = name
        mpl.colormaps.register(cmap, name=cmap.name, force=True)
        return cmap

    @classmethod
    def from_matplotlib(cls, name_or_cmap: Any, n: int = 256):
        """Sample a matplotlib colormap (by name or instance) into a `Colormap`.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> len(Colormap.from_matplotlib("gray", n=8))  # doctest: +SKIP
        8
        """
        mpl = _matplotlib()
        cmap = mpl.colormaps[name_or_cmap] if isinstance(name_or_cmap, str) else name_or_cmap
        rgba = np.asarray(cmap(np.linspace(0.0, 1.0, n)))
        return cls(rgba[:, :3], name=getattr(cmap, "name", "matplotlib"), space="srgb")

    # -- images and notebooks --------------------------------------------------------

    def to_image(self, width: int = 256, height: int = 32, vertical: bool = False) -> np.ndarray:
        """Render a swatch as a ``(height, width, 3)`` uint8 array.

        With ``vertical=True`` the map runs bottom (first colour) to top, and the
        image is still ``height`` rows by ``width`` columns.

        Examples
        --------
        >>> from lcms2.colormap import Colormap
        >>> Colormap.from_colors(["black", "white"]).to_image(64, 8).shape
        (8, 64, 3)
        """
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        length = height if vertical else width
        t = np.linspace(0.0, 1.0, length)
        strip = self(t, bytes=True)
        if vertical:
            return np.ascontiguousarray(np.repeat(strip[::-1, None, :], width, axis=1))
        return np.ascontiguousarray(np.repeat(strip[None, :, :], height, axis=0))

    def _repr_png_(self) -> bytes:
        return png_bytes(self.to_image(256, 32))

    def _repr_html_(self) -> str:
        step = max(len(self) // 64, 1)
        stops = ", ".join(to_hex(np.clip(self.colors[::step], 0.0, 1.0)))
        title = html.escape(self.name)
        return (
            f'<div style="font-family:sans-serif;font-size:12px"><div>{title} ({len(self)})</div>'
            f'<div style="width:256px;height:24px;border:1px solid #888;'
            f'background:linear-gradient(to right, {stops})"></div></div>'
        )

    # -- files -----------------------------------------------------------------------

    def save(self, path: Any) -> Path:
        """Write to ``.png`` (swatch), ``.csv`` (r,g,b floats), ``.json`` or ``.txt`` (hex lines).

        Returns
        -------
        pathlib.Path
            The written path.

        Raises
        ------
        ValueError
            For any other extension.

        Examples
        --------
        >>> import tempfile, pathlib
        >>> from lcms2.colormap import Colormap
        >>> cm = Colormap.from_colors(["black", "white"], n=4)
        >>> p = cm.save(pathlib.Path(tempfile.mkdtemp()) / "ramp.txt")
        >>> Colormap.load(p).hex() == cm.hex()
        True
        """
        p = Path(path)
        ext = p.suffix.lower()
        colors = np.clip(self.colors, 0.0, 1.0)
        if ext == ".png":
            p.write_bytes(png_bytes(self.to_image(max(len(self), 256), 32)))
        elif ext == ".csv":
            with p.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["r", "g", "b"])
                w.writerows([repr(float(v)) for v in row] for row in colors)
        elif ext == ".json":
            doc = {"name": self.name, "space": self.space, "colors": colors.tolist()}
            p.write_text(json.dumps(doc, indent=1), encoding="utf8")
        elif ext == ".txt":
            p.write_text("\n".join(to_hex(colors)) + "\n", encoding="utf8")
        else:
            raise ValueError(
                f"cannot save to {ext or 'a path without an extension'!r}; use .png, .csv, .json or .txt"
            )
        return p

    @classmethod
    def load(cls, path: Any):
        """Read a colormap written by `save` (``.csv``, ``.json`` or ``.txt``).

        ``.txt`` holds one colour per line in any form `lcms2._parse.parse_color` accepts.
        """
        p = Path(path)
        ext = p.suffix.lower()
        name = p.stem
        space = "srgb"
        if ext == ".csv":
            with p.open(newline="") as f:
                rows = [r for r in csv.reader(f) if r]
            rows = rows[1:] if rows and rows[0][0].strip().lower() == "r" else rows
            colors = np.array([[float(v) for v in r[:3]] for r in rows])
        elif ext == ".json":
            doc = json.loads(p.read_text(encoding="utf8"))
            colors = np.asarray(doc["colors"], dtype=np.float64)
            name, space = doc.get("name", name), doc.get("space", space)
        elif ext == ".txt":
            lines = [ln.strip() for ln in p.read_text(encoding="utf8").splitlines() if ln.strip()]
            colors = parse_colors(lines)[:, :3]
        else:
            raise ValueError(f"cannot load {ext or 'a path without an extension'!r}; use .csv, .json or .txt")
        return cls(colors, name=name, space=space)
