"""
Tag access, tag readers and capability queries of :class:`lcms2.Profile`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _lcms2
from ._constants import Intent

__all__ = ["ProfileData"]

_DIRECTIONS = {"input": 0, "output": 1, "proof": 2}


def _wrap_curve(cap: Any) -> Any:
    """A ToneCurve for a curve capsule when the ToneCurve class is available, else the capsule."""
    try:
        from ._curves import ToneCurve  # type: ignore[import-not-found]
    except ImportError:
        return cap
    for ctor in ("_from_handle", "from_handle"):
        if hasattr(ToneCurve, ctor):
            return getattr(ToneCurve, ctor)(cap)
    return cap


def _direction(direction: str | int) -> int:
    if isinstance(direction, int):
        return direction
    try:
        return _DIRECTIONS[direction.lower()]
    except KeyError:
        raise ValueError(f"direction must be 'input', 'output' or 'proof', got {direction!r}") from None


class ProfileData:
    """Mixin with the tag-level API; the concrete class stores the capsule in ``_handle``."""

    __slots__ = ()
    _handle: Any

    # ---- Tags ----
    @property
    def tags(self) -> list[str]:
        """Tag signatures present in the profile (4-character strings)."""
        return _lcms2.profile_tags(self._handle)

    def has_tag(self, sig: str) -> bool:
        """Whether the profile has tag ``sig``."""
        return _lcms2.profile_is_tag(self._handle, sig)

    def read_tag_raw(self, sig: str) -> bytes:
        """Serialised tag data including its 8-byte type header. KeyError if absent."""
        return _lcms2.profile_read_raw_tag(self._handle, sig)

    def write_tag_raw(self, sig: str, data: bytes) -> None:
        """Store raw tag data (with its type header) under ``sig``."""
        _lcms2.profile_write_raw_tag(self._handle, sig, data)
        self._handle = _lcms2.profile_open(_lcms2.profile_to_bytes(self._handle))
        self._source_space = None

    def _xyz(self, sig: str) -> np.ndarray | None:
        v = _lcms2.profile_read_xyz(self._handle, sig)
        return None if v is None else np.array(v)

    @property
    def media_white_point(self) -> np.ndarray | None:
        """XYZ of the media white point tag, or None."""
        return self._xyz("wtpt")

    @property
    def luminance(self) -> np.ndarray | None:
        """XYZ of the luminance tag (cd/m²), or None."""
        return self._xyz("lumi")

    @property
    def colorants(self) -> np.ndarray | None:
        """3x3 array of the rXYZ, gXYZ, bXYZ rows, or None when not a matrix-shaper."""
        rows = [self._xyz(s) for s in ("rXYZ", "gXYZ", "bXYZ")]
        return None if any(r is None for r in rows) else np.array(rows)

    @property
    def chromatic_adaptation(self) -> np.ndarray | None:
        """The ``chad`` 3x3 matrix, or None."""
        m = _lcms2.profile_read_chad(self._handle)
        return None if m is None else np.array(m)

    @property
    def trc(self) -> dict[str, Any]:
        """Tone curves by tag (``rTRC``, ``gTRC``, ``bTRC``, ``kTRC``) that are present."""
        out = {}
        for sig in ("rTRC", "gTRC", "bTRC", "kTRC"):
            c = _lcms2.profile_read_trc(self._handle, sig)
            if c is not None:
                out[sig] = _wrap_curve(c)
        return out

    @property
    def named_colors(self) -> list[tuple] | None:
        """Named colours ``(name, prefix, suffix, pcs, colorants)`` or None."""
        return _lcms2.profile_named_colors(self._handle)

    # ---- Capabilities ----
    @property
    def is_matrix_shaper(self) -> bool:
        """Whether the profile is a matrix-shaper."""
        return _lcms2.profile_is_matrix_shaper(self._handle)

    def is_clut(self, intent: Any = "perceptual", direction: str = "input") -> bool:
        """Whether ``intent`` is implemented by a CLUT in ``direction`` (input/output/proof)."""
        return _lcms2.profile_is_clut(self._handle, int(Intent.parse(intent)), _direction(direction))

    def is_intent_supported(self, intent: Any, direction: str = "input") -> bool:
        """Whether the profile implements ``intent`` in ``direction`` (input/output/proof)."""
        return _lcms2.profile_is_intent_supported(
            self._handle, int(Intent.parse(intent)), _direction(direction)
        )

    @property
    def supported_intents(self) -> list[Intent]:
        """ICC intents the profile implements as input or output."""
        h = self._handle
        return [
            i
            for i in list(Intent)[:4]
            if _lcms2.profile_is_intent_supported(h, int(i), 0)
            or _lcms2.profile_is_intent_supported(h, int(i), 1)
        ]

    @property
    def tac(self) -> float:
        """Total area coverage (max ink sum, percent) of an output profile; 0 when not applicable."""
        return _lcms2.profile_detect_tac(self._handle)

    def black_point(self, intent: Any = "perceptual", *, destination: bool = False) -> np.ndarray | None:
        """Detected black point XYZ (Y = 1 white) or None.

        Examples
        --------
        >>> import lcms2
        >>> bp = lcms2.Profile("sRGB").black_point("relative")
        >>> bool(bp[1] < 0.01)
        True
        """
        v = _lcms2.profile_detect_black_point(self._handle, int(Intent.parse(intent)), 0, destination)
        return None if v is None else np.array(v)
