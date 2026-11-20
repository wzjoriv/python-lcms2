"""Rendering intents, transform flags and profile header enums."""

from __future__ import annotations

import re
from enum import Enum, IntEnum, IntFlag

__all__ = ["ColorSpaceSignature", "DeviceClass", "Flag", "Intent"]


def _key(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip()).upper()


class Intent(IntEnum):
    """ICC rendering intents plus LittleCMS's black-preserving intents.

    Strings are accepted wherever an intent is expected, case-insensitively, with the short
    aliases ``"relative"`` and ``"absolute"``.

    Examples
    --------
    >>> Intent.parse("relative")
    <Intent.RELATIVE_COLORIMETRIC: 1>
    >>> Intent.parse("Absolute Colorimetric")
    <Intent.ABSOLUTE_COLORIMETRIC: 3>
    """

    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3
    PRESERVE_K_ONLY_PERCEPTUAL = 10
    PRESERVE_K_ONLY_RELATIVE_COLORIMETRIC = 11
    PRESERVE_K_ONLY_SATURATION = 12
    PRESERVE_K_PLANE_PERCEPTUAL = 13
    PRESERVE_K_PLANE_RELATIVE_COLORIMETRIC = 14
    PRESERVE_K_PLANE_SATURATION = 15

    @classmethod
    def parse(cls, value: Intent | int | str) -> Intent:
        """Convert an int or a name (case-insensitive, aliases allowed) to an :class:`Intent`.

        Raises
        ------
        ValueError
            For unknown names or numbers, listing the valid names.
        """
        if isinstance(value, Intent):
            return value
        if isinstance(value, str):
            k = _key(value)
            k = _INTENT_ALIASES.get(k, k)
            if k in cls.__members__:
                return cls[k]
        elif isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                pass
        names = ", ".join(m.name.lower() for m in cls)
        raise ValueError(
            f"unknown rendering intent {value!r}; use one of: {names} (or 'relative'/'absolute')"
        )


_INTENT_ALIASES = {
    "RELATIVE": "RELATIVE_COLORIMETRIC",
    "ABSOLUTE": "ABSOLUTE_COLORIMETRIC",
    "COLORIMETRIC": "RELATIVE_COLORIMETRIC",
}


class Flag(IntFlag):
    """LittleCMS transform flags (``cmsFLAGS_*``).

    ``Flag.parse`` accepts an int, a :class:`Flag` or a string such as ``"NOOPTIMIZE,BLACKPOINTCOMPENSATION"``
    (separators ``,`` ``|`` ``;`` or spaces). ``8BITS_DEVICELINK`` is spelled ``EIGHTBITS_DEVICELINK`` in
    Python.

    Examples
    --------
    >>> Flag.parse("gamutcheck|softproofing") == Flag.GAMUTCHECK | Flag.SOFTPROOFING
    True
    >>> int(Flag.grid_points(33))
    2162688
    """

    NONE = 0
    CLUT_POST_LINEARIZATION = 0x0001
    FORCE_CLUT = 0x0002
    NOWHITEONWHITEFIXUP = 0x0004
    EIGHTBITS_DEVICELINK = 0x0008
    CLUT_PRE_LINEARIZATION = 0x0010
    GUESSDEVICECLASS = 0x0020
    NOCACHE = 0x0040
    KEEP_SEQUENCE = 0x0080
    NOOPTIMIZE = 0x0100
    NULLTRANSFORM = 0x0200
    HIGHRESPRECALC = 0x0400
    LOWRESPRECALC = 0x0800
    GAMUTCHECK = 0x1000
    BLACKPOINTCOMPENSATION = 0x2000
    SOFTPROOFING = 0x4000
    NONEGATIVES = 0x8000
    NODEFAULTRESOURCEDEF = 0x01000000
    COPY_ALPHA = 0x04000000

    @staticmethod
    def grid_points(n: int) -> Flag:
        """Flag requesting ``n`` CLUT grid points per dimension (``cmsFLAGS_GRIDPOINTS``)."""
        if not 0 <= n <= 255:
            raise ValueError(f"grid points must be 0-255, got {n}")
        return Flag((n & 0xFF) << 16)

    @classmethod
    def parse(cls, value: Flag | int | str | None) -> Flag:
        """Convert ``None``, an int or a flag string to a :class:`Flag`."""
        if value is None:
            return cls.NONE
        if isinstance(value, str):
            out = cls.NONE
            for part in filter(None, re.split(r"[\s,;|]+", value)):
                k = _key(part)
                k = {"8BITS_DEVICELINK": "EIGHTBITS_DEVICELINK", "BPC": "BLACKPOINTCOMPENSATION"}.get(k, k)
                if k not in cls.__members__:
                    names = ", ".join(cls.__members__)
                    raise ValueError(f"unknown flag {part!r}; valid flags: {names}")
                out |= cls[k]
            return out
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        raise TypeError(f"flags must be a Flag, an int or a string like 'NOOPTIMIZE,BPC', got {value!r}")


class DeviceClass(str, Enum):
    """ICC profile/device class, with readable values (``"display"``) and the header signature."""

    INPUT = "input"
    DISPLAY = "display"
    OUTPUT = "output"
    LINK = "link"
    ABSTRACT = "abstract"
    COLOR_SPACE = "colorspace"
    NAMED_COLOR = "named color"

    def __str__(self) -> str:
        return self.value

    @property
    def signature(self) -> str:
        """The 4-character header signature (``"mntr"`` for DISPLAY)."""
        return _CLASS_SIGS[self]

    @classmethod
    def from_signature(cls, sig: str) -> DeviceClass | str:
        """The class for a header signature; unknown signatures are returned unchanged."""
        for k, v in _CLASS_SIGS.items():
            if v == sig:
                return k
        return sig


_CLASS_SIGS = {
    DeviceClass.INPUT: "scnr",
    DeviceClass.DISPLAY: "mntr",
    DeviceClass.OUTPUT: "prtr",
    DeviceClass.LINK: "link",
    DeviceClass.ABSTRACT: "abst",
    DeviceClass.COLOR_SPACE: "spac",
    DeviceClass.NAMED_COLOR: "nmcl",
}


class ColorSpaceSignature(str, Enum):
    """ICC data colour spaces by header signature without padding (``"RGB"``, ``"Lab"``, ``"CMYK"``).

    Members compare equal to their strings: ``profile.color_space == "RGB"``.
    """

    XYZ = "XYZ"
    LAB = "Lab"
    LUV = "Luv"
    YCBCR = "YCbr"
    YXY = "Yxy"
    RGB = "RGB"
    GRAY = "GRAY"
    HSV = "HSV"
    HLS = "HLS"
    CMYK = "CMYK"
    CMY = "CMY"
    MCH5 = "5CLR"
    MCH6 = "6CLR"
    MCH7 = "7CLR"
    MCH8 = "8CLR"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_signature(cls, sig: str) -> ColorSpaceSignature | str:
        """The member for a (possibly space-padded) signature; unknown ones come back stripped."""
        s = sig.rstrip()
        try:
            return cls(s)
        except ValueError:
            return s
