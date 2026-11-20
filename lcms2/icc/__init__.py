"""Reading and writing ICC profiles (v2, v4 and v5) in pure NumPy.

- `ICCProfile`: a profile's header and raw tags.
- `ICCSpace`: a `~lcms2.spaces.ColorSpace` read from a profile.
- `write_icc_bytes`: serialize any `~lcms2.spaces.ColorSpace` as a profile
  (also available as `ColorSpace.to_icc_profile`).
- `ICCParseError`, `ICCWriteError`: reading and writing errors.
"""

from .._errors import ICCParseError, ICCWriteError
from ._io.profile import ICCProfile
from ._space import ICCSpace
from ._versions import normalize_icc_version
from ._writer import write_icc_bytes

__all__ = [
    "ICCProfile",
    "ICCSpace",
    "ICCParseError",
    "ICCWriteError",
    "write_icc_bytes",
    "normalize_icc_version",
]
