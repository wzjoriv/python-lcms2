"""ICC revision numbers accepted when writing profiles."""

from __future__ import annotations

import numpy as np

from ._errors import ICCWriteError

__all__ = ["normalize_icc_version"]

_DEFAULT_REVISION = {2: (2, 4, 0), 4: (4, 4, 0), 5: (5, 0, 0)}

_KNOWN_REVISIONS = {
    2: frozenset((2, minor, 0) for minor in range(5)),
    4: frozenset((4, minor, 0) for minor in range(5)),
    5: frozenset({(5, 0, 0)}),
}


def normalize_icc_version(version: int | tuple) -> tuple:
    """Return `version` as a published ``(major, minor, bugfix)`` ICC revision.

    Raises
    ------
    ICCWriteError
        If the version is not ICC v2, v4 or v5, or not a published revision.
    """
    if isinstance(version, (int, np.integer)) and not isinstance(version, bool):
        if int(version) not in _DEFAULT_REVISION:
            raise ICCWriteError(f"unsupported ICC major version {version!r}; use 2, 4 or 5")
        return _DEFAULT_REVISION[int(version)]
    version = tuple(version)  # type: ignore[arg-type]
    if len(version) != 3 or not all(isinstance(v, (int, np.integer)) for v in version):
        raise ICCWriteError(
            f"version must be 2, 4, 5 or a (major, minor, bugfix) tuple; got {version!r}"
        )
    version = tuple(int(v) for v in version)
    if version[0] not in _KNOWN_REVISIONS:
        raise ICCWriteError(f"unsupported ICC major version {version[0]!r}; use 2, 4 or 5")
    if version not in _KNOWN_REVISIONS[version[0]]:
        known = ", ".join(".".join(map(str, r)) for r in sorted(_KNOWN_REVISIONS[version[0]]))
        raise ICCWriteError(
            f"unknown ICC revision {'.'.join(map(str, version))}; known revisions: {known}"
        )
    return version
