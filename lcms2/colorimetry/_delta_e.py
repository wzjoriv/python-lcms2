"""
CIELAB color differences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .. import _lcms2
from ._core import _triple

__all__ = ["delta_e"]

_DE_METHODS = {"76": 0, "94": 1, "2000": 2, "cmc": 3, "bfd": 4}
_DE_PARAMS = {2: ("Kl", "Kc", "Kh"), 3: ("l", "c")}
_DE_DEFAULTS = {2: (1.0, 1.0, 1.0), 3: (2.0, 1.0)}


def delta_e(lab1: ArrayLike, lab2: ArrayLike, method: str = "2000", **params: float) -> np.ndarray:
    """Color difference between CIELAB colors.

    Parameters
    ----------
    lab1, lab2 : array_like
        Lab values of shape ``(..., 3)``; the two operands broadcast against each other.
    method : {"2000", "76", "94", "cmc", "bfd"}
        CIEDE2000 (default), CIE76 (Euclidean), CIE94, CMC l:c or BFD. A leading "de" or
        "cie" is accepted ("de2000").
    **params
        ``Kl``, ``Kc``, ``Kh`` (all default 1) for ``"2000"``; ``l`` (2) and ``c`` (1) for
        ``"cmc"``.

    Returns
    -------
    numpy.ndarray
        Differences with the broadcast shape minus the last axis (a scalar for one pair).

    Raises
    ------
    ValueError
        For an unknown method or an array whose last axis is not 3.
    TypeError
        For a parameter the chosen method does not take.

    Examples
    --------
    >>> from lcms2.colorimetry import delta_e
    >>> round(float(delta_e([50, 2.6772, -79.7751], [50, 0, -82.7485])), 4)
    2.0425
    >>> float(delta_e([50, 0, 0], [53, 4, 0], method="76"))
    5.0
    """
    key = str(method).lower().removeprefix("cie").removeprefix("de").strip()
    if key not in _DE_METHODS:
        raise ValueError(f"unknown delta E method {method!r}; use one of {sorted(_DE_METHODS)}")
    code = _DE_METHODS[key]
    names = _DE_PARAMS.get(code, ())
    extra = set(params) - set(names)
    if extra:
        accepted = f"accepts {', '.join(names)}" if names else "takes no parameters"
        raise TypeError(f"delta E method {key!r} {accepted}; got {', '.join(sorted(extra))}")
    values = [params.get(n, d) for n, d in zip(names, _DE_DEFAULTS.get(code, ()), strict=False)]
    a, b = np.broadcast_arrays(_triple(lab1, "lab1"), _triple(lab2, "lab2"))
    out = _lcms2.delta_e(np.ascontiguousarray(a), np.ascontiguousarray(b), code, *map(float, values))
    return out[()] if out.ndim == 0 else out
