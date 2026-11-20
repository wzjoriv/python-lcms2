"""Channel count and device range of a color space."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def resolve_channels_and_domain(n_channels: int | None, domain) -> tuple[int, list[tuple[float, float]]]:
    """Combine a declared channel count with a device range.

    Parameters
    ----------
    n_channels : int, optional
        Declared channel count.
    domain : (lo, hi) or list of (lo, hi)
        One pair shared by every channel, or one pair per channel.

    Returns
    -------
    tuple
        The channel count, taken from `domain` when `n_channels` is None, and
        one ``(lo, hi)`` pair per channel.

    Raises
    ------
    TypeError
        If the channel count is neither declared nor implied by `domain`.
    ValueError
        If the counts disagree or a pair is not finite with ``lo < hi``.
    """
    try:
        arr = np.asarray(domain, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"device_domain must be (lo, hi) or a list of them, got {domain!r}") from e
    if arr.ndim == 1 and arr.shape == (2,):
        per_channel = False
    elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
        per_channel = True
    else:
        raise ValueError(f"device_domain must be (lo, hi) or a list of them, got {domain!r}")

    if per_channel:
        if n_channels is None:
            n_channels = arr.shape[0]
        elif arr.shape[0] != n_channels:
            raise ValueError(f"device_domain lists {arr.shape[0]} channels but n_channels is {n_channels}")
        pairs = [(float(lo), float(hi)) for lo, hi in arr]
    else:
        if n_channels is None:
            raise TypeError(
                "the channel count is unknown: set n_channels, or give device_domain "
                "as one (lo, hi) pair per channel"
            )
        pairs = [(float(arr[0]), float(arr[1]))] * int(n_channels)

    if isinstance(n_channels, bool) or int(n_channels) != n_channels or n_channels < 1:
        raise ValueError(f"n_channels must be a positive integer, got {n_channels!r}")
    for lo, hi in pairs:
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"device_domain entry {(lo, hi)} must be finite with lo < hi")
    return int(n_channels), pairs


def check_circular_channels(channels: Iterable[int], n_channels: int) -> frozenset:
    """Validate circular channel indices.

    Raises
    ------
    ValueError
        If an index is not a channel of the space.
    """
    channels = frozenset(int(c) for c in channels)
    bad = sorted(c for c in channels if not 0 <= c < n_channels)
    if bad:
        raise ValueError(f"circular_device_channels {bad} are not channels of {n_channels}")
    return channels
