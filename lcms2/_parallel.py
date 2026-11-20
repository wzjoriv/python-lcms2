"""
Thread pool shared by the large-array operations (transforms, colour differences).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["get_threads", "parallel_worthwhile", "run_chunks", "set_threads"]

_MIN_ITEMS = 1 << 18
_ITEMS_PER_TASK = 1 << 16
_threads = min(os.cpu_count() or 1, 16)
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def set_threads(n: int | None) -> None:
    """Set how many threads large transforms use.

    Transforms and colour differences over 262,144 pixels or more are split into chunks run in
    parallel (LittleCMS runs without the GIL). Pass 1 to work on the calling thread only.

    Parameters
    ----------
    n : int or None
        Number of threads; ``None`` restores the default, ``min(os.cpu_count(), 16)``.

    Raises
    ------
    ValueError
        If `n` is less than 1.

    See Also
    --------
    get_threads : The current setting.

    Examples
    --------
    >>> import lcms2
    >>> lcms2.set_threads(1)
    >>> lcms2.get_threads()
    1
    >>> lcms2.set_threads(None)
    """
    global _threads, _pool
    if n is None:
        n = min(os.cpu_count() or 1, 16)
    if int(n) < 1:
        raise ValueError(f"threads must be at least 1, got {n}")
    with _pool_lock:
        _threads = int(n)
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None


def get_threads() -> int:
    """Number of threads large transforms use.

    Returns
    -------
    int

    See Also
    --------
    set_threads : Change it.

    Examples
    --------
    >>> import lcms2
    >>> lcms2.get_threads() >= 1
    True
    """
    return _threads


def parallel_worthwhile(n: int) -> bool:
    """Whether ``n`` items are worth splitting across threads."""
    return _threads > 1 and n >= _MIN_ITEMS


def run_chunks(n: int, work: Callable[[int, int], object]) -> None:
    """Call ``work(start, stop)`` over contiguous ranges of ``range(n)`` on the shared pool."""
    global _pool
    tasks = max(1, min(_threads, n // _ITEMS_PER_TASK))
    bounds = np.linspace(0, n, tasks + 1, dtype=np.int64)
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(_threads, thread_name_prefix="lcms2")
        pool = _pool
    jobs = [pool.submit(work, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    for job in jobs:
        job.result()
