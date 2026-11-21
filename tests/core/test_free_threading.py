"""
Free-threaded CPython: the extension keeps the GIL disabled and transforms are safe to share.
"""

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import lcms2

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded build")
def test_import_keeps_gil_disabled():
    assert not sys._is_gil_enabled()


def test_shared_transform_from_many_threads():
    t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("Lab"), "Lab_DBL")
    rng = np.random.default_rng(0)
    batches = [rng.integers(0, 256, (2048, 3), dtype=np.uint8) for _ in range(32)]
    expected = [t.apply(b) for b in batches]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(t.apply, batches))
    for got, want in zip(results, expected, strict=True):
        np.testing.assert_array_equal(got, want)
