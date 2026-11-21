"""
Large color space conversions run in parallel chunks and match the single-threaded result.
"""

import numpy as np
import pytest

import lcms2
from lcms2 import _lcms2
from lcms2.spaces import srgb


@pytest.fixture
def threads():
    before = lcms2.get_threads()
    yield
    lcms2.set_threads(before)


@pytest.mark.parametrize("target", ["oklab", "oklch", "display-p3", "xyz-d50", "hsv"])
def test_parallel_matches_single_thread(threads, target):
    x = np.random.default_rng(1).random((600, 600, 3))
    lcms2.set_threads(1)
    want = srgb.to(target, x)
    lcms2.set_threads(8)
    np.testing.assert_array_equal(srgb.to(target, x), want)


def test_mat3_matches_matmul():
    m = np.random.default_rng(2).normal(size=(3, 3))
    v = np.random.default_rng(3).normal(size=(4, 5, 3))
    np.testing.assert_allclose(_lcms2.mat3(m, v), v @ m.T, rtol=1e-14, atol=1e-15)


def test_mat3_rejects_bad_shapes():
    with pytest.raises(ValueError):
        _lcms2.mat3(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError):
        _lcms2.mat3(np.eye(3), np.zeros(4))
