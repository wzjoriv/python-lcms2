import threading

import numpy as np
import pytest

import lcms2
from lcms2.spaces import ColorSpace, hsv, oklab, oklch, srgb


@pytest.fixture
def threads():
    yield
    lcms2.set_threads(None)


@pytest.mark.parametrize("target", [oklab, oklch, hsv])
def test_parallel_matches_single_thread(threads, target):
    data = np.random.default_rng(4).random((700, 400, 3))
    lcms2.set_threads(1)
    ref = srgb.to(target, data)
    lcms2.set_threads(8)
    np.testing.assert_array_equal(srgb.to(target, data), ref)


def test_user_spaces_stay_on_the_calling_thread(threads):
    seen = set()

    class Recording(ColorSpace):
        name = "recording-test"
        n_channels = 3

        def to_xyz(self, values):
            seen.add(threading.get_ident())
            return np.asarray(values, dtype=np.float64)

        def from_xyz(self, xyz):
            return np.asarray(xyz, dtype=np.float64)

    lcms2.set_threads(8)
    Recording().to(oklab, np.random.default_rng(5).random((700, 400, 3)))
    assert seen == {threading.get_ident()}
