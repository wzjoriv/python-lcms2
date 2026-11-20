import numpy as np
import pytest

import lcms2


@pytest.fixture
def threads():
    yield
    lcms2.set_threads(None)


@pytest.mark.parametrize(
    "dtype,dst,channels",
    [(np.uint8, "Display P3", 3), (np.uint16, "Adobe RGB", 3), (np.float64, "Lab", 3), (np.uint8, "sRGB", 4)],
)
def test_parallel_matches_single_thread(threads, dtype, dst, channels):
    rng = np.random.default_rng(1)
    shape = (700, 400, channels)
    data = (rng.random(shape) * (255 if dtype == np.uint8 else 65535 if dtype == np.uint16 else 1)).astype(
        dtype
    )
    t = lcms2.Transform("sRGB", dst)
    lcms2.set_threads(1)
    ref = t(data)
    lcms2.set_threads(8)
    np.testing.assert_array_equal(t(data), ref)


def test_parallel_writes_into_out(threads):
    data = np.random.default_rng(2).random((600, 500, 3))
    t = lcms2.Transform("sRGB", "Display P3")
    out = np.empty_like(data)
    lcms2.set_threads(4)
    assert t.apply(data, out) is out
    lcms2.set_threads(1)
    np.testing.assert_array_equal(out, t(data))


def test_parallel_gray_without_channel_axis(threads):
    gray = lcms2.Profile.gray(gamma=2.2)
    data = np.random.default_rng(3).random((800, 400))
    t = lcms2.Transform(gray, "sRGB")
    lcms2.set_threads(1)
    ref = t(data)
    lcms2.set_threads(8)
    np.testing.assert_array_equal(t(data), ref)


def test_set_threads_validates(threads):
    with pytest.raises(ValueError, match="at least 1"):
        lcms2.set_threads(0)
    lcms2.set_threads(3)
    assert lcms2.get_threads() == 3
    lcms2.set_threads(None)
    assert lcms2.get_threads() >= 1
