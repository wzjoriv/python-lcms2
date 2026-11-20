import numpy as np
import pytest

import lcms2
from lcms2.colormap import Colormap


@pytest.fixture
def threads():
    yield
    lcms2.set_threads(None)


@pytest.mark.parametrize(
    "kwargs", [{}, {"vmin": -0.2, "vmax": 0.9}, {"bytes": True, "alpha": 0.5}, {"nan_color": "red"}]
)
def test_parallel_matches_single_thread(threads, kwargs):
    v = np.random.default_rng(6).random((700, 400))
    v[3, :7] = np.nan
    v[5, :2] = [-3.0, 4.0]
    cm = Colormap.get("glacier")
    lcms2.set_threads(1)
    ref = cm(v, **kwargs)
    lcms2.set_threads(8)
    np.testing.assert_array_equal(cm(v, **kwargs), ref)


def test_auto_range_uses_the_whole_array(threads):
    v = np.linspace(-5.0, 5.0, 600 * 500).reshape(600, 500)
    cm = Colormap.from_colors(["black", "white"], n=256)
    lcms2.set_threads(8)
    out = cm(v, vmin=None, vmax=v.max())
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[-1, -1], 1.0)
