import numpy as np
import pytest

from lcms2.colormap import Colormap
from lcms2.colormap import list as list_maps
from lcms2.colormap._build import oklab_to_lch
from lcms2.spaces import oklab, srgb

SEQUENTIAL = ["glacier", "ember", "meadow", "orchid", "lagoon", "sunrise"]


@pytest.mark.parametrize("name", SEQUENTIAL)
def test_sequential_builtin_quality(name):
    cm = Colormap.get(name)
    assert cm.is_lightness_monotonic()
    assert np.all(np.diff(cm.lightness()) > 0)
    assert cm.uniformity() < 0.1
    assert cm.colors.min() >= 0 and cm.colors.max() <= 1


def test_sequential_options():
    cm = Colormap.sequential(250, lightness=(0.9, 0.2), n=64)
    assert len(cm) == 64 and np.all(np.diff(cm.lightness()) < 0)
    assert np.allclose(cm.lightness()[[0, -1]], [0.9, 0.2], atol=2e-3)
    assert Colormap.sequential("#2060c0").is_lightness_monotonic()
    sweep = Colormap.sequential(260, 170)
    h = oklab_to_lch(srgb.to(oklab, sweep.colors[100:150]))[:, 2] * 360
    assert h.min() > 150 and h.max() < 280
    with pytest.raises(ValueError):
        Colormap.sequential(10, n=1)


@pytest.mark.parametrize("n", [255, 256])
def test_diverging_symmetric(n):
    cm = Colormap.diverging(250, 40, n=n)
    assert np.allclose(cm.lightness(), cm.lightness()[::-1], atol=1e-6)
    assert np.all(np.diff(cm.lightness()[: n // 2]) > 0)  # lightness rises toward the centre
    assert cm.lightness().argmax() in (n // 2, (n - 1) // 2)
    assert cm.colors.min() >= 0 and cm.colors.max() <= 1


def test_diverging_center_options():
    auto = Colormap.diverging("#2060c0", "#c04020", center=None, n=3)
    dark = Colormap.diverging(250, 40, center="#202020", n=3)
    assert auto.hex()[1] == "#f2f2f2"
    assert dark.hex()[1] == "#202020"
    assert auto.colors[0, 2] > auto.colors[0, 0] and auto.colors[-1, 0] > auto.colors[-1, 2]


def test_cyclic():
    cm = Colormap.cyclic(30, n=120)
    assert cm.lightness().std() < 0.005
    lab = srgb.to(oklab, cm.colors)
    steps = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    assert np.linalg.norm(lab[0] - lab[-1]) < 2.5 * steps.max()  # wraps seamlessly
    h = oklab_to_lch(lab)[:, 2]
    assert np.ptp(np.unwrap(h * 2 * np.pi)) / (2 * np.pi) > 0.9


@pytest.mark.parametrize("k", [1, 3, 8, 12])
def test_qualitative_distinct(k):
    cm = Colormap.qualitative(k)
    assert len(cm) == k
    if k > 1:
        lab = srgb.to(oklab, cm.colors)
        dist = np.linalg.norm(lab[:, None] - lab[None], axis=-1) + np.eye(k) * 9
        assert dist.min() > 0.07
    with pytest.raises(ValueError):
        Colormap.qualitative(0)


def test_builtin_names_present():
    assert {"glacier", "polar", "halo", "parade"} <= set(list_maps())
