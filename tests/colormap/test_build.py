import numpy as np
import pytest

from lcms2.colormap import Colormap
from lcms2.colormap._build import gamut_map_oklab, oklab_to_lch
from lcms2.spaces import oklab, srgb


def test_endpoints_exact():
    for space in ("oklab", "oklch", "lch", "hsv", "srgb", "xyz-d50", "display_p3"):
        cm = Colormap.from_colors(["#102030", "#a0b0c0"], space=space, n=33)
        assert np.array_equal(cm.colors[0], np.array([16, 32, 48]) / 255)
        assert np.array_equal(cm.colors[-1], np.array([160, 176, 192]) / 255)


def test_space_instance_and_errors():
    assert Colormap.from_colors(["red", "blue"], space=oklab, n=3).space == "oklab"
    with pytest.raises(ValueError, match="unknown colour space"):
        Colormap.from_colors(["red", "blue"], space="nope")
    with pytest.raises(ValueError):
        Colormap.from_colors(["red"])
    with pytest.raises(ValueError):
        Colormap.from_colors(["red", "blue"], n=1)
    with pytest.raises(ValueError, match="gamut"):
        Colormap.from_colors(["red", "blue"], gamut="bad")
    with pytest.raises(ValueError, match="strictly increasing"):
        Colormap.from_colors(["red", "green", "blue"], positions=[0, 0.5, 0.5])
    with pytest.raises(ValueError, match="3 entries"):
        Colormap.from_colors(["red", "green", "blue"], positions=[0, 1])


def test_positions_place_control_colours():
    cm = Colormap.from_colors(["#000", "#f00", "#fff"], positions=[0, 0.25, 1], n=5)
    assert cm.hex()[1] == "#ff0000"
    cm = Colormap.from_colors(["#000", "#f00", "#fff"], positions=[10, 12.5, 20], n=5)
    assert cm.hex()[1] == "#ff0000"


def test_hue_takes_shortest_arc():
    # red (hue ~29 deg) to blue (~264 deg): shortest arc goes through magenta (hue ~328), not green
    mid = Colormap.from_colors(["red", "blue"], space="oklch", n=3).colors[1]
    hue = oklab_to_lch(srgb.to(oklab, mid))[2] * 360
    assert 290 < hue < 360
    # and from near 350 to near 10 degrees passes through 0, not 180
    a = oklab.to(srgb, np.array([0.7, 0.1 * np.cos(np.radians(350)), 0.1 * np.sin(np.radians(350))]))
    b = oklab.to(srgb, np.array([0.7, 0.1 * np.cos(np.radians(10)), 0.1 * np.sin(np.radians(10))]))
    m = Colormap.from_colors([a, b], space="oklch", n=3).colors[1]
    h = oklab_to_lch(srgb.to(oklab, m))[2] * 360
    assert min(h, 360 - h) < 5


def test_achromatic_endpoint_borrows_hue():
    cm = Colormap.from_colors(["white", "#2060c0"], space="oklch", n=9)
    hues = oklab_to_lch(srgb.to(oklab, cm.colors[1:-1]))[:, 2] * 360
    target = oklab_to_lch(srgb.to(oklab, cm.colors[-1:]))[0, 2] * 360
    assert np.allclose(hues, target, atol=2)


@pytest.mark.parametrize("mode", ["chroma", "clip"])
def test_gamut_modes_stay_inside(mode):
    cm = Colormap.from_colors(["#00ffff", "#ff00ff", "#ffff00"], space="oklch", n=64, gamut=mode)
    assert cm.colors.min() >= 0 and cm.colors.max() <= 1
    lab = np.stack([np.full(40, 0.6), np.linspace(-0.4, 0.4, 40), np.linspace(0.4, -0.4, 40)], -1)
    out = gamut_map_oklab(lab, mode)
    assert out.min() >= 0 and out.max() <= 1


def test_chroma_gamut_preserves_lightness_and_hue():
    lab = np.array([[0.6, 0.3, 0.2], [0.8, -0.3, 0.1], [0.4, 0.05, -0.3], [0.5, 0.02, 0.01]])
    out = gamut_map_oklab(lab, "chroma")
    assert out.min() >= 0 and out.max() <= 1
    got = srgb.to(oklab, out)
    assert np.allclose(got[:, 0], lab[:, 0], atol=2e-3)
    ref, res = oklab_to_lch(lab), oklab_to_lch(got)
    assert np.all(res[:, 1] <= ref[:, 1] + 1e-6)
    dh = np.abs(np.mod(res[:3, 2] - ref[:3, 2] + 0.5, 1) - 0.5)
    assert np.all(dh < 2e-3)
    assert np.allclose(got[3], lab[3], atol=1e-6)  # in gamut: unchanged


def test_gamut_none_leaves_out_of_range():
    out = gamut_map_oklab(np.array([[0.6, 0.35, 0.2]]), "none")
    assert out.min() < 0 or out.max() > 1
    cm = Colormap.from_colors(["#00ffff", "#ff00ff"], space="display_p3", n=9, gamut="none")
    assert cm.colors.shape == (9, 3)
