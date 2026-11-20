import numpy as np
import pytest

from lcms2.colormap import Colormap, get
from lcms2.colormap import list as list_maps


@pytest.fixture
def ramp():
    return Colormap.from_colors(["black", "white"], n=11, name="ramp")


def test_immutable(ramp):
    with pytest.raises(AttributeError):
        ramp.name = "x"
    with pytest.raises(ValueError):
        ramp.colors[0, 0] = 1
    with pytest.raises(ValueError):
        Colormap(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        Colormap(np.full((3, 3), np.nan))


def test_call_basics(ramp):
    assert ramp(0.0).shape == (3,)
    assert ramp(np.zeros((4, 5))).shape == (4, 5, 3)
    assert np.allclose(ramp([0.0, 1.0]), [[0] * 3, [1] * 3])
    assert np.allclose(ramp(0.55), (ramp.colors[5] + ramp.colors[6]) / 2)
    assert np.allclose(ramp([-5, 9]), [[0] * 3, [1] * 3])  # clamped


def test_call_vmin_vmax(ramp):
    assert np.allclose(ramp([10, 20], vmin=10, vmax=20), [[0] * 3, [1] * 3])
    data = np.array([2.0, 4.0, 6.0])
    assert np.allclose(ramp(data, vmin=2), ramp([0, 0.5, 1]))  # vmax from data
    assert np.allclose(ramp(data, vmax=6), ramp([0, 0.5, 1]))
    assert np.allclose(ramp([3.0, 3.0], vmin=3, vmax=3), ramp([0.0, 0.0]))  # degenerate


def test_call_nan_alpha_bytes(ramp):
    out = ramp([0.0, np.nan, 1.0], nan_color="red")
    assert np.allclose(out[1], [1, 0, 0])
    assert np.allclose(ramp([np.nan])[0], 0.5)
    rgba = ramp([0.0, np.nan], alpha=0.5)
    assert rgba.shape == (2, 4) and rgba[0, 3] == 0.5 and rgba[1, 3] == 0.0
    assert ramp([np.nan], alpha=0.5, nan_color="blue")[0, 3] == 0.5
    arr = ramp(np.array([[0.0, 1.0]]), alpha=np.array([[1.0, 0.0]]))
    assert arr.shape == (1, 2, 4) and arr[0, 1, 3] == 0
    b = ramp([0.0, 1.0], bytes=True)
    assert b.dtype == np.uint8 and b.tolist() == [[0, 0, 0], [255, 255, 255]]
    assert ramp([0.0], bytes=True, alpha=1.0).tolist() == [[0, 0, 0, 255]]
    assert ramp(np.array([np.nan, np.nan]), vmax=1.0).shape == (2, 3)


def test_single_colour_map():
    cm = Colormap([[0.2, 0.4, 0.6]])
    assert np.allclose(cm([0.0, 1.0]), [[0.2, 0.4, 0.6]] * 2)


def test_reversed_resampled_slicing(ramp):
    r = ramp.reversed()
    assert np.array_equal(r.colors, ramp.colors[::-1]) and r.name == "ramp_r"
    assert r.reversed() == Colormap(ramp.colors, name="other")
    rs = ramp.resampled(5)
    assert len(rs) == 5 and np.allclose(rs.colors[[0, -1]], ramp.colors[[0, -1]])
    assert len(ramp.resampled(1)) == 1
    with pytest.raises(ValueError):
        ramp.resampled(0)
    sub = ramp[2:6]
    assert len(sub) == 4 and np.array_equal(sub.colors, ramp.colors[2:6])
    assert ramp[::-1] == r
    with pytest.raises(TypeError):
        ramp[3]
    with pytest.raises(ValueError):
        ramp[5:5]


def test_eq_hash_repr_len(ramp):
    same = Colormap(ramp.colors, name="different")
    assert ramp == same and hash(ramp) == hash(same)
    assert ramp != ramp.reversed()
    assert ramp != 3
    assert len(ramp) == 11
    text = repr(ramp)
    assert "ramp" in text and "#000000" in text and "#ffffff" in text


def test_hex(ramp):
    assert ramp.hex()[0] == "#000000" and len(ramp.hex()) == 11


def test_builtins():
    names = list_maps()
    assert len(names) >= 8 and names == sorted(names)
    for n in names:
        cm = Colormap.get(n)
        assert cm.name == n and cm.colors.min() >= 0 and cm.colors.max() <= 1
    assert get("glacier_r") == Colormap.get("glacier").reversed()
    assert get("GLACIER") == get("glacier")
    with pytest.raises(KeyError, match="available"):
        Colormap.get("viridis")
