"""NumPy Lab/LCh/xyY conversions, adaptation, white points and CIECAM02."""

import numpy as np
import pytest

from lcms2 import _lcms2
from lcms2.colorimetry import (
    CIECAM02,
    D50,
    D65,
    D50_xyY,
    D65_xyY,
    adapt,
    lab_to_lch,
    lab_to_xyz,
    lch_to_lab,
    temperature_from_white_point,
    white_point_from_temperature,
    xyy_to_xyz,
    xyz_to_lab,
    xyz_to_xyy,
)

RNG = np.random.default_rng(7)


def test_xyz_to_lab_matches_lcms():
    xyz = RNG.uniform(0, 1.1, size=(2000, 3))
    xyz[:5] = [[0, 0, 0], [1e-6, 1e-6, 1e-6], [0.005, 0.005, 0.005], [D50[0], 1, D50[2]], [0.2, 0.0088, 0.3]]
    np.testing.assert_allclose(xyz_to_lab(xyz), _lcms2._xyz_to_lab(D50, xyz), atol=1e-10, rtol=0)
    np.testing.assert_allclose(xyz_to_lab(xyz, D65), _lcms2._xyz_to_lab(D65, xyz), atol=1e-10, rtol=0)


def test_lab_to_xyz_matches_lcms():
    lab = RNG.uniform([0, -100, -100], [100, 100, 100], size=(2000, 3))
    lab[:3] = [[0, 0, 0], [1, 0, 0], [8.0, 1, -1]]
    np.testing.assert_allclose(lab_to_xyz(lab), _lcms2._lab_to_xyz(D50, lab), atol=1e-10, rtol=0)
    np.testing.assert_allclose(lab_to_xyz(lab, D65), _lcms2._lab_to_xyz(D65, lab), atol=1e-10, rtol=0)


def test_lab_round_trip():
    xyz = RNG.uniform(0, 1, size=(100, 3))
    np.testing.assert_allclose(lab_to_xyz(xyz_to_lab(xyz)), xyz, atol=1e-12)


def test_lch_round_trip_and_range():
    lab = RNG.uniform([0, -100, -100], [100, 100, 100], size=(100, 3))
    lch = lab_to_lch(lab)
    assert (lch[:, 2] >= 0).all() and (lch[:, 2] < 360).all()
    np.testing.assert_allclose(lch_to_lab(lch), lab, atol=1e-10)


def test_xyy_round_trip_and_black():
    xyz = RNG.uniform(0.01, 1, size=(50, 3))
    np.testing.assert_allclose(xyy_to_xyz(xyz_to_xyy(xyz)), xyz, atol=1e-12)
    np.testing.assert_allclose(xyz_to_xyy([0, 0, 0])[:2], D50_xyY[:2])
    np.testing.assert_array_equal(xyy_to_xyz([0.3, 0.0, 0.5]), [0, 0, 0])


def test_constants_are_consistent_and_read_only():
    np.testing.assert_allclose(xyz_to_xyy(D50), D50_xyY, atol=1e-4)
    np.testing.assert_allclose(xyz_to_xyy(D65), D65_xyY, atol=1e-4)
    with pytest.raises(ValueError):
        D50[0] = 1.0


@pytest.mark.parametrize("method", ["bradford", "von_kries", "xyz_scaling"])
def test_adapt_maps_white_to_white_and_inverts(method):
    np.testing.assert_allclose(adapt(D65, D65, D50, method), D50, atol=1e-12)
    xyz = RNG.uniform(0, 1, size=(20, 3))
    back = adapt(adapt(xyz, D65, D50, method), D50, D65, method)
    np.testing.assert_allclose(back, xyz, atol=1e-12)


def test_adapt_bradford_d65_to_d50_known_value():
    # sRGB red primary under D65 -> D50 (the sRGB profile colorant).
    out = adapt([0.4124, 0.2126, 0.0193], D65, D50)
    np.testing.assert_allclose(out, [0.4360, 0.2225, 0.0139], atol=2e-3)


def test_adapt_unknown_method():
    with pytest.raises(ValueError, match="unknown adaptation method"):
        adapt(D65, D65, D50, "nope")


def test_d65_temperature():
    assert float(temperature_from_white_point(D65_xyY)) == pytest.approx(6504, abs=2)
    assert float(temperature_from_white_point(D65_xyY[:2])) == pytest.approx(6504, abs=2)


def test_temperature_round_trip_and_shapes():
    k = np.array([4000.0, 5003.0, 6504.0, 9000.0])
    xyy = white_point_from_temperature(k)
    assert xyy.shape == (4, 3)
    np.testing.assert_allclose(xyy[:, 2], 1.0)
    np.testing.assert_allclose(temperature_from_white_point(xyy), k, rtol=2e-3)
    assert white_point_from_temperature(6504).shape == (3,)
    np.testing.assert_allclose(white_point_from_temperature(6504), D65_xyY, atol=2e-3)
    assert float(temperature_from_white_point(D50_xyY)) == pytest.approx(5003, abs=10)


def test_temperature_out_of_range():
    with pytest.raises(ValueError, match="4000 K and 25000 K"):
        white_point_from_temperature(1000)


# CIECAM02 worked example (Fairchild, "Color Appearance Models"; CIE 159): sample
# XYZ = (19.01, 20.00, 21.78) under white (95.05, 100, 108.88), La = 318.31, Yb = 20, average
# surround, D from the model: J = 41.73, C = 0.1047, h = 219.0.
def test_ciecam02_published_example():
    cam = CIECAM02([0.9505, 1.0, 1.0888], adapting_luminance=318.31, background=20, surround="average")
    j, c, h = cam.forward([0.1901, 0.2, 0.2178])
    assert j == pytest.approx(41.73, abs=0.01)
    assert c == pytest.approx(0.1047, abs=0.002)
    assert h == pytest.approx(219.0, abs=0.1)


def test_ciecam02_white_has_j_100():
    for surround in ("average", "dim", "dark", "cutsheet"):
        cam = CIECAM02(D65, 64, 20, surround)
        assert cam.forward(D65)[0] == pytest.approx(100.0, abs=1e-3)


def test_ciecam02_round_trip_vectorised():
    cam = CIECAM02(D65, 40, 20, "dim", degree=0.9)
    xyz = RNG.uniform(0.05, 0.9, size=(8, 4, 3))
    jch = cam.forward(xyz)
    assert jch.shape == xyz.shape
    # LittleCMS uses rounded model matrices, so the inverse is exact to about 1e-5.
    np.testing.assert_allclose(cam.reverse(jch), xyz, atol=2e-4)


def test_ciecam02_validation():
    with pytest.raises(ValueError, match="surround"):
        CIECAM02(D65, 64, 20, "bright")
    with pytest.raises(ValueError, match="degree"):
        CIECAM02(D65, 64, 20, degree=1.5)
    with pytest.raises(ValueError, match="white_point"):
        CIECAM02([1, 2], 64)
    assert "CIECAM02" in repr(CIECAM02(D65))
