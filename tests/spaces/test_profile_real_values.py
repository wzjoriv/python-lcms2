"""
Generated profiles take and return a space's real values, matching the exact NumPy math.
"""

import warnings

import numpy as np
import pytest

import lcms2
from lcms2 import Profile, spaces

# Max dE2000 between the profile and the exact math, on in-gamut colors (default grid).
TOL = {"oklab": 0.5, "lch": 1.0, "oklch": 1.5, "cmyk": 0.5, "xyz-d50": 0.01, "xyz-d65": 0.02}


@pytest.fixture(scope="module")
def xyz():
    rgb = np.random.default_rng(7).random((2000, 3)) * 0.98 + 0.01
    return spaces.srgb.to_xyz(rgb)


@pytest.mark.parametrize("name", sorted(TOL))
def test_forward_matches_exact_math(name, xyz):
    cs = spaces.get(name)
    dev = cs.from_xyz(xyz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = Profile.from_bytes(cs.to_icc_profile()["bytes"])
    got = profile.transform(dev, "Lab", intent="relative")
    want = lcms2.colorimetry.xyz_to_lab(cs.to_xyz(dev))
    assert lcms2.delta_e(got, want, method="2000").max() <= TOL[name]


ROUND_TRIP = [
    "oklab",
    "xyz-d50",
    "cmyk",
    "lch",
    "oklch",
]


@pytest.mark.parametrize("name", ROUND_TRIP)
def test_round_trip_keeps_the_color(name, xyz):
    cs = spaces.get(name)
    dev = cs.from_xyz(xyz)
    profile = cs.profile
    back = Profile("Lab").transform(
        profile.transform(dev, "Lab", intent="relative"), profile, intent="relative"
    )
    lab = lcms2.colorimetry.xyz_to_lab
    assert lcms2.delta_e(lab(cs.to_xyz(back)), lab(cs.to_xyz(dev)), method="2000").max() <= 5 * TOL[name]


def test_profile_transform_agrees_with_convert():
    v = np.array([0.7, 0.1, -0.05])
    np.testing.assert_allclose(
        spaces.oklab.profile.transform(v, "Lab"), lcms2.convert(v, "oklab", "Lab"), atol=0.5
    )


def test_profile_from_space_uses_the_exact_math_on_output():
    cmyk = Profile.from_space(spaces.cmyk)
    rgb = np.array([[255, 255, 255], [255, 0, 0], [0, 0, 0]], np.uint8)
    np.testing.assert_array_equal(
        Profile("sRGB").transform(rgb, cmyk), [[0, 0, 0, 0], [0, 255, 255, 0], [0, 0, 0, 255]]
    )


def test_loaded_profile_wraps_hue_across_360():
    loaded = Profile.from_bytes(spaces.lch.to_icc_profile()["bytes"])
    lch = np.array([[60.0, 40.0, 359.0], [60.0, 40.0, 1.0]])
    back = Profile("Lab").transform(loaded.transform(lch, "Lab"), loaded)
    d = np.abs(back[:, 2] - lch[:, 2])
    assert np.minimum(d, 360.0 - d).max() < 2.0
