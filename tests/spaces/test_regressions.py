"""Regression tests for bugs fixed while merging the color-space library."""

import subprocess
import sys

import numpy as np
import pytest

import lcms2.icc
import lcms2.spaces
from lcms2.colorimetry import D50, D65
from lcms2.icc import ICCParseError, ICCProfile, ICCSpace
from lcms2.icc._reader import read_circular_channels
from lcms2.spaces import (
    ColorSpace,
    GraySpace,
    LogGraySpace,
    a98_rgb,
    display_p3,
    oklab,
    oklch,
    rec2020,
    srgb,
)

# 1. OKLab ---------------------------------------------------------------------------------------

OKLAB_REFERENCE = {
    "white": ([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
    "red": ([1.0, 0.0, 0.0], [0.627955, 0.224863, 0.125846]),
    "blue": ([0.0, 0.0, 1.0], [0.452014, -0.032457, -0.311528]),
    "green": ([0.0, 1.0, 0.0], [0.866440, -0.233888, 0.179498]),
    "black": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("name", OKLAB_REFERENCE)
def test_oklab_matches_the_canonical_definition(name):
    rgb, expected = OKLAB_REFERENCE[name]
    np.testing.assert_allclose(srgb.to(oklab, rgb), expected, atol=1e-6)


def test_oklab_white_is_lightness_one_and_achromatic():
    out = srgb.to(oklab, [1.0, 1.0, 1.0])
    assert abs(out[0] - 1.0) < 1e-6
    assert np.all(np.abs(out[1:]) < 1e-6)


@pytest.mark.parametrize("name", ["red", "blue", "green"])
def test_oklab_roundtrips_to_srgb(name):
    rgb = np.array(OKLAB_REFERENCE[name][0])
    np.testing.assert_allclose(oklab.to(srgb, srgb.to(oklab, rgb)), rgb, atol=1e-9)


def test_oklch_is_the_polar_form_of_oklab():
    for rgb, (L, a, b) in ((v[0], v[1]) for v in OKLAB_REFERENCE.values()):
        out = srgb.to(oklch, rgb)
        assert abs(out[0] - L) < 1e-6
        assert abs(out[1] - np.hypot(a, b)) < 1e-6
        if np.hypot(a, b) > 1e-3:
            assert abs(out[2] - (np.arctan2(b, a) / (2 * np.pi)) % 1.0) < 1e-5


def test_oklch_red_reference():
    out = srgb.to(oklch, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out, [0.627955, 0.257683, 29.2339 / 360.0], atol=2e-5)


# 2. user spaces accept lists ----------------------------------------------------------------------


def test_callable_space_accepts_lists_and_scalars_in_a_list():
    cs = ColorSpace(
        n_channels=2,
        to_xyz=lambda v: np.stack([v[..., 0], np.ones_like(v[..., 0]), v[..., 1]], axis=-1),
        from_xyz=lambda xyz: xyz[..., [0, 2]],
        name="listy",
    )
    np.testing.assert_allclose(cs.to_xyz([0.25, 0.5]), [0.25, 1.0, 0.5])
    np.testing.assert_allclose(cs.to_xyz([[0.25, 0.5], [1.0, 2.0]]), [[0.25, 1, 0.5], [1, 1, 2]])
    np.testing.assert_allclose(cs.from_xyz([0.25, 1.0, 0.5]), [0.25, 0.5])


def test_subclass_space_accepts_lists_and_integer_arrays():
    class Doubler(ColorSpace):
        name = "doubler-regression"
        n_channels = 1

        def to_xyz(self, v):
            return np.repeat(v * 2.0, 3, axis=-1)

        def from_xyz(self, xyz):
            return xyz[..., :1] / 2.0

    cs = Doubler()
    np.testing.assert_allclose(cs.to_xyz([0.5]), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cs.to_xyz(np.array([[1]])), [[2.0, 2.0, 2.0]])
    np.testing.assert_allclose(cs.from_xyz([2, 2, 2]), [1.0])


def test_builtin_spaces_accept_lists():
    np.testing.assert_allclose(srgb.to_xyz([1, 1, 1]), D50, atol=1e-6)
    np.testing.assert_allclose(srgb.to(oklab, [[1, 0, 0]]), [[0.627955, 0.224863, 0.125846]], atol=1e-6)


# 3. gray spaces -----------------------------------------------------------------------------------


@pytest.mark.parametrize("black", [0.0, 1.0, -0.1, 2.0, np.nan, "x", None])
def test_loggray_rejects_black_outside_open_unit_interval(black):
    with pytest.raises(ValueError, match="black"):
        LogGraySpace(black=black)


@pytest.mark.parametrize("white", [[1.0, 1.0], [[1.0, 1.0, 1.0]], 1.0])
def test_loggray_and_gray_reject_bad_white_shape(white):
    with pytest.raises(ValueError, match="white"):
        LogGraySpace(white=white)
    with pytest.raises(ValueError, match="white"):
        GraySpace(white=white)


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.inf, np.nan, "x"])
def test_gray_rejects_bad_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        GraySpace(gamma=gamma)


@pytest.mark.parametrize("cls", [GraySpace, LogGraySpace])
def test_gray_default_white_is_a_copy_of_d50(cls):
    cs = cls()
    assert cs.white is not D50
    assert not np.shares_memory(cs.white, D50)
    before = D50.copy()
    cs.white[:] = 0.0
    np.testing.assert_array_equal(D50, before)
    assert not np.shares_memory(cls(white=D50).white, D50)


def test_shared_white_points_are_read_only():
    for white in (D50, D65):
        with pytest.raises(ValueError):
            white[0] = 0.0


def test_loggray_roundtrip_with_valid_black():
    cs = LogGraySpace(black=0.01)
    g = np.linspace(0, 1, 5)[:, None]
    np.testing.assert_allclose(cs.from_xyz(cs.to_xyz(g)), g, atol=1e-12)


# 4. one crsC parser -------------------------------------------------------------------------------


def _profile_with_crsc(payload):
    raw = srgb.to_icc_profile(version=4)["bytes"]
    profile = ICCProfile.load(raw)
    profile.tags[b"crsC"] = payload
    return profile


def test_circular_tag_is_parsed_by_the_reader_helper_and_icc_space():
    import struct

    profile = _profile_with_crsc(struct.pack(">II", 1, 2))
    assert read_circular_channels(profile) == frozenset({2})
    assert ICCSpace(profile).circular_device_channels == frozenset({2})


@pytest.mark.parametrize("payload", [b"", b"\x00\x00", b"\x00\x00\x00\x05\x00\x00\x00\x01"])
def test_malformed_circular_tag_raises_parse_error(payload):
    profile = _profile_with_crsc(payload)
    with pytest.raises(ICCParseError, match="crsC"):
        read_circular_channels(profile)
    with pytest.raises(ICCParseError):
        ICCSpace(profile)


# 5. no __version__ in subpackages -----------------------------------------------------------------


def test_subpackages_have_no_version():
    assert not hasattr(lcms2.spaces, "__version__")
    assert not hasattr(lcms2.icc, "__version__")


# Imports ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["lcms2.icc", "lcms2.spaces"])
def test_import_order_does_not_matter(first):
    code = (
        f"import {first}; import lcms2.icc as i, lcms2.spaces as s; "
        "assert s.ICCSpace is i.ICCSpace; assert 'ICCSpace' in s.__all__; "
        "assert issubclass(i.ICCSpace, s.ColorSpace)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


# Extra: white points -------------------------------------------------------------------------------


@pytest.mark.parametrize("space", [srgb, display_p3, rec2020, a98_rgb])
def test_rgb_white_maps_to_d50(space):
    np.testing.assert_allclose(space.to_xyz([1.0, 1.0, 1.0]), D50, atol=1e-6)
