"""CSS Color 4 predefined RGB and XYZ spaces."""

import numpy as np
import pytest

from lcms2.colorimetry import D50
from lcms2.spaces import (
    A98RGBSpace,
    ColorSpace,
    DisplayP3LinearSpace,
    DisplayP3Space,
    ProPhotoRGBSpace,
    Rec2020Space,
    SRGBLinearSpace,
    SRGBSpace,
    XYZD50Space,
    XYZD65Space,
)


def _sample_unit(n_channels, n, seed=0, lo=0.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(lo, hi, size=n) for _ in range(n_channels)], axis=-1)


def test_registered_names():
    names = set(ColorSpace.list())
    for n in (
        "srgb-linear",
        "display-p3-linear",
        "a98-rgb",
        "prophoto-rgb",
        "rec2020",
        "xyz-d65",
        "xyz-d50",
    ):
        assert n in names


def test_srgb_linear_roundtrip_single_and_batch():
    sp = SRGBLinearSpace()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=1)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_srgb_linear_matches_srgb_curve_at_endpoints():
    lin, gam = SRGBLinearSpace(), SRGBSpace()
    assert np.allclose(lin.to_xyz([1.0, 1.0, 1.0]), gam.to_xyz([1.0, 1.0, 1.0]), atol=1e-9)
    assert np.allclose(lin.to_xyz([0.0, 0.0, 0.0]), gam.to_xyz([0.0, 0.0, 0.0]), atol=1e-9)
    mid_lin = lin.to_xyz([0.5, 0.5, 0.5])
    mid_gam = gam.to_xyz([0.5, 0.5, 0.5])
    assert not np.allclose(mid_lin, mid_gam, atol=1e-3)


def test_srgb_linear_write_tier():
    sp = SRGBLinearSpace()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_p3_linear_roundtrip_single_and_batch():
    sp = DisplayP3LinearSpace()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=2)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_p3_linear_white_matches_p3_gamma_white():
    lin, gam = DisplayP3LinearSpace(), DisplayP3Space()
    assert np.allclose(lin.to_xyz([1.0, 1.0, 1.0]), gam.to_xyz([1.0, 1.0, 1.0]), atol=1e-9)


def test_p3_linear_write_tier():
    sp = DisplayP3LinearSpace()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_a98_roundtrip_single_and_batch():
    sp = A98RGBSpace()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=3)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_a98_known_points():
    sp = A98RGBSpace()
    white_xyz = sp.to_xyz([1.0, 1.0, 1.0])
    black_xyz = sp.to_xyz([0.0, 0.0, 0.0])
    srgb_white = SRGBSpace().to_xyz([1.0, 1.0, 1.0])
    assert np.allclose(white_xyz, srgb_white, atol=1e-3)
    assert np.allclose(black_xyz, [0.0, 0.0, 0.0], atol=1e-9)


def test_a98_gamma_exponent():
    sp = A98RGBSpace()
    lin_half = sp.to_xyz([0.5, 0.5, 0.5])
    expected_lin = 0.5 ** (563.0 / 256.0)
    white_y = sp.to_xyz([1.0, 1.0, 1.0])[1]
    assert np.isclose(lin_half[1] / white_y, expected_lin, rtol=1e-6)


def test_a98_write_tier():
    sp = A98RGBSpace()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_prophoto_roundtrip_single_and_batch():
    sp = ProPhotoRGBSpace()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=4)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_prophoto_roundtrip_near_toe():
    sp = ProPhotoRGBSpace()
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [16.0 / 512.0, 16.0 / 512.0, 16.0 / 512.0],
            [0.05, 0.05, 0.05],
            [1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_prophoto_white_is_d50_no_extra_adaptation():
    sp = ProPhotoRGBSpace()
    white_xyz = sp.to_xyz([1.0, 1.0, 1.0])
    assert np.allclose(white_xyz, D50, atol=1e-3)


def test_prophoto_write_tier():
    sp = ProPhotoRGBSpace()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_rec2020_roundtrip_single_and_batch():
    sp = Rec2020Space()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=5)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_rec2020_roundtrip_near_toe():
    sp = Rec2020Space()
    beta = 0.018053968510807
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [beta * 4.5 * 0.5, beta * 4.5 * 0.5, beta * 4.5 * 0.5],
            [beta * 4.5, beta * 4.5, beta * 4.5],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_rec2020_known_points():
    sp = Rec2020Space()
    white_xyz = sp.to_xyz([1.0, 1.0, 1.0])
    black_xyz = sp.to_xyz([0.0, 0.0, 0.0])
    srgb_white = SRGBSpace().to_xyz([1.0, 1.0, 1.0])
    assert np.allclose(white_xyz, srgb_white, atol=1e-3)
    assert np.allclose(black_xyz, [0.0, 0.0, 0.0], atol=1e-9)


def test_rec2020_wider_gamut_than_srgb():
    srgb = SRGBSpace()
    r2020 = Rec2020Space()
    red_xyz = srgb.to_xyz([1.0, 0.0, 0.0])
    r2020_rgb = r2020.from_xyz(red_xyz)
    assert np.all(r2020_rgb >= -1e-6) and np.all(r2020_rgb <= 1.0 + 1e-6)


def test_rec2020_write_tier():
    sp = Rec2020Space()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_xyz_d65_roundtrip_single_and_batch():
    sp = XYZD65Space()
    x = np.array([0.5, 0.4, 0.3])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=6)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_xyz_d65_white_maps_to_d50_pcs_white():
    from lcms2.colorimetry import D65

    sp = XYZD65Space()
    out = sp.to_xyz(D65)
    assert np.allclose(out, D50, atol=1e-6)


def test_xyz_d65_matches_srgb_white_adaptation():
    from lcms2.colorimetry import D65

    sp = XYZD65Space()
    assert np.allclose(sp.to_xyz(D65), SRGBSpace().to_xyz([1.0, 1.0, 1.0]), atol=1e-6)


def test_xyz_d65_write_tier():
    sp = XYZD65Space()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_xyz_d50_is_identity():
    sp = XYZD50Space()
    x = np.array([0.5, 0.4, 0.3])
    assert np.allclose(sp.to_xyz(x), x, atol=1e-12)
    assert np.allclose(sp.from_xyz(x), x, atol=1e-12)

    x = _sample_unit(3, 200, seed=7)
    assert np.allclose(sp.to_xyz(x), x, atol=1e-12)
    assert np.allclose(sp.from_xyz(x), x, atol=1e-12)


def test_xyz_d50_white_is_d50():
    sp = XYZD50Space()
    assert np.allclose(sp.to_xyz(D50), D50, atol=1e-12)


def test_xyz_d50_write_tier():
    sp = XYZD50Space()
    assert sp.to_icc_profile(version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile(version=5)["method"] == "exact_mpet"


def test_srgb_linear_through_to():
    lin, srgb = SRGBLinearSpace(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    v = srgb.to("srgb-linear", pixel)
    back = lin.to(srgb, v)
    assert np.allclose(pixel, back, atol=1e-9)


def test_xyz_d50_through_to():
    xyzd50, srgb = XYZD50Space(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    v = srgb.to("xyz-d50", pixel)
    back = xyzd50.to(srgb, v)
    assert np.allclose(pixel, back, atol=1e-9)


@pytest.mark.parametrize(
    "cls",
    [
        SRGBLinearSpace,
        DisplayP3LinearSpace,
        A98RGBSpace,
        ProPhotoRGBSpace,
        Rec2020Space,
        XYZD65Space,
        XYZD50Space,
    ],
)
def test_icc_roundtrip_via_written_profile(tmp_path, cls):
    sp = cls()
    path = str(tmp_path / f"{sp.name}.icc")
    sp.to_icc_profile(path, version=4)
    from lcms2.icc import ICCSpace

    reloaded = ICCSpace(path)
    x = _sample_unit(sp.n_channels, 50, seed=42)
    expected_xyz = sp.to_xyz(x)
    got_xyz = reloaded.to_xyz(x)
    assert np.allclose(got_xyz, expected_xyz, atol=1e-4)
