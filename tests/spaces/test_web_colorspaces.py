"""CSS Color 4 HWB, CMYK, LCh, OKLCh and Display P3 spaces."""

import os
import tempfile

import numpy as np
import pytest

from lcms2.icc import ICCSpace
from lcms2.spaces import (
    CMYKSpace,
    ColorSpace,
    DisplayP3Space,
    HWBSpace,
    LChSpace,
    OKLChSpace,
    SRGBSpace,
)


def _sample(cs, n, seed=0):
    rng = np.random.default_rng(seed)
    domain = cs.device_domain
    if isinstance(domain[0], (tuple, list)):
        cols = [rng.uniform(lo, hi, size=n) for lo, hi in domain]
    else:
        lo, hi = domain
        cols = [rng.uniform(lo, hi, size=n) for _ in range(cs.n_channels)]
    return np.stack(cols, axis=-1)


def test_registered_names():
    names = set(ColorSpace.list())
    for n in ("hwb", "cmyk", "lch", "oklch", "display-p3"):
        assert n in names


def test_hwb_roundtrip_single_and_batch():
    hwb = HWBSpace()
    x = np.array([0.37, 0.2, 0.3])
    back = hwb.from_xyz(hwb.to_xyz(x))
    assert np.allclose(x, back, atol=1e-6)

    rng = np.random.default_rng(1)
    h = rng.uniform(0, 1, size=64)
    w = rng.uniform(0, 0.5, size=64)
    b = rng.uniform(0, 0.5, size=64)
    x = np.stack([h, w, b], axis=-1)
    back = hwb.from_xyz(hwb.to_xyz(x))
    assert np.allclose(x, back, atol=1e-6)


def test_hwb_normalizes_when_over_saturated():
    hwb = HWBSpace()
    x = np.array([0.5, 0.7, 0.6])
    xyz = hwb.to_xyz(x)
    back = hwb.from_xyz(xyz)
    wb_sum = back[1] + back[2]
    assert wb_sum == pytest.approx(1.0, abs=1e-6)
    assert back[1] / back[2] == pytest.approx(x[1] / x[2], rel=1e-6)


def test_hwb_known_points():
    hwb = HWBSpace()
    srgb = SRGBSpace()
    white_xyz = srgb.to_xyz(np.array([1.0, 1.0, 1.0]))
    black_xyz = srgb.to_xyz(np.array([0.0, 0.0, 0.0]))
    red_xyz = srgb.to_xyz(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(hwb.from_xyz(white_xyz), [0.0, 1.0, 0.0], atol=1e-4)
    assert np.allclose(hwb.from_xyz(black_xyz), [0.0, 0.0, 1.0], atol=1e-4)
    h, w, b = hwb.from_xyz(red_xyz)
    assert w == pytest.approx(0.0, abs=1e-4)
    assert b == pytest.approx(0.0, abs=1e-4)
    assert h == pytest.approx(0.0, abs=1e-3)


def test_hwb_write_tier():
    hwb = HWBSpace()
    r4 = hwb.to_icc_profile(version=4)
    assert r4["method"] == "clut_fallback"
    r5 = hwb.to_icc_profile(version=5)
    assert r5["method"] == "exact_calc"


def test_hwb_through_to():
    hwb, srgb = HWBSpace(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    h = srgb.to("hwb", pixel)
    back = hwb.to(srgb, h)
    assert np.allclose(pixel, back, atol=1e-6)


def test_cmyk_rgb_roundtrip():
    cmyk = CMYKSpace()
    rgb = np.random.default_rng(2).uniform(0, 1, size=(50, 3))
    xyz = SRGBSpace().to_xyz(rgb)
    c = cmyk.from_xyz(xyz)
    back_xyz = cmyk.to_xyz(c)
    back_rgb = SRGBSpace().from_xyz(back_xyz)
    assert np.allclose(rgb, back_rgb, atol=1e-6)


def test_cmyk_known_points():
    cmyk = CMYKSpace()
    srgb = SRGBSpace()
    white_xyz = srgb.to_xyz(np.array([1.0, 1.0, 1.0]))
    black_xyz = srgb.to_xyz(np.array([0.0, 0.0, 0.0]))
    red_xyz = srgb.to_xyz(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(cmyk.from_xyz(white_xyz), [0, 0, 0, 0], atol=1e-2)
    assert np.allclose(cmyk.from_xyz(black_xyz), [0, 0, 0, 100], atol=1e-2)
    assert np.allclose(cmyk.from_xyz(red_xyz), [0, 100, 100, 0], atol=1e-2)


def test_cmyk_write_tier():
    cmyk = CMYKSpace()
    r4 = cmyk.to_icc_profile(version=4)
    assert r4["method"] == "clut_fallback"
    r5 = cmyk.to_icc_profile(version=5)
    assert r5["method"] == "exact_calc"


def test_cmyk_through_to():
    cmyk, srgb = CMYKSpace(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    c = srgb.to("cmyk", pixel)
    back = cmyk.to(srgb, c)
    assert np.allclose(pixel, back, atol=1e-6)


def test_lch_roundtrip_single_and_batch():
    lch = LChSpace()
    x = np.array([54.0, 30.0, 0.3])
    assert np.allclose(lch.from_xyz(lch.to_xyz(x)), x, atol=1e-6)

    x = _sample(lch, 100, seed=3)
    assert np.allclose(lch.from_xyz(lch.to_xyz(x)), x, atol=1e-6)


def test_lch_known_points():
    lch = LChSpace()
    srgb = SRGBSpace()
    white_xyz = srgb.to_xyz(np.array([1.0, 1.0, 1.0]))
    black_xyz = srgb.to_xyz(np.array([0.0, 0.0, 0.0]))
    red_xyz = srgb.to_xyz(np.array([1.0, 0.0, 0.0]))

    L, C, h = lch.from_xyz(white_xyz)
    assert L == pytest.approx(100.0, abs=1e-2)
    assert C == pytest.approx(0.0, abs=1e-2)

    L, C, h = lch.from_xyz(black_xyz)
    assert L == pytest.approx(0.0, abs=1e-6)
    assert C == pytest.approx(0.0, abs=1e-6)

    L, C, h = lch.from_xyz(red_xyz)
    assert L == pytest.approx(54.3, abs=1.0)
    assert C == pytest.approx(106.8, abs=2.0)
    assert h == pytest.approx(40.8, abs=2.0)


def test_lch_write_tier():
    lch = LChSpace()
    for v in (2, 4):
        r = lch.to_icc_profile(version=v)
        assert r["method"] == "clut_fallback", (v, r["method"])
    r5 = lch.to_icc_profile(version=5)
    assert r5["method"] == "exact_calc", r5["method"]


def test_lch_exact_calc_roundtrip():
    lch = LChSpace()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lch.icc")
        r = lch.to_icc_profile(path, version=5)
        assert r["method"] == "exact_calc"
        reloaded = ICCSpace(path)
        x = _sample(lch, 200, seed=11)
        expected_xyz = lch.to_xyz(x)
        got_xyz = reloaded.to_xyz(x)
        assert np.allclose(got_xyz, expected_xyz, atol=1e-4)

        y = lch.from_xyz(expected_xyz)
        got_y = reloaded.from_xyz(expected_xyz)
        assert np.allclose(got_y, y, atol=1e-4)


def test_lch_through_to():
    lch, srgb = LChSpace(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    v = srgb.to("lch", pixel)
    back = lch.to(srgb, v)
    assert np.allclose(pixel, back, atol=1e-4)


def test_oklch_roundtrip_single_and_batch():
    oklch = OKLChSpace()
    x = np.array([0.6, 0.1, 0.25])
    assert np.allclose(oklch.from_xyz(oklch.to_xyz(x)), x, atol=1e-6)

    x = _sample(oklch, 100, seed=4)
    assert np.allclose(oklch.from_xyz(oklch.to_xyz(x)), x, atol=1e-6)


def test_oklch_known_points():
    oklch = OKLChSpace()
    srgb = SRGBSpace()
    white_xyz = srgb.to_xyz(np.array([1.0, 1.0, 1.0]))
    black_xyz = srgb.to_xyz(np.array([0.0, 0.0, 0.0]))

    L, C, h = oklch.from_xyz(black_xyz)
    assert L == pytest.approx(0.0, abs=1e-6)
    assert C == pytest.approx(0.0, abs=1e-6)

    L, C, h = oklch.from_xyz(white_xyz)
    assert L == pytest.approx(1.0, abs=0.05)


def test_oklch_write_tier():
    oklch = OKLChSpace()
    for v in (2, 4):
        r = oklch.to_icc_profile(version=v)
        assert r["method"] == "clut_fallback", (v, r["method"])
    r5 = oklch.to_icc_profile(version=5)
    assert r5["method"] == "exact_calc", r5["method"]


def test_oklch_exact_calc_roundtrip():
    oklch = OKLChSpace()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "oklch.icc")
        r = oklch.to_icc_profile(path, version=5)
        assert r["method"] == "exact_calc"
        reloaded = ICCSpace(path)
        x = _sample(oklch, 200, seed=12)
        expected_xyz = oklch.to_xyz(x)
        got_xyz = reloaded.to_xyz(x)
        assert np.allclose(got_xyz, expected_xyz, atol=1e-4)

        y = oklch.from_xyz(expected_xyz)
        got_y = reloaded.from_xyz(expected_xyz)
        d = np.abs(got_y - y)
        d[:, 2] = np.minimum(d[:, 2], 360.0 - d[:, 2]) / 360.0
        assert d.max() <= 1e-4


def test_oklch_through_to():
    oklch, srgb = OKLChSpace(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    v = srgb.to("oklch", pixel)
    back = oklch.to(srgb, v)
    assert np.allclose(pixel, back, atol=1e-4)


def test_display_p3_roundtrip_single_and_batch():
    p3 = DisplayP3Space()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(p3.from_xyz(p3.to_xyz(x)), x, atol=1e-9)

    x = _sample(p3, 200, seed=5)
    assert np.allclose(p3.from_xyz(p3.to_xyz(x)), x, atol=1e-9)


def test_display_p3_known_points():
    p3 = DisplayP3Space()
    white_xyz = p3.to_xyz(np.array([1.0, 1.0, 1.0]))
    black_xyz = p3.to_xyz(np.array([0.0, 0.0, 0.0]))

    srgb_white_xyz = SRGBSpace().to_xyz(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(white_xyz, srgb_white_xyz, atol=1e-3)
    assert np.allclose(black_xyz, [0.0, 0.0, 0.0], atol=1e-9)


def test_display_p3_wider_gamut_than_srgb():
    srgb = SRGBSpace()
    p3 = DisplayP3Space()
    red_xyz = srgb.to_xyz(np.array([1.0, 0.0, 0.0]))
    p3_rgb = p3.from_xyz(red_xyz)
    assert np.all(p3_rgb >= -1e-6) and np.all(p3_rgb <= 1.0 + 1e-6)


def test_display_p3_write_tier():
    p3 = DisplayP3Space()
    r2 = p3.to_icc_profile(version=2)
    assert r2["method"] == "exact_matrix_trc"
    r4 = p3.to_icc_profile(version=4)
    assert r4["method"] == "exact_matrix_trc"
    r5 = p3.to_icc_profile(version=5)
    assert r5["method"] == "exact_mpet"


def test_display_p3_through_to():
    p3, srgb = DisplayP3Space(), SRGBSpace()
    pixel = [0.8, 0.2, 0.4]
    v = srgb.to("display-p3", pixel)
    back = p3.to(srgb, v)
    assert np.allclose(pixel, back, atol=1e-9)
