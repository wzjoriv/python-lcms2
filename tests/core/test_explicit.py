"""The explicit format-string form of Transform (ported from the 0.1 tests)."""

import numpy as np
import pytest

import lcms2


def test_neutral_grey():
    t = lcms2.Transform(lcms2.Profile("Lab"), "Lab_DBL", lcms2.Profile("sRGB"), "RGB_16", "PERCEPTUAL", "NONE")
    np.testing.assert_allclose(t.apply([50, 0, 0]), [30561, 30561, 30561])


def test_rgb_to_cmyk(cmyk):
    t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_DBL", cmyk, "CMYK_DBL")
    assert t.apply([0.1, 0.2, 0.3]).shape == (4,)
    assert t.input_format == "RGB_DBL" and t.output_format == "CMYK_DBL"


def test_full_image_and_case_insensitive_formats():
    t = lcms2.Transform(lcms2.Profile("sRGB"), "rgb_8", lcms2.Profile("ProPhoto RGB"), "Rgb_16")
    img = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    out = t.apply(img)
    assert out.shape == (48, 64, 3) and out.dtype == np.uint16


def test_bgr_and_float32_lab():
    srgb, lab = lcms2.Profile("sRGB"), lcms2.Profile("Lab")
    to_lab = lcms2.Transform(srgb, "BGR_8", lab, "Lab_FLT")
    out = to_lab.apply(np.array([[0, 0, 255]], np.uint8))  # BGR red
    np.testing.assert_allclose(out[0], lcms2.convert([255, 0, 0], "sRGB", "Lab"), atol=0.1)
    assert out.dtype == np.float32


def test_dtype_mismatch_and_wrong_channels():
    t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("Lab"), "Lab_DBL")
    with pytest.raises(TypeError, match="astype"):
        t.apply(np.zeros(3, np.float64))
    with pytest.raises(ValueError, match="3 channels"):
        t.apply(np.zeros(4, np.uint8))


def test_sliced_input():
    t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("XYZ"), "XYZ_DBL")
    z = np.random.default_rng(4).integers(0, 256, (5, 3), dtype=np.uint8)
    np.testing.assert_allclose(t.apply(z[::2]), t.apply(z)[::2])


def test_planar_channel_first():
    t = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8_PLANAR", lcms2.Profile("sRGB"), "RGB_8_PLANAR")
    x = np.random.default_rng(2).integers(0, 256, (3, 4, 5), dtype=np.uint8)
    np.testing.assert_array_equal(t.apply(x), x)
    t2 = lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8_PLANAR", lcms2.Profile("sRGB"), "RGB_8")
    np.testing.assert_array_equal(t2.apply(x), np.moveaxis(x, 0, -1))


def test_unknown_intent_and_flag():
    with pytest.raises(ValueError, match="unknown rendering intent"):
        lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("Lab"), "Lab_DBL", "VIVID")
    with pytest.raises(ValueError, match="unknown flag"):
        lcms2.Transform(lcms2.Profile("sRGB"), "RGB_8", lcms2.Profile("Lab"), "Lab_DBL", flags="FAST")


def test_flags_string_form():
    t = lcms2.Transform(lcms2.Profile("Lab"), "Lab_DBL", lcms2.Profile("sRGB"), "RGB_16",
                        intent="PERCEPTUAL", flags="NOOPTIMIZE,BLACKPOINTCOMPENSATION")
    assert t.flags & lcms2.Flag.NOOPTIMIZE
    with pytest.raises(ValueError, match="out_dtype"):
        t.apply([50.0, 0, 0], out_dtype=np.uint8)
