"""Transform with format inference."""

import threading

import numpy as np
import pytest

import lcms2


def test_reference_values():
    t = lcms2.Transform("sRGB", "Lab", intent="relative")
    np.testing.assert_allclose(t([255, 255, 255]), [100, 0, 0], atol=1e-3)
    np.testing.assert_allclose(t([255, 0, 0]), [54.29, 80.80, 69.89], atol=0.02)


def test_int_into_lab_gives_float64_and_out_dtype():
    t = lcms2.Transform("sRGB", "Lab")
    assert t(np.zeros(3, np.uint8)).dtype == np.float64
    assert t(np.zeros(3, np.uint8), out_dtype=np.float32).dtype == np.float32
    assert t(np.zeros(3, np.float32)).dtype == np.float32


def test_float_lists_and_rgb_round_trip():
    fwd, back = lcms2.Transform("sRGB", "Lab"), lcms2.Transform("Lab", "sRGB")
    x = np.random.default_rng(0).uniform(0, 1, (10, 3))
    np.testing.assert_allclose(back(fwd(x)), x, atol=1e-3)
    assert fwd([[0.5, 0.5, 0.5]]).shape == (1, 3)


def test_alpha_channel_is_copied():
    t = lcms2.Transform("sRGB", "Lab")
    x = np.array([[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.75]])
    y = t(x)
    assert y.shape == (2, 4)
    np.testing.assert_allclose(y[:, 3], [0.25, 0.75])
    np.testing.assert_allclose(y[:, :3], t(x[:, :3]))
    rgba8 = np.array([[255, 0, 0, 7]], np.uint8)
    assert lcms2.Transform("sRGB", "Adobe RGB")(rgba8)[0, 3] == 7


def test_gray_shapes():
    gray = lcms2.Profile.gray("D50", 1.0)
    to_rgb = lcms2.Transform(gray, "sRGB")
    assert to_rgb(np.zeros((4, 5), np.uint8)).shape == (4, 5, 3)
    assert to_rgb(np.zeros((4, 5, 1), np.uint8)).shape == (4, 5, 3)
    to_gray = lcms2.Transform("sRGB", gray)
    assert to_gray(np.zeros((4, 5, 3), np.uint8)).shape == (4, 5)
    assert lcms2.Transform(gray, gray)(np.zeros((4, 1), np.uint8)).shape == (4, 1)
    assert lcms2.Transform(gray, gray)(np.zeros((4, 2), np.uint8)).shape == (4, 2)


def test_empty_and_scalar_gray():
    t = lcms2.Transform("sRGB", "Lab")
    assert t(np.zeros((0, 3), np.uint8)).shape == (0, 3)
    assert t(np.zeros((2, 0, 3), np.float32)).shape == (2, 0, 3)
    gray = lcms2.Profile.gray("D50", 1.0)
    assert lcms2.Transform(gray, "sRGB")(np.array(0.5)).shape == (3,)


def test_out_parameter():
    t = lcms2.Transform("sRGB", "Lab")
    x = np.zeros((3, 3), np.uint8)
    out = np.empty((3, 3))
    assert t.apply(x, out) is out
    with pytest.raises(ValueError, match="shape"):
        t.apply(x, np.empty((3, 4)))
    with pytest.raises(ValueError, match="dtype"):
        t.apply(x, np.empty((3, 3), np.float32))
    with pytest.raises(ValueError, match="contiguous"):
        t.apply(x, np.empty((3, 6))[:, ::2])


def test_bad_input_messages(cmyk):
    t = lcms2.Transform("sRGB", cmyk)
    with pytest.raises(ValueError, match=r"3 \(or 4 with alpha\), got shape \(2, 5\)"):
        t(np.zeros((2, 5), np.uint8))
    with pytest.raises(TypeError, match="uint8, uint16"):
        t(np.zeros((2, 3), np.int64))
    assert t(np.zeros((2, 3), np.uint8)).shape == (2, 4)


def test_cmyk_input_channels(cmyk):
    t = lcms2.Transform(cmyk, "sRGB")
    assert t(np.zeros((2, 4), np.uint8)).shape == (2, 3)
    assert t(np.zeros((2, 5), np.uint8)).shape == (2, 4)  # CMYK + alpha


def test_incompatible_profiles_raise_cmserror():
    lnk = lcms2.Profile.ink_limiting(250)
    t = lcms2.Transform("sRGB", lnk)
    with pytest.raises(lcms2.CMSError) as e:
        t(np.zeros(3, np.uint8))
    assert isinstance(e.value, ValueError)
    assert str(e.value)


def test_proofing_with_gamut_check_changes_results(cmyk):
    x = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.5]])
    plain = lcms2.Transform("sRGB", "sRGB")(x)
    proofed = lcms2.Transform("sRGB", "sRGB", proof=cmyk)(x)
    alarm = lcms2.Transform("sRGB", "sRGB", proof=cmyk, gamut_check=True)(x)
    assert not np.allclose(plain[0], proofed[0], atol=1e-3)
    assert not np.allclose(proofed[0], alarm[0], atol=1e-3)
    with pytest.raises(ValueError, match="proof"):
        lcms2.Transform("sRGB", "sRGB", gamut_check=True)


def test_multiprofile_equals_chained():
    abstract = lcms2.Profile.abstract(brightness=5, saturation=10)
    x = np.random.default_rng(3).uniform(0, 1, (20, 3))
    multi = lcms2.Transform(["sRGB", abstract, "Adobe RGB"], intent="relative")(x)
    lab = lcms2.Transform("sRGB", "Lab", intent="relative")(x)
    lab2 = lcms2.Transform(abstract, "Lab", intent="relative")(lab)
    chained = lcms2.Transform("Lab", "Adobe RGB", intent="relative")(lab2)
    np.testing.assert_allclose(multi, chained, atol=5e-3)


def test_extended_per_profile_options():
    t = lcms2.Transform(["sRGB", "Lab"], intent=["relative", "relative"], bpc=[False, False])
    np.testing.assert_allclose(t([1.0, 1.0, 1.0]), [100, 0, 0], atol=1e-2)
    with pytest.raises(ValueError, match="one value per profile"):
        lcms2.Transform(["sRGB", "Lab"], intent=["relative"])


def test_device_link_round_trip():
    t = lcms2.Transform("sRGB", "Adobe RGB")
    link = t.to_device_link()
    assert str(link.device_class) == "link"
    back = lcms2.Profile.from_bytes(link.to_bytes())
    x = np.random.default_rng(5).integers(0, 256, (50, 3), dtype=np.uint8)
    np.testing.assert_allclose(lcms2.Transform([back])(x).astype(int), t(x).astype(int), atol=6)


def test_threads_match_serial():
    t = lcms2.Transform("sRGB", "Lab")
    imgs = [np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8) for i in range(8)]
    serial = [t(im) for im in imgs]
    results = [None] * len(imgs)

    def work(i):
        for _ in range(5):
            results[i] = t(imgs[i])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(imgs))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for a, b in zip(results, serial, strict=True):
        np.testing.assert_array_equal(a, b)


def test_flags_and_repr():
    t = lcms2.Transform("sRGB", "Lab", bpc=True, optimize=False, precision="high", flags="NONEGATIVES")
    for f in ("BLACKPOINTCOMPENSATION", "NOOPTIMIZE", "HIGHRESPRECALC", "NONEGATIVES", "NOCACHE"):
        assert t.flags & lcms2.Flag[f]
    assert "sRGB built-in" in repr(t)
    with pytest.raises(ValueError, match="precision"):
        lcms2.Transform("sRGB", "Lab", precision="max")
    with pytest.raises(TypeError, match="Transform"):
        lcms2.Transform("sRGB")


def test_alarm_codes_and_adaptation():
    old = lcms2.get_alarm_codes()
    try:
        lcms2.set_alarm_codes([1, 2, 3])
        assert lcms2.get_alarm_codes()[:4] == (1, 2, 3, 0)
    finally:
        lcms2.set_alarm_codes(old)
    prev = lcms2.set_adaptation_state(0.5)
    assert lcms2.set_adaptation_state(prev) == 0.5
