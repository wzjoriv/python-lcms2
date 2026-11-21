"""
Defaults and name handling users rely on: intent, space names, device links, raw tags.
"""

import numpy as np
import pytest

import lcms2
from lcms2 import Intent, Profile, Transform, spaces


def test_default_intent_is_relative_colorimetric():
    assert Transform("sRGB", "Lab").intent == Intent.RELATIVE_COLORIMETRIC
    cmyk = spaces.cmyk.profile
    rgb = np.array([[0.2, 0.5, 0.8]])
    np.testing.assert_array_equal(
        Profile("sRGB").transform(rgb, cmyk), Profile("sRGB").transform(rgb, cmyk, intent="relative")
    )


@pytest.mark.parametrize("name", ["srgb_linear", "SRGB-Linear", "display_p3"])
def test_space_names_accept_underscores(name):
    assert spaces.get(name) is spaces.get(name.lower().replace("_", "-"))
    assert lcms2.convert([1.0, 0.5, 0.0], "sRGB", name).shape == (3,)


def test_transform_from_a_single_device_link():
    link = Transform("sRGB", "Display P3").to_device_link()
    x = np.array([[10, 120, 240]], np.uint8)
    np.testing.assert_array_equal(Transform(link).apply(x), Transform([link]).apply(x))
    with pytest.raises(TypeError, match="device link"):
        Transform(Profile("sRGB"))


def test_raw_tag_write_then_cooked_and_raw_reads():
    p = Profile("sRGB")
    raw = p.read_tag_raw("cprt")
    p.write_tag_raw("cprt", raw)
    assert p.copyright == "No copyright, use freely"
    assert p.read_tag_raw("cprt") == raw


def test_repr_hides_the_internal_nocache_flag():
    assert "flags" not in repr(Transform("sRGB", "Lab"))
    assert "BLACKPOINTCOMPENSATION" in repr(Transform("sRGB", "Lab", bpc=True))
