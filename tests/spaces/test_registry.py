"""Color space registration and lookup by name."""

import numpy as np
import pytest

import lcms2.spaces  # noqa: F401 - registers SRGBSpace/HSVSpace/etc.
from lcms2.spaces import ColorSpace, GraySpace, HSVSpace, ICCSpace, SRGBSpace


def test_bundled_spaces_are_registered_by_import_alone():
    names = ColorSpace.list()
    for expected in ("srgb", "hsv", "gray", "oklab", "loggray", "twochannelchroma"):
        assert expected in names


def test_list_is_sorted_list_of_str():
    names = ColorSpace.list()
    assert names == sorted(names)
    assert all(isinstance(n, str) for n in names)


def test_colorspace_base_and_iccspace_are_not_registered():
    names = ColorSpace.list()
    assert "colorspace" not in names
    assert "iccspace" not in names


def test_new_subclass_is_registered_automatically_no_manual_call():
    class MyNewSpace(ColorSpace):
        n_channels = 1

        def to_xyz(self, x):
            return np.repeat(x, 3, axis=-1)

        def from_xyz(self, xyz):
            return xyz[..., :1]

    assert "mynewspace" in ColorSpace.list()
    assert ColorSpace.list().count("mynewspace") == 1


def test_subclass_can_set_explicit_lowercase_name():
    assert "srgb" in ColorSpace.list()
    from lcms2.spaces._base import _REGISTRY

    assert _REGISTRY["srgb"] is SRGBSpace


def test_to_by_name_roundtrips_srgb_to_hsv_to_srgb():
    rgb = np.array([[0.8, 0.3, 0.1], [0.1, 0.9, 0.4]])
    srgb = SRGBSpace()
    hsv = srgb.to("hsv", rgb)
    back = ColorSpace(n_channels=3, to_xyz=HSVSpace().to_xyz, from_xyz=HSVSpace().from_xyz).to("srgb", hsv)
    assert np.allclose(rgb, back, atol=1e-6)


def test_to_with_instance_argument():
    rgb = np.array([0.5, 0.5, 0.5])
    srgb = SRGBSpace()
    gray_target = GraySpace(gamma=2.2)
    result = srgb.to(gray_target, rgb)
    assert result.shape == (1,)
    expected = gray_target.from_xyz(srgb.to_xyz(rgb))
    assert np.allclose(result, expected)


def test_to_unknown_name_raises_helpful_error():
    srgb = SRGBSpace()
    with pytest.raises(KeyError, match="not-a-real-space"):
        srgb.to("not-a-real-space", np.array([0.1, 0.2, 0.3]))


def test_to_class_needing_constructor_args_raises_helpful_type_error():
    class NeedsArgSpace(ColorSpace):
        n_channels = 1

        def __init__(self, required_arg, **kwargs):
            self.required_arg = required_arg
            super().__init__(**kwargs)

        def to_xyz(self, x):
            return np.repeat(x, 3, axis=-1)

        def from_xyz(self, xyz):
            return xyz[..., :1]

    srgb = SRGBSpace()
    with pytest.raises(TypeError, match="NeedsArgSpace"):
        srgb.to("needsargspace", np.array([0.5, 0.5, 0.5]))


def test_to_supports_batched_and_single_pixel_shapes():
    srgb = SRGBSpace()
    single = srgb.to("hsv", np.array([0.2, 0.4, 0.6]))
    assert single.shape == (3,)
    batch = srgb.to("hsv", np.array([[0.2, 0.4, 0.6], [0.1, 0.1, 0.1]]))
    assert batch.shape == (2, 3)


def test_registering_a_colliding_name_warns_but_does_not_raise():
    class OneOffSpace(ColorSpace):
        name = "my-colliding-name"
        n_channels = 1

        def to_xyz(self, x):
            return np.repeat(x, 3, axis=-1)

        def from_xyz(self, xyz):
            return xyz[..., :1]

    with pytest.warns(UserWarning, match="my-colliding-name"):

        class AnotherOneOffSpace(ColorSpace):
            name = "my-colliding-name"
            n_channels = 1

            def to_xyz(self, x):
                return np.repeat(x, 3, axis=-1)

            def from_xyz(self, xyz):
                return xyz[..., :1]

    from lcms2.spaces._base import _REGISTRY

    assert _REGISTRY["my-colliding-name"] is AnotherOneOffSpace


def test_iccspace_exposes_description_and_is_not_registered(tmp_path):
    space = SRGBSpace()
    path = str(tmp_path / "srgb.icc")
    space.to_icc_profile(path, description="My Test sRGB Profile")
    loaded = ICCSpace(path)
    assert loaded.description == "My Test sRGB Profile"
    assert loaded.name == "My Test sRGB Profile"
    assert "my test srgb profile" not in ColorSpace.list()
