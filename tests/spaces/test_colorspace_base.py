import numpy as np
import pytest

from lcms2.spaces import ColorSpace


def test_base_with_callables():
    cs = ColorSpace(n_channels=3, to_xyz=lambda x: x, from_xyz=lambda x: x)
    x = np.ones((2, 3))
    assert np.allclose(cs.to_xyz(x), x)
    assert np.allclose(cs.from_xyz(x), x)


def test_base_missing_callable_raises():
    with pytest.raises(TypeError):
        ColorSpace(n_channels=3, to_xyz=lambda x: x)


def test_base_missing_n_channels_raises():
    with pytest.raises(TypeError):
        ColorSpace(to_xyz=lambda x: x, from_xyz=lambda x: x)


def test_subclass_override_detected():
    class MySpace(ColorSpace):
        n_channels = 3

        def to_xyz(self, x):
            return x * 2

        def from_xyz(self, x):
            return x / 2

    cs = MySpace()
    x = np.ones((2, 3))
    assert np.allclose(cs.to_xyz(x), x * 2)
    assert np.allclose(cs.from_xyz(x), x / 2)


def test_subclass_override_plus_callable_is_ambiguous():
    class MySpace(ColorSpace):
        n_channels = 3

        def to_xyz(self, x):
            return x

    with pytest.raises(TypeError):
        MySpace(to_xyz=lambda x: x, from_xyz=lambda x: x)


def test_subclass_partial_override_plus_callable_for_other_direction():
    class MySpace(ColorSpace):
        n_channels = 3

        def to_xyz(self, x):
            return x * 3

    cs = MySpace(from_xyz=lambda x: x / 3)
    x = np.ones((2, 3))
    assert np.allclose(cs.to_xyz(x), x * 3)
    assert np.allclose(cs.from_xyz(x), x / 3)
