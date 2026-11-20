import warnings

import numpy as np
import pytest

import lcms2
from lcms2 import ToneCurve


def test_convert_accepts_css_string():
    np.testing.assert_allclose(lcms2.convert("#ffffff", "sRGB", "Lab"), [100, 0, 0], atol=1e-2)


def test_convert_accepts_list_of_css_strings():
    out = lcms2.convert(["red", "rgb(0 0 255)"], "sRGB", "oklab")
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], [0.628, 0.225, 0.126], atol=1e-3)


def test_int_tuple_of_ones_warns():
    with pytest.warns(UserWarning, match="0..255"):
        lcms2.parse_color((1, 0, 0))


def test_int_tuple_255_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(lcms2.parse_color((255, 0, 0)), [1, 0, 0])


def test_tone_curve_then_composes():
    c = ToneCurve.gamma(2.0).then(ToneCurve.gamma(1.5))
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(c(x), x**3.0, atol=2e-3)


def test_tone_curve_then_rejects_non_curve():
    with pytest.raises(TypeError):
        ToneCurve.linear().then(object())
