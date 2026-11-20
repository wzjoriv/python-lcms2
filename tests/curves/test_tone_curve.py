"""ToneCurve construction, evaluation and analysis."""

import numpy as np
import pytest

from lcms2 import ToneCurve

X = np.linspace(0, 1, 101)


def test_gamma_values():
    g = ToneCurve.gamma(2.2)
    np.testing.assert_allclose(g(X), X**2.2, atol=1e-6)
    assert g.parametric_type == 1
    assert g.estimate_gamma() == pytest.approx(2.2, abs=0.01)


def test_shapes_and_dtypes():
    g = ToneCurve.gamma(2.0)
    assert g(0.5) == pytest.approx(0.25, abs=1e-6)
    assert g(np.zeros((3, 4, 5))).shape == (3, 4, 5)
    assert g(np.zeros((0, 3))).shape == (0, 3)
    assert g(np.float32([0.5])).dtype == np.float64
    u = g(np.array([0, 32768, 65535], dtype=np.uint16))
    assert u.dtype == np.uint16
    assert u[0] == 0 and u[2] == 65535 and abs(int(u[1]) - 16384) <= 2
    non_contig = np.linspace(0, 1, 20)[::2]
    np.testing.assert_allclose(g(non_contig), non_contig**2, atol=1e-6)
    with pytest.raises(TypeError, match="uint16"):
        g(np.array([1, 2, 3]))


def test_large_array_matches_formula():
    g = ToneCurve.gamma(1.8)
    big = np.random.default_rng(0).random((300, 300))
    np.testing.assert_allclose(g(big), big**1.8, atol=1e-6)


def test_parametric_srgb_matches_preset():
    p = ToneCurve.parametric(4, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
    np.testing.assert_array_equal(p(X), ToneCurve.srgb()(X))
    ref = np.where(X <= 0.04045, X / 12.92, ((X + 0.055) / 1.055) ** 2.4)
    np.testing.assert_allclose(ToneCurve.srgb()(X), ref, atol=1e-6)


def test_rec709_and_linear():
    r = ToneCurve.rec709()
    assert r(1.0) == pytest.approx(1.0, abs=1e-6)
    assert r(0.0) == pytest.approx(0.0, abs=1e-6)
    lin = ToneCurve.linear()
    assert lin.is_linear
    np.testing.assert_allclose(lin(X), X, atol=1e-7)
    assert not ToneCurve.gamma(2.2).is_linear


@pytest.mark.parametrize(
    "kind,count", [(1, 1), (2, 3), (3, 4), (4, 5), (5, 7), (6, 4), (7, 5), (8, 5), (108, 1), (109, 1)]
)
def test_parametric_param_counts(kind, count):
    with pytest.raises(ValueError, match=f"takes {count} parameter"):
        ToneCurve.parametric(kind, [1.0] * (count + 1))
    with pytest.raises(ValueError, match=f"takes {count} parameter"):
        ToneCurve.parametric(-kind, [1.0] * (count + 1))
    with pytest.raises(ValueError, match=f"takes {count} parameter"):
        ToneCurve.parametric(kind, [1.0] * (count - 1) if count > 1 else [])


def test_parametric_unknown_type():
    with pytest.raises(ValueError, match="unsupported parametric curve type 9"):
        ToneCurve.parametric(9, [1.0])


def test_parametric_inverse_type():
    c = ToneCurve.parametric(-1, [2.0])
    assert c.parametric_type == -1
    np.testing.assert_allclose(c(X), X**0.5, atol=1e-6)


def test_tabulated_float_and_uint16():
    c = ToneCurve.tabulated(np.array([0.0, 0.25, 1.0]))
    assert c(0.5) == pytest.approx(0.25, abs=1e-6)
    t16 = (np.linspace(0, 1, 256) ** 2 * 65535).astype(np.uint16)
    c16 = ToneCurve.tabulated(t16)
    np.testing.assert_allclose(c16(X), X**2, atol=2e-4)
    assert c16.parametric_type == 0
    assert ToneCurve(t16).table.dtype == np.uint16


def test_tabulated_errors():
    with pytest.raises(ValueError, match="between 2 and 65535"):
        ToneCurve.tabulated(np.array([0.5]))
    with pytest.raises(TypeError, match="uint16"):
        ToneCurve.tabulated(np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="1-D"):
        ToneCurve.tabulated(np.zeros((2, 2)))


def test_inverse_round_trip():
    for c in (ToneCurve.gamma(2.2), ToneCurve.srgb()):
        np.testing.assert_allclose(c.inverse()(c(X)), X, atol=2e-3)
    np.testing.assert_allclose(ToneCurve.srgb().inverse(8192)(ToneCurve.srgb()(X)), X, atol=1e-3)


def test_join():
    g = ToneCurve.gamma(2.2)
    np.testing.assert_allclose(g.join(g)(X), X, atol=2e-3)
    np.testing.assert_allclose(g.join(ToneCurve.linear())(X), X**2.2, atol=2e-3)


def test_smooth_and_copy():
    values = np.linspace(0, 1, 64) ** 2.2 * 65535
    values[20] += 300
    values[40] -= 200
    c = ToneCurve.tabulated(values.astype(np.uint16))
    s = c.smooth(2.0)
    assert s.is_monotonic
    assert s is not c
    dup = c.copy()
    assert dup is not c
    np.testing.assert_array_equal(dup.table, c.table)
    np.testing.assert_array_equal(c.table, ToneCurve.tabulated(values.astype(np.uint16)).table)


def test_flags():
    g = ToneCurve.gamma(2.2)
    assert g.is_monotonic and not g.is_descending and not g.is_multisegment
    desc = ToneCurve.tabulated(np.linspace(1, 0, 16))
    assert desc.is_descending and desc.is_monotonic
    assert isinstance(ToneCurve.srgb().is_multisegment, bool)
    assert len(g.table) == 4096


def test_repr_and_construction():
    assert "parametric type 1" in repr(ToneCurve.gamma(2.2))
    assert "tabulated" in repr(ToneCurve([0.0, 0.5, 1.0]))
    assert ToneCurve(2.2).parametric_type == 1


def test_bad_handle_raises_type_error():
    with pytest.raises(TypeError):
        ToneCurve.linear().join(object())
