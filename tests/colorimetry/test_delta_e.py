"""Color difference tests, including the Sharma, Wu & Dalal (2005) CIEDE2000 data set."""

import numpy as np
import pytest

from lcms2.colorimetry import delta_e

# L1 a1 b1 L2 a2 b2 dE00 -- Table 1 of Sharma, Wu and Dalal, "The CIEDE2000 color-difference
# formula: implementation notes, supplementary test data, and mathematical observations" (2005).
SHARMA = np.array(
    [
        [50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425],
        [50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615],
        [50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412],
        [50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000],
        [50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669],
        [50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195],
        [50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045],
        [50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461],
        [50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065],
        [50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
        [50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977],
        [50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030],
        [50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000],
        [50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000],
        [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
        [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630],
        [61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731],
        [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
        [22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373],
        [36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
        [90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441],
        [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
        [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
        [2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082],
    ]
)


def test_sharma_set_size():
    assert SHARMA.shape == (34, 7)


def test_sharma_vectorised():
    got = delta_e(SHARMA[:, 0:3], SHARMA[:, 3:6])
    np.testing.assert_allclose(got, SHARMA[:, 6], atol=1e-4)


def test_sharma_swapped_is_symmetric():
    got = delta_e(SHARMA[:, 3:6], SHARMA[:, 0:3])
    np.testing.assert_allclose(got, SHARMA[:, 6], atol=1e-4)


def test_delta_e76_is_euclidean():
    rng = np.random.default_rng(1)
    a = rng.uniform([0, -80, -80], [100, 80, 80], size=(50, 3))
    b = rng.uniform([0, -80, -80], [100, 80, 80], size=(50, 3))
    np.testing.assert_allclose(delta_e(a, b, method="76"), np.linalg.norm(a - b, axis=-1), rtol=1e-12)


@pytest.mark.parametrize("method", ["2000", "76", "94", "cmc", "bfd", "de2000", "CIE76"])
def test_identical_colors_give_zero(method):
    assert delta_e([50, 10, -10], [50, 10, -10], method=method) == pytest.approx(0.0, abs=1e-9)


def test_broadcasting_and_scalar_result():
    ref = np.array([50.0, 0.0, 0.0])
    many = np.zeros((4, 5, 3)) + [50, 3, 4]
    out = delta_e(many, ref, method="76")
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out, 5.0)
    assert np.ndim(delta_e(ref, ref)) == 0


def test_cmc_and_2000_parameters_change_result():
    a, b = [50, 0, 0], [60, 10, 10]
    assert delta_e(a, b, method="cmc", l=1, c=1) != delta_e(a, b, method="cmc", l=2, c=1)
    assert delta_e(a, b, Kl=2) < delta_e(a, b)


def test_inputs_not_modified():
    a = np.array([[50.0, 1, 2]])
    b = np.array([[52.0, 0, 0]])
    a0, b0 = a.copy(), b.copy()
    delta_e(a, b)
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)


def test_errors():
    with pytest.raises(ValueError, match="unknown delta E method"):
        delta_e([0, 0, 0], [0, 0, 0], method="nope")
    with pytest.raises(ValueError, match="last axis of length 3"):
        delta_e([0, 0], [0, 0, 0])
    with pytest.raises(TypeError, match="Kl"):
        delta_e([0, 0, 0], [0, 0, 0], method="2000", l=1)
    with pytest.raises(TypeError, match="takes no parameters"):
        delta_e([0, 0, 0], [0, 0, 0], method="76", Kl=1)
