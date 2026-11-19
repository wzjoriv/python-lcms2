import numpy as np
import pytest

from lcms2.colormap import Colormap
from lcms2.colormap._cvd_data import MATRICES
from lcms2.colormap._diagnostics import cvd_matrix


@pytest.fixture
def cm():
    return Colormap.from_colors(["#001040", "#f0a020", "#ffffe0"], n=32)


def test_lightness_and_deltas(cm):
    L = cm.lightness()
    assert L.shape == (32,) and L[0] < 0.3 < 0.9 < L[-1]
    d = cm.perceptual_deltas()
    assert d.shape == (31,) and np.all(d > 0)
    assert cm.uniformity() == pytest.approx(d.std() / d.mean())
    assert Colormap([[0, 0, 0], [1, 1, 1]]).uniformity() == 0.0


def test_monotonic_detection(cm):
    assert cm.is_lightness_monotonic()
    assert cm.reversed().is_lightness_monotonic()
    assert not Colormap.from_colors(["black", "white", "black"]).is_lightness_monotonic()


def test_cvd_severity_zero_is_identity(cm):
    for d in ("protan", "deutan", "tritan"):
        assert cm.simulate(d, 0.0) == cm
        assert np.allclose(cvd_matrix(d, 0.0), np.eye(3))


def test_machado_rows_sum_to_one():
    for key, mats in MATRICES.items():
        assert mats.shape == (11, 3, 3)
        assert np.allclose(mats.sum(axis=2), 1.0, atol=2e-6), key
        assert np.array_equal(mats[0], np.eye(3))
    assert np.allclose(MATRICES["deutan"][10][0], [0.367322, 0.860646, -0.227968])


def test_cvd_interpolates_between_tabulated_severities():
    mid = cvd_matrix("protan", 0.25)
    assert np.allclose(mid, (MATRICES["protan"][2] + MATRICES["protan"][3]) / 2)
    assert np.allclose(cvd_matrix("tritanopia", 1.0), MATRICES["tritan"][10])


def test_cvd_greys_unchanged_and_reds_greens_converge():
    grey = Colormap.from_colors(["#202020", "#e0e0e0"], n=5)
    assert np.allclose(grey.simulate("deutan").colors, grey.colors, atol=2e-3)
    rg = Colormap([[1, 0, 0], [0, 0.6, 0]])
    sim = rg.simulate("deutan")
    before = np.linalg.norm(rg.colors[0] - rg.colors[1])
    assert np.linalg.norm(sim.colors[0] - sim.colors[1]) < before * 0.7
    assert np.all((sim.colors >= 0) & (sim.colors <= 1))


def test_cvd_errors(cm):
    with pytest.raises(ValueError, match="deficiency"):
        cm.simulate("purple")
    with pytest.raises(ValueError, match="severity"):
        cm.simulate("protan", 1.5)
    assert Colormap.get("polar").simulate("deutan").name.startswith("polar")
