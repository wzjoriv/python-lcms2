"""lcms2.colorimetry is the one place the white points, Lab and Bradford maths live."""

import importlib

import numpy as np
import pytest

from lcms2 import colorimetry
from lcms2.spaces import SRGBSpace


def test_spaces_has_no_colorimetry_module_of_its_own():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lcms2.spaces._colorimetry")


def test_spaces_use_the_same_white_points():
    from lcms2.spaces import _gray, _srgb

    assert _srgb.D65 is colorimetry.D65
    assert _srgb.D50 is colorimetry.D50
    assert _gray.D50 is colorimetry.D50


def test_adaptation_matrix_matches_adapt_and_srgb_matrix():
    m = colorimetry.adaptation_matrix(colorimetry.D65, colorimetry.D50)
    assert np.allclose(m @ colorimetry.D65, colorimetry.D50)
    assert np.allclose(colorimetry.adapt(np.eye(3), colorimetry.D65, colorimetry.D50), m.T)
    # white of sRGB lands on D50 through the same matrix
    assert np.allclose(SRGBSpace().to_xyz([1.0, 1.0, 1.0]), colorimetry.D50, atol=1e-6)
    with pytest.raises(ValueError, match="unknown adaptation method"):
        colorimetry.adaptation_matrix(colorimetry.D65, colorimetry.D50, "nope")


def test_lab_round_trip_through_the_single_implementation():
    rng = np.random.default_rng(0)
    xyz = rng.random((500, 3)) * np.array([0.95, 1.0, 0.8])
    assert np.allclose(colorimetry.lab_to_xyz(colorimetry.xyz_to_lab(xyz)), xyz, atol=1e-12)
    assert np.allclose(colorimetry.xyz_to_lab(colorimetry.D50), [100.0, 0.0, 0.0], atol=1e-9)


def test_pcs_encodings_are_private_to_icc():
    assert not hasattr(colorimetry, "encode_lab_pcs")
    from lcms2.icc import _pcs

    lab = np.array([[50.0, 20.0, -30.0]])
    assert np.allclose(_pcs.decode_lab_pcs(_pcs.encode_lab_pcs(lab)), lab)
    assert np.allclose(_pcs.decode_lab_pcs(_pcs.encode_lab_pcs(lab, True), True), lab)
