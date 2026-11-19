"""BT.2100 linear, PQ and HLG spaces."""

import numpy as np

from lcms2.spaces import (
    ColorSpace,
    Rec2020Space,
    Rec2100HLGSpace,
    Rec2100LinearSpace,
    Rec2100PQSpace,
)
from lcms2.spaces._rec2100_hlg import HLG_A, HLG_B, HLG_C


def _sample_unit(n_channels, n, seed=0, lo=0.0, hi=1.0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(lo, hi, size=n) for _ in range(n_channels)], axis=-1)


def test_registered_names():
    names = set(ColorSpace.list())
    for n in ("rec2100-linear", "rec2100-pq", "rec2100-hlg"):
        assert n in names


def test_rec2100_linear_roundtrip_single_and_batch():
    sp = Rec2100LinearSpace()
    x = np.array([0.5, 0.25, 0.75])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)

    x = _sample_unit(3, 200, seed=1)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-9)


def test_rec2100_linear_matches_rec2020_linear_stage():
    lin, rec2020 = Rec2100LinearSpace(), Rec2020Space()
    assert np.allclose(lin.to_xyz([1.0, 1.0, 1.0]), rec2020.to_xyz([1.0, 1.0, 1.0]), atol=1e-9)
    assert np.allclose(lin.to_xyz([0.0, 0.0, 0.0]), rec2020.to_xyz([0.0, 0.0, 0.0]), atol=1e-9)


def test_rec2100_linear_write_tier():
    sp = Rec2100LinearSpace()
    assert (
        sp.to_icc_profile("/tmp/rec2100_linear_v2.icc", version=2)["method"] == "exact_matrix_trc"
    )
    assert (
        sp.to_icc_profile("/tmp/rec2100_linear_v4.icc", version=4)["method"] == "exact_matrix_trc"
    )
    assert sp.to_icc_profile("/tmp/rec2100_linear_v5.icc", version=5)["method"] == "exact_mpet"


def test_rec2100_pq_roundtrip_single_and_batch():
    sp = Rec2100PQSpace()
    x = np.array([0.8, 0.2, 0.4])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-8)

    x = _sample_unit(3, 300, seed=2)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-7)


def test_rec2100pq_decode_endpoints():
    from lcms2.spaces._rec2100_pq import pq_decode, pq_encode

    assert np.isclose(pq_decode(np.array(0.0)), 0.0, atol=1e-12)
    assert np.isclose(pq_decode(np.array(1.0)), 1.0, atol=1e-9)
    assert np.isclose(pq_encode(np.array(0.0)), 0.0, atol=1e-5)
    assert np.isclose(pq_encode(np.array(1.0)), 1.0, atol=1e-9)


def test_rec2100_pq_white_matches_relative_pcs_white():
    pq = Rec2100PQSpace()
    rec2020 = Rec2020Space()
    xyz_pq = pq.to_xyz([1.0, 1.0, 1.0])
    xyz_2020 = rec2020.to_xyz([1.0, 1.0, 1.0])
    assert np.allclose(xyz_pq, xyz_2020, atol=1e-6)


def test_rec2100_pq_write_tier():
    sp = Rec2100PQSpace()
    assert sp.to_icc_profile("/tmp/rec2100_pq_v2.icc", version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile("/tmp/rec2100_pq_v4.icc", version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile("/tmp/rec2100_pq_v5.icc", version=5)["method"] == "exact_mpet"


def test_rec2100_hlg_roundtrip_single_and_batch():
    sp = Rec2100HLGSpace()
    x = np.array([0.8, 0.2, 0.4])
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-8)

    x = _sample_unit(3, 300, seed=3)
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-7)


def test_rec2100_hlg_roundtrip_spans_both_oetf_branches():
    sp = Rec2100HLGSpace()
    x = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.5, 0.5, 0.5],
            [0.99, 0.99, 0.99],
        ]
    )
    assert np.allclose(sp.from_xyz(sp.to_xyz(x)), x, atol=1e-7)


def test_rec2100hlg_encode_boundary_continuity():
    L = 1.0 / 12.0
    branch_low = np.sqrt(3.0 * L)
    branch_high = HLG_A * np.log(12.0 * L - HLG_B) + HLG_C
    assert np.isclose(branch_low, branch_high, atol=1e-12)


def test_rec2100_hlg_inverse_oetf_boundary_continuity():
    E = 0.5
    branch_low = (E * E) / 3.0
    branch_high = (np.exp((E - HLG_C) / HLG_A) + HLG_B) / 12.0
    assert np.isclose(branch_low, branch_high, atol=1e-12)


def test_rec2100hlg_encode_matches_inverse_across_boundary():
    from lcms2.spaces._rec2100_hlg import hlg_decode, hlg_encode

    L = np.array([0.0, 1.0 / 24, 1.0 / 12, 1.0 / 6, 0.5, 1.0])
    E = hlg_encode(L)
    assert np.allclose(hlg_decode(E), L, atol=1e-9)


def test_rec2100_hlg_write_tier():
    sp = Rec2100HLGSpace()
    assert sp.to_icc_profile("/tmp/rec2100_hlg_v2.icc", version=2)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile("/tmp/rec2100_hlg_v4.icc", version=4)["method"] == "exact_matrix_trc"
    assert sp.to_icc_profile("/tmp/rec2100_hlg_v5.icc", version=5)["method"] == "exact_mpet"
