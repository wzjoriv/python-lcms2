"""Round trips and ICC writing of the built-in spaces."""

import numpy as np
import pytest

from lcms2.icc import ICCParseError, ICCSpace
from lcms2.spaces import (
    ColorSpace,
    GraySpace,
    HSVSpace,
    LogGraySpace,
    OKLabSpace,
    SRGBSpace,
)

from .toys import TwoChannelChromaSpace


def _sample(cs, n, seed=0):
    rng = np.random.default_rng(seed)
    domain = cs.device_domain
    if isinstance(domain[0], (tuple, list)):
        cols = [rng.uniform(lo, hi, size=n) for lo, hi in domain]
    else:
        lo, hi = domain
        cols = [rng.uniform(lo, hi, size=n) for _ in range(cs.n_channels)]
    return np.stack(cols, axis=-1)


def test_self_roundtrip_all_spaces():
    for cs, _atol in [
        (SRGBSpace(), 1e-9),
        (GraySpace(2.2), 1e-9),
        (OKLabSpace(), 1e-9),
        (HSVSpace(), 1e-9),
        (LogGraySpace(), 1e-9),
    ]:
        x = _sample(cs, 200)
        back = cs.from_xyz(cs.to_xyz(x))
        assert np.allclose(x, back, atol=1e-6), type(cs).__name__


@pytest.mark.parametrize(
    "version,expect_method",
    [
        ((4, 4, 0), "exact_matrix_trc"),
        ((2, 4, 0), "exact_matrix_trc"),
        ((5, 0, 0), "exact_mpet"),
    ],
)
def test_srgb_is_exact_never_clut(version, expect_method):
    cs = SRGBSpace()
    r = cs.to_icc_profile(version=version)
    assert r["method"] == expect_method, r["notes"]


@pytest.mark.parametrize(
    "version,expect_method",
    [
        ((4, 2, 0), "exact_gray_trc"),
        ((5, 0, 0), "exact_mpet"),
    ],
)
def test_gray_is_exact_never_clut(version, expect_method):
    cs = GraySpace(2.2)
    r = cs.to_icc_profile(version=version)
    assert r["method"] == expect_method, r["notes"]


def test_oklab_is_exact_in_v5():
    cs = OKLabSpace()
    r = cs.to_icc_profile(version=(5, 0, 0))
    assert r["method"] == "exact_mpet", r["notes"]


def test_oklab_is_exact_lut_in_v4_and_clut_in_v2(tmp_path):
    cs = OKLabSpace()
    assert cs.to_icc_profile(version=2)["method"] == "clut_fallback"
    path = str(tmp_path / "oklab.icc")
    assert cs.to_icc_profile(path, version=4)["method"] == "exact_lut"
    from lcms2.spaces import srgb

    xyz = srgb.to_xyz(np.random.default_rng(0).random((500, 3)))
    reloaded = ICCSpace(path)
    assert np.max(np.abs(reloaded.to_xyz(cs.from_xyz(xyz)) - xyz)) < 5e-4
    assert np.max(np.abs(reloaded.from_xyz(xyz) - cs.from_xyz(xyz))) < 5e-4


def test_hsv_is_clut_in_v4_but_exact_calc_in_v5():
    cs = HSVSpace()
    r_v4 = cs.to_icc_profile(version=(4, 4, 0))
    assert r_v4["method"] == "clut_fallback"
    r_v5 = cs.to_icc_profile(version=(5, 0, 0))
    assert r_v5["method"] == "exact_calc", r_v5["notes"]


@pytest.mark.parametrize(
    "cs,version,atol",
    [
        (SRGBSpace(), (4, 4, 0), 2e-3),
        (SRGBSpace(), (5, 0, 0), 1e-3),
        (GraySpace(2.2), (4, 2, 0), 2e-3),
        (GraySpace(2.2), (5, 0, 0), 1e-4),
        (LogGraySpace(), (5, 0, 0), 5e-3),
        (OKLabSpace(), (5, 0, 0), 1e-3),
    ],
)
def test_exact_paths_reload_within_tight_tolerance(cs, version, atol, tmp_path):
    path = str(tmp_path / "space.icc")
    r = cs.to_icc_profile(path, version=version)
    assert r["method"].startswith("exact")
    reloaded = ICCSpace(path)
    x = _sample(cs, 300, seed=1)
    xyz1, xyz2 = cs.to_xyz(x), reloaded.to_xyz(x)
    assert np.max(np.abs(xyz1 - xyz2)) < atol
    b1, b2 = cs.from_xyz(xyz1), reloaded.from_xyz(xyz1)
    assert np.max(np.abs(b1 - b2)) < atol


def test_hsv_clut_forward_direction_is_accurate(tmp_path):
    cs = HSVSpace()
    path = str(tmp_path / "hsv.icc")
    r = cs.to_icc_profile(path, version=(4, 4, 0), grid_size=33)
    assert r["method"] == "clut_fallback"
    reloaded = ICCSpace(path)
    x = _sample(cs, 400, seed=2)
    xyz1, xyz2 = cs.to_xyz(x), reloaded.to_xyz(x)
    assert np.max(np.abs(xyz1 - xyz2)) < 0.05


def test_hsv_inverse_is_reasonable_away_from_singularities(tmp_path):
    cs = HSVSpace()
    path = str(tmp_path / "hsv.icc")
    cs.to_icc_profile(path, version=(4, 4, 0), grid_size=33)
    reloaded = ICCSpace(path)
    rng = np.random.default_rng(2)
    H = rng.uniform(0, 1, 800)
    S = rng.uniform(0.5, 0.9, 800)
    V = rng.uniform(0.5, 0.9, 800)
    x = np.stack([H, S, V], axis=-1)
    xyz = cs.to_xyz(x)
    back, back2 = cs.from_xyz(xyz), reloaded.from_xyz(xyz)
    d = np.abs(back - back2)
    d[:, 0] = np.minimum(d[:, 0], 1 - d[:, 0])
    assert np.median(d) < 0.02, "typical-case Hue/Sat recovery should still be tight"


def test_gray_and_loggray_are_exact():
    gray = GraySpace(2.2)
    assert gray.to_icc_profile(version=(4, 2, 0))["method"] == "exact_gray_trc"
    assert gray.to_icc_profile(version=(5, 0, 0))["method"] == "exact_mpet"

    loggray = LogGraySpace()
    assert loggray.to_icc_profile(version=(5, 0, 0))["method"] == "exact_mpet"


def test_arbitrary_channel_reordering_traces_exactly_not_just_diagonal():

    class ReorderedSpace(ColorSpace):
        n_channels = 3

        def to_xyz(self, d):
            d = np.asarray(d, dtype=np.float64)
            a = np.power(d[..., 2], 1.0)
            b = np.power(d[..., 1], 1.0)
            c = np.power(d[..., 0], 1.0)
            return np.stack([a, b, c], axis=-1) @ np.eye(3)

        def from_xyz(self, xyz):
            xyz = np.asarray(xyz, dtype=np.float64)
            a = xyz[..., 1] * 0.5
            b = xyz[..., 1]
            c = xyz[..., 0]
            return np.stack([a, b, c], axis=-1)

    cs = ReorderedSpace()
    r = cs.to_icc_profile(version=(5, 0, 0))
    assert r["method"] == "exact_mpet", r["notes"]

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reordered.icc")
        cs.to_icc_profile(path, version=(5, 0, 0))
        reloaded = ICCSpace(path)
        x = _sample(cs, 200, seed=7)
        xyz1, xyz2 = cs.to_xyz(x), reloaded.to_xyz(x)
        assert np.max(np.abs(xyz1 - xyz2)) < 1e-4


def test_two_channel_space_self_roundtrip():
    cs = TwoChannelChromaSpace()
    x = _sample(cs, 200, seed=3)
    back = cs.from_xyz(cs.to_xyz(x))
    assert np.allclose(x, back, atol=1e-9)


@pytest.mark.parametrize("version", [2, 4])
def test_two_channel_space_is_exact_lut_in_v2_v4(tmp_path, version):
    cs = TwoChannelChromaSpace()
    path = str(tmp_path / "twochan.icc")
    r = cs.to_icc_profile(path, version=version)
    assert r["method"] == "exact_lut", r["notes"]
    reloaded = ICCSpace(path)
    x = _sample(cs, 100, seed=4)
    assert np.max(np.abs(cs.to_xyz(x) - reloaded.to_xyz(x))) < 1e-4
    assert np.max(np.abs(x - reloaded.from_xyz(cs.to_xyz(x)))) < 1e-3


@pytest.mark.parametrize("version", [2, 4])
def test_two_channel_space_clut_fallback_in_v2_v4(tmp_path, version):
    cs = TwoChannelChromaSpace()
    path = str(tmp_path / "twochan.icc")
    r = cs.to_icc_profile(path, version=version, force_clut=True)
    assert r["method"] == "clut_fallback"
    reloaded = ICCSpace(path)
    x = _sample(cs, 100, seed=4)
    assert np.max(np.abs(cs.to_xyz(x) - reloaded.to_xyz(x))) < 1e-3


def test_two_channel_space_is_exact_in_v5_not_a_needless_clut():
    cs = TwoChannelChromaSpace()
    r = cs.to_icc_profile(version=(5, 0, 0))
    assert r["method"] == "exact_mpet", r["notes"]

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "twochan_v5.icc")
        cs.to_icc_profile(path, version=(5, 0, 0))
        reloaded = ICCSpace(path)
        x = _sample(cs, 200, seed=5)
        xyz1, xyz2 = cs.to_xyz(x), reloaded.to_xyz(x)
        assert np.max(np.abs(xyz1 - xyz2)) < 1e-4


def test_hsv_v5_is_not_masked_as_a_plain_matrix_curve_chain():
    from lcms2.icc._tracer.chain import decompose_chain
    from lcms2.icc._tracer.tracing import trace_function

    cs = HSVSpace()
    node, out_ch = trace_function(cs.to_xyz, 3)
    assert decompose_chain(node, 3, out_ch) is None
    node, out_ch = trace_function(cs.from_xyz, 3)
    assert decompose_chain(node, 3, out_ch) is None


def test_hue_wraparound_interpolation_is_fixed_not_just_disclosed():
    from lcms2.icc._interp import sample_clut

    clut = np.array([[0.99], [0.01]])
    coords = np.array([[0.5]])
    wrong = sample_clut(clut, coords, circular_channels=frozenset())[0, 0]
    fixed = sample_clut(clut, coords, circular_channels={0})[0, 0]
    assert abs(wrong - 0.5) < 1e-6
    assert fixed < 0.05 or fixed > 0.95


def test_hsv_v5_calc_roundtrips_via_crs_reader_self_consistency_only():
    cs = HSVSpace()
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hsv_calc.icc")
        r = cs.to_icc_profile(path, version=(5, 0, 0))
        assert r["method"] == "exact_calc", r["notes"]
        reloaded = ICCSpace(path)
        rng = np.random.default_rng(42)
        x = np.stack([rng.uniform(0, 1, 500) for _ in range(3)], axis=-1)
        xyz_ref = cs.to_xyz(x)
        xyz_got = reloaded.to_xyz(x)
        assert np.allclose(xyz_ref, xyz_got, atol=1e-5), "to_xyz calc round trip mismatch"

        xyz2 = cs.to_xyz(_sample(cs, 500, seed=7))
        dev_ref = cs.from_xyz(xyz2)
        dev_got = reloaded.from_xyz(xyz2)
        d_ = np.abs(dev_ref - dev_got)
        d_[:, 0] = np.minimum(d_[:, 0], 1 - d_[:, 0])
        assert np.max(d_) < 1e-4, "from_xyz calc round trip mismatch"


def _run_calc(ops, x, in_ch, out_ch):
    from lcms2.icc._io import mpe

    decoded = mpe.decode_calc_element(mpe.encode_calc_element(in_ch, out_ch, ops))
    mpe.validate_calc_element(decoded, in_ch, out_ch)
    return mpe.eval_calc_element(decoded, x, out_ch)


def test_calc_element_encode_decode_eval_with_branches():
    ops = [
        ("in  ", (0, 1)),
        ("max ", (0, 0)),
        ("data", 2.0),
        ("mul ", (0, 0)),
        ("in  ", (0, 0)),
        ("in  ", (1, 0)),
        ("gt  ", (0, 0)),
        ("if  ", 1),
        ("else", 1),
        ("data", 1.0),
        ("data", 0.0),
        ("sub ", (0, 0)),
        ("out ", (0, 0)),
    ]
    x = np.array([[0.3, 0.7], [0.9, 0.2], [0.5, 0.5]])
    got = _run_calc(ops, x, 2, 1)[:, 0]
    expected = np.maximum(x[:, 0], x[:, 1]) * 2 - (x[:, 0] > x[:, 1])
    assert np.allclose(got, expected)


@pytest.mark.parametrize(
    "sig,fn",
    [
        ("neg ", np.negative),
        ("abs ", np.abs),
        ("sqrt", np.sqrt),
        ("cbrt", np.cbrt),
        ("sq  ", np.square),
        ("flor", np.floor),
        ("ceil", np.ceil),
        ("exp ", np.exp),
        ("ln  ", np.log),
        ("log ", np.log10),
        ("sin ", np.sin),
        ("cos ", np.cos),
        ("tan ", np.tan),
        ("atan", np.arctan),
    ],
)
def test_calc_unary_ops_match_numpy(sig, fn):
    x = np.array([[0.1], [0.5], [2.5]])
    got = _run_calc([("in  ", (0, 0)), (sig, (0, 0)), ("out ", (0, 0))], x, 1, 1)[:, 0]
    assert np.allclose(got, fn(x[:, 0]))


def test_calc_reference_semantics():
    x = np.array([[3.0, 0.0], [-7.0, 2.0], [1.0, -1.0]])
    div = _run_calc([("in  ", (0, 1)), ("div ", (0, 0)), ("out ", (0, 0))], x, 2, 1)[:, 0]
    assert np.allclose(div, [0.0, -3.5, -1.0])
    mod = _run_calc([("in  ", (0, 1)), ("mod ", (0, 0)), ("out ", (0, 0))], x, 2, 1)[:, 0]
    assert np.allclose(mod, [0.0, np.fmod(-7.0, 2.0), 0.0])
    atan = _run_calc(
        [("in  ", (1, 0)), ("in  ", (0, 0)), ("atn2", (0, 0)), ("out ", (0, 0))], x, 2, 1
    )[:, 0]
    assert np.allclose(atan, np.arctan2(x[:, 0], x[:, 1]))


def test_compiled_mod_uses_numpy_semantics():
    from lcms2.icc._tracer.calc import compile_calc_program
    from lcms2.icc._tracer.tracing import trace_function

    root, _ = trace_function(lambda x: np.stack([np.mod(x[..., 0], 1.0)], -1), 1)
    x = np.array([[-0.25], [1.75], [0.5]])
    got = _run_calc(compile_calc_program(root, 1, 1), x, 1, 1)[:, 0]
    assert np.allclose(got, np.mod(x[:, 0], 1.0))


def test_calc_rejects_unknown_ops_and_underflow():
    from lcms2.icc._io import mpe

    for ops in (
        [("zzzz", (0, 0)), ("out ", (0, 0))],
        [("add ", (0, 0)), ("out ", (0, 0))],
        [("in  ", (5, 0)), ("out ", (0, 0))],
        [("in  ", (0, 0)), ("pop ", (0, 0))],
    ):
        decoded = mpe.decode_calc_element(mpe.encode_calc_element(1, 1, ops))
        with pytest.raises(ICCParseError):
            mpe.validate_calc_element(decoded, 1, 1)


def test_hsv_v5_icc2_calc_matches_pure_numpy_across_many_samples():
    import os
    import tempfile

    cs = HSVSpace()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hsv_icc2.icc")
        r = cs.to_icc_profile(path, version=(5, 0, 0))
        assert r["method"] == "exact_calc", r["notes"]
        reloaded = ICCSpace(path)

        rng = np.random.default_rng(123)
        n = 2000
        hsv = np.stack([rng.uniform(0, 1, n) for _ in range(3)], axis=-1)
        xyz_ref = cs.to_xyz(hsv)
        xyz_got = reloaded.to_xyz(hsv)
        assert np.allclose(xyz_ref, xyz_got, atol=1e-5)

        rgb_samples = _sample(cs, n, seed=99)
        xyz2 = cs.to_xyz(rgb_samples)
        dev_ref = cs.from_xyz(xyz2)
        dev_got = reloaded.from_xyz(xyz2)
        d_ = np.abs(dev_ref - dev_got)
        d_[:, 0] = np.minimum(d_[:, 0], 1 - d_[:, 0])
        assert np.max(d_) < 1e-4


@pytest.mark.parametrize("version", [2, 4])
def test_non_unit_device_domain_stays_exact_in_v2_v4(version, tmp_path):
    from lcms2.spaces._colorimetry import D50

    class LogY(ColorSpace):
        name = "test_log_y_domain"
        n_channels = 1
        device_domain = (-23.025850929940457, 0.0)

        def to_xyz(self, l):
            return np.exp(l[..., 0])[..., None] * D50

        def from_xyz(self, xyz):
            return np.stack([np.log(xyz[..., 1] / D50[1] + 1e-10)], axis=-1)

    cs = LogY()
    path = tmp_path / "logy.icc"
    assert cs.to_icc_profile(str(path), version=version)["method"] == "exact_gray_trc"
    reloaded = ICCSpace(str(path))
    xyz = np.linspace(0.01, 1.0, 50)[:, None] * D50
    np.testing.assert_allclose(reloaded.from_xyz(xyz), cs.from_xyz(xyz), atol=0.05)
