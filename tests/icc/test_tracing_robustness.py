"""Spaces written in varied numpy styles are written exactly and reload accurately."""

import math

import numpy as np
import pytest

from lcms2.icc._io.profile import ICCProfile
from lcms2.icc._tracer.tracing import TraceAbort, trace_function
from lcms2.spaces import ColorSpace, ICCSpace, RGBSpace, srgb
from lcms2.spaces._colorimetry import D50
from lcms2.spaces._srgb import SRGBSpace, srgb_decode, srgb_encode

M = SRGBSpace.matrix
M_INV = np.linalg.inv(M)


def _space(name, n, to_xyz, from_xyz, domain=(0.0, 1.0)):
    return type(
        name,
        (ColorSpace,),
        {
            "name": name,
            "n_channels": n,
            "device_domain": domain,
            "to_xyz": lambda self, x: to_xyz(np.asarray(x, dtype=np.float64)),
            "from_xyz": lambda self, x: from_xyz(np.asarray(x, dtype=np.float64)),
        },
    )()


def _assign_channels(x):
    lin = np.empty_like(x)
    for i in range(3):
        lin[..., i] = srgb_decode(x[..., i])
    return lin @ M.T


def _assign_into_copy(x):
    lin = x @ M_INV.T
    out = lin.copy()
    out[..., :] = srgb_encode(lin)
    return out


RGB_STYLES = {
    "item_assignment": (_assign_channels, _assign_into_copy),
    "column_stack": (
        lambda x: np.column_stack([srgb_decode(x[..., i]) for i in range(3)]) @ M.T,
        lambda x: srgb_encode(x @ M_INV.T),
    ),
    "vectorized": (lambda x: srgb_decode(x) @ M.T, lambda x: srgb_encode(x @ M_INV.T)),
    "slices": (
        lambda x: np.concatenate([srgb_decode(x[..., i : i + 1]) for i in range(3)], -1) @ M.T,
        lambda x: srgb_encode(x @ M_INV.T),
    ),
    "negative_index": (
        lambda x: np.stack([srgb_decode(x[..., i - 3]) for i in range(3)], -1) @ M.T,
        lambda x: srgb_encode(x @ M_INV.T),
    ),
    "einsum": (
        lambda x: np.einsum("ij,...j->...i", M, srgb_decode(x)),
        lambda x: srgb_encode(np.einsum("...j,ij->...i", x, M_INV)),
    ),
    "unpack_split": (
        lambda x: np.concatenate([srgb_decode(c) for c in np.split(x, 3, axis=-1)], -1) @ M.T,
        lambda x: srgb_encode(x @ M_INV.T),
    ),
    "per_channel_vectors": (
        lambda x: (srgb_decode(x) * np.array([1.0, 1.0, 1.0]) + np.zeros(3)) @ M.T,
        lambda x: srgb_encode((x @ M_INV.T) / np.array([1.0, 1.0, 1.0])),
    ),
    "untraceable_python": (
        lambda x: (
            np.vectorize(lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)(x)
            @ M.T
        ),
        lambda x: srgb_encode(x @ M_INV.T),
    ),
}


@pytest.mark.parametrize("version", [2, 4, 5])
@pytest.mark.parametrize("style", list(RGB_STYLES))
def test_rgb_styles_are_exact(style, version, tmp_path):
    to_xyz, from_xyz = RGB_STYLES[style]
    cs = _space(f"style-{style}", 3, to_xyz, from_xyz)
    path = tmp_path / "p.icc"
    result = cs.to_icc_profile(str(path), version=version)
    expected = "exact_mpet" if version == 5 else "exact_matrix_trc"
    assert result["method"] == expected, result["notes"]
    x = np.random.default_rng(0).random((200, 3))
    assert np.max(np.abs(ICCSpace(str(path)).to_xyz(x) - srgb.to_xyz(x))) < 2e-4


def test_per_channel_constant_vector_is_not_collapsed_to_its_first_entry():
    root, width = trace_function(lambda x: x * np.array([1.0, 2.0, 3.0]) + np.array([4, 5, 6]), 3)
    assert width == 3
    from lcms2.icc._tracer.chain import decompose_chain

    chain = decompose_chain(root, 3, 3)
    (kind, matrix, offset), *_ = chain
    assert kind == "matrix"
    np.testing.assert_allclose(np.diag(matrix), [1, 2, 3])
    np.testing.assert_allclose(offset, [4, 5, 6])


def test_assignment_into_copy_leaves_original_unchanged():
    def fn(x):
        y = x.copy()
        y[..., 0] = x[..., 0] * 2.0
        return x + y

    from lcms2.icc._tracer.chain import decompose_chain

    root, _ = trace_function(fn, 3)
    (kind, matrix, offset), *_ = decompose_chain(root, 3, 3)
    np.testing.assert_allclose(np.diag(matrix), [3.0, 2.0, 2.0])


def test_broadcasting_mismatched_widths_raises():
    with pytest.raises(TraceAbort):
        trace_function(lambda x: x + np.array([1.0, 2.0]), 3)


LOG_Y_DOMAIN = (np.log(1e-10), 0.0)


def _log_luminance(direction):
    def to_xyz(lum):
        y = np.exp(lum[..., 0])
        return np.stack([y * d for d in direction], axis=-1)

    def from_xyz(xyz):
        return np.log(xyz[..., 1:2] / direction[1] + 1e-10)

    return _space(f"logy-{direction[0]:.3f}", 1, to_xyz, from_xyz, LOG_Y_DOMAIN)


@pytest.mark.parametrize(
    "direction,version,method,mean_tol",
    [
        (D50, 4, "exact_gray_trc", 1e-3),
        (D50, 5, "exact_mpet", 1e-4),
        (np.array([0.0, 1.0, 0.0]), 2, "exact_lut", 5e-2),
        (np.array([0.0, 1.0, 0.0]), 4, "exact_lut", 5e-2),
        (np.array([0.0, 1.0, 0.0]), 5, "exact_mpet", 1e-4),
    ],
)
def test_log_luminance_reload_accuracy(direction, version, method, mean_tol, tmp_path):
    cs = _log_luminance(direction)
    path = tmp_path / "logy.icc"
    assert cs.to_icc_profile(str(path), version=version)["method"] == method
    rgb = np.random.default_rng(3).random((5000, 3))
    expected = srgb.to(cs, rgb)
    got = srgb.to(ICCSpace(str(path)), rgb)
    assert np.all(np.isfinite(got))
    assert np.mean(np.abs(expected - got)) < mean_tol


def test_wrong_output_shape_is_reported():
    cs = _space(
        "badshape",
        1,
        lambda l: np.stack([np.zeros_like(l), np.exp(l), np.zeros_like(l)], axis=-1),
        lambda x: np.log(x[..., 1:2]),
    )
    with pytest.raises(ValueError, match="returned shape"):
        cs.to_xyz(np.zeros((4, 1)))


def test_device_domain_argument():
    cs = ColorSpace(
        n_channels=1,
        to_xyz=lambda g: np.asarray(g)[..., 0:1] * D50,
        from_xyz=lambda x: np.asarray(x)[..., 1:2],
        device_domain=(0.0, 2.0),
    )
    assert cs.device_domain == [(0.0, 2.0)]
    assert ICCSpace(cs.to_icc_profile()["bytes"]).device_domain == [(0.0, 2.0)]


def test_rgbspace_subclass_is_registered_and_exact():
    class Wide(RGBSpace):
        name = "test-wide-rgb"
        matrix = M * 1.0

    assert "test-wide-rgb" in ColorSpace.list()
    assert "rgbspace" not in ColorSpace.list()
    assert Wide().to_icc_profile()["method"] == "exact_matrix_trc"


def test_math_module_space_uses_numeric_probe():
    def to_xyz(g):
        y = np.vectorize(lambda v: math.pow(v, 2.2))(g[..., 0])
        return y[..., None] * D50

    cs = _space("mathgray", 1, to_xyz, lambda x: np.power(x[..., 1:2], 1 / 2.2))
    assert cs.to_icc_profile(version=4)["method"] == "exact_gray_trc"
    assert cs.to_icc_profile(version=5)["method"] == "exact_mpet"


@pytest.mark.parametrize("version", [2, 4])
def test_legacy_profiles_use_only_legal_tag_types(version):
    for cs in (srgb, _log_luminance(np.array([0.0, 1.0, 0.0]))):
        for force in (False, True):
            prof = ICCProfile.from_bytes(
                cs.to_icc_profile(version=version, force_clut=force)["bytes"]
            )
            types = {body[:4] for body in prof.tags.values()}
            if version == 2:
                assert not types & {b"para", b"mAB ", b"mBA ", b"mluc", b"mpet"}
            else:
                assert not types & {b"mpet", b"desc"}


@pytest.mark.parametrize("version", [2, 4])
def test_header_class_and_profile_id(version):
    trc = ICCProfile.from_bytes(srgb.to_icc_profile(version=version)["bytes"])
    lut = ICCProfile.from_bytes(srgb.to_icc_profile(version=version, force_clut=True)["bytes"])
    assert trc.header.device_class == b"mntr"
    assert lut.header.device_class == b"spac"
    assert (trc.header.profile_id == b"\x00" * 16) == (version == 2)


_FUZZ_CURVES = [
    lambda x: x**2.2,
    lambda x: np.sqrt(np.abs(x)),
    lambda x: np.exp(x) - 1,
    lambda x: np.where(x < 0.1, x * 2, x**0.5),
    lambda x: np.clip(x, 0, 1),
    lambda x: np.tanh(x),
    lambda x: np.cbrt(x),
]


def _random_fn(rng, n_in, n_out, depth):
    stages, width = [], n_in
    for d in range(depth):
        if rng.random() < 0.5 or d == depth - 1:
            w_out = n_out if d == depth - 1 else int(rng.integers(1, 4))
            stages.append(("m", rng.normal(size=(width, w_out)), rng.normal(size=w_out) * 0.1))
            width = w_out
        else:
            stages.append(
                ("c", [_FUZZ_CURVES[rng.integers(len(_FUZZ_CURVES))] for _ in range(width)])
            )

    def fn(x):
        x = np.asarray(x, dtype=np.float64)
        for stage in stages:
            if stage[0] == "m":
                x = x @ stage[1] + stage[2]
            else:
                x = np.stack([f(x[..., i]) for i, f in enumerate(stage[1])], axis=-1)
        return x

    return fn


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_random_spaces_write_and_reload():
    from lcms2.icc import ICCWriteError

    rng = np.random.default_rng(5)
    for _ in range(40):
        n = int(rng.choice([1, 2, 3, 4]))
        fwd = _random_fn(rng, n, 3, int(rng.integers(1, 4)))
        cs = ColorSpace(n_channels=n, to_xyz=fwd, from_xyz=_random_fn(rng, 3, n, 2))
        for version in (2, 4, 5):
            try:
                result = cs.to_icc_profile(version=version)
            except ICCWriteError:
                continue
            if not result["method"].startswith("exact"):
                continue
            x = rng.random((200, n))
            expected, got = fwd(x), ICCSpace(result["bytes"]).to_xyz(x)
            keep = np.all(np.isfinite(expected), axis=-1)
            if version != 5:
                keep &= np.all((expected >= 0) & (expected <= 65535 / 32768), axis=-1)
            if keep.any():
                err = np.abs(expected[keep] - got[keep])
                assert np.percentile(err, 95) <= 5e-3 * max(1.0, np.abs(expected[keep]).max())
