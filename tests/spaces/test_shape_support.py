"""Single-pixel and batched array shapes."""

import os
import tempfile

import numpy as np

from lcms2.icc import ICCSpace
from lcms2.spaces import (
    ColorSpace,
    GraySpace,
    HSVSpace,
    LogGraySpace,
    OKLabSpace,
    SRGBSpace,
)

from .toys import TwoChannelChromaSpace

ALL_SPACES = [
    SRGBSpace(),
    GraySpace(2.2),
    OKLabSpace(),
    HSVSpace(),
    LogGraySpace(),
    TwoChannelChromaSpace(),
]


def _single_pixel(cs, seed=0):
    rng = np.random.default_rng(seed)
    domain = cs.device_domain
    if isinstance(domain[0], (tuple, list)):
        return np.array([rng.uniform(lo, hi) for lo, hi in domain])
    lo, hi = domain
    return rng.uniform(lo, hi, size=cs.n_channels)


def test_bare_pixel_to_xyz_from_xyz_shapes_and_matches_batched():
    for cs in ALL_SPACES:
        x1 = _single_pixel(cs, seed=1)
        xyz1 = cs.to_xyz(x1)
        assert xyz1.shape == (3,), f"{type(cs).__name__} to_xyz shape"
        back1 = cs.from_xyz(xyz1)
        assert back1.shape == (cs.n_channels,), f"{type(cs).__name__} from_xyz shape"

        x_batched = x1[None, :]
        xyz_batched = cs.to_xyz(x_batched)
        assert np.allclose(xyz1, xyz_batched[0]), f"{type(cs).__name__} to_xyz batch mismatch"
        back_batched = cs.from_xyz(xyz_batched)
        assert np.allclose(back1, back_batched[0]), f"{type(cs).__name__} from_xyz batch mismatch"


def test_bare_pixel_roundtrip():
    for cs in ALL_SPACES:
        x = _single_pixel(cs, seed=2)
        back = cs.from_xyz(cs.to_xyz(x))
        assert back.shape == x.shape
        assert np.allclose(x, back, atol=1e-6), type(cs).__name__


def test_bare_pixel_through_reloaded_icc_profile_v4():
    for cs, version in [(TwoChannelChromaSpace(), (4, 4, 0)), (HSVSpace(), (4, 4, 0))]:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.icc")
            cs.to_icc_profile(path, version=version)
            reloaded = ICCSpace(path)
            x = _single_pixel(cs, seed=3)
            xyz = reloaded.to_xyz(x)
            assert xyz.shape == (3,)
            back = reloaded.from_xyz(xyz)
            assert back.shape == (cs.n_channels,)


def test_bare_pixel_through_reloaded_icc_profile_v5_calc():
    cs = HSVSpace()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hsv_v5.icc")
        r = cs.to_icc_profile(path, version=(5, 0, 0))
        assert r["method"] == "exact_calc"
        reloaded = ICCSpace(path)
        x = _single_pixel(cs, seed=4)
        xyz = reloaded.to_xyz(x)
        assert xyz.shape == (3,)
        back = reloaded.from_xyz(xyz)
        assert back.shape == (3,)


def test_bare_pixel_with_callable_colorspace():
    def to_xyz(x):
        return np.stack([x[..., 0], x[..., 1], x[..., 2]], axis=-1)

    def from_xyz(xyz):
        return xyz

    cs = ColorSpace(n_channels=3, to_xyz=to_xyz, from_xyz=from_xyz)
    x = np.array([0.1, 0.2, 0.3])
    xyz = cs.to_xyz(x)
    assert xyz.shape == (3,)
    assert np.allclose(xyz, x)
