"""Channel count and device range declarations."""

import numpy as np
import pytest

from lcms2.spaces import ColorSpace, ICCSpace, lch, oklab, oklch, srgb


def _space(**kwargs):
    return ColorSpace(
        to_xyz=lambda x: np.zeros(np.shape(x)[:-1] + (3,)),
        from_xyz=lambda xyz: np.asarray(xyz)[..., : kwargs.get("n_out", 1)],
        **{k: v for k, v in kwargs.items() if k != "n_out"},
    )


def test_device_domain_is_always_one_pair_per_channel():
    assert srgb.device_domain == [(0.0, 1.0)] * 3
    assert _space(n_channels=2, device_domain=(-1, 1), n_out=2).device_domain == [(-1.0, 1.0)] * 2
    assert oklab.device_domain == [(0.0, 1.0), (-0.4, 0.4), (-0.4, 0.4)]


def test_channel_count_comes_from_a_per_channel_domain():
    assert oklab.n_channels == lch.n_channels == oklch.n_channels == 3
    cs = _space(device_domain=[(0, 1), (0, 2)], n_out=2)
    assert cs.n_channels == 2

    class TwoChannel(ColorSpace):
        device_domain = [(0, 1), (-1, 1)]

        def to_xyz(self, x):
            return np.zeros(np.shape(x)[:-1] + (3,))

        def from_xyz(self, xyz):
            return np.asarray(xyz)[..., :2]

    assert TwoChannel().n_channels == 2


def test_matching_declarations_are_accepted():
    assert _space(n_channels=2, device_domain=[(0, 1), (0, 2)], n_out=2).n_channels == 2


def test_disagreeing_declarations_raise():
    with pytest.raises(ValueError, match="lists 3 channels but n_channels is 2"):
        _space(n_channels=2, device_domain=[(0, 1)] * 3)


def test_a_shared_pair_needs_n_channels():
    with pytest.raises(TypeError, match="n_channels"):
        _space(device_domain=(0.0, 1.0))
    with pytest.raises(TypeError, match="n_channels"):
        _space()


@pytest.mark.parametrize(
    "domain",
    [(-np.inf, 0.0), (0.0, np.nan), (1.0, 1.0), (2.0, 1.0), [(0, 1), (3, 3)], (0, 1, 2), "ab"],
)
def test_invalid_domains_raise_when_the_space_is_created(domain):
    with pytest.raises(ValueError):
        _space(n_channels=1, device_domain=domain)


@pytest.mark.parametrize("n_channels", [0, -1, 1.5])
def test_invalid_channel_counts_raise(n_channels):
    with pytest.raises(ValueError):
        _space(n_channels=n_channels)


def test_circular_channels_must_exist():
    class Bad(ColorSpace):
        n_channels = 2
        circular_device_channels = frozenset({2})

        def to_xyz(self, x):
            return x

        def from_xyz(self, xyz):
            return xyz

    with pytest.raises(ValueError, match="circular_device_channels"):
        Bad()


@pytest.mark.parametrize("cs", [oklab, lch, srgb], ids=lambda cs: cs.name)
def test_loaded_profile_has_the_same_domain(cs):
    loaded = ICCSpace(cs.to_icc_profile()["bytes"])
    assert loaded.n_channels == cs.n_channels
    np.testing.assert_allclose(loaded.device_domain, cs.device_domain)
    assert loaded.circular_device_channels == cs.circular_device_channels
