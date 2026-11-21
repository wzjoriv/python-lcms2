"""
Spaces without a usable from_xyz are written as input profiles holding A2B0 only.
"""

import numpy as np
import pytest

import lcms2
from lcms2 import ICCWriteError, Profile
from lcms2.spaces import ColorSpace, srgb


class ScannerSpace(ColorSpace):
    name = "test-scanner-one-way"
    n_channels = 3

    def to_xyz(self, rgb):
        return srgb.to_xyz(np.clip(np.asarray(rgb, dtype=np.float64), 0, 1) ** 1.1)

    def from_xyz(self, xyz):
        raise NotImplementedError("scanner characterization has no inverse")


def test_input_profile_with_a2b_only():
    cs = ScannerSpace()
    result = cs.to_icc_profile()
    assert result["method"] == "clut_fallback"
    assert any("A2B0 only" in n for n in result["notes"])
    p = Profile.from_bytes(result["bytes"])
    assert str(p.device_class) == "input" and p.has_tag("A2B0") and not p.has_tag("B2A0")
    x = np.random.default_rng(0).random((200, 3))
    want = lcms2.colorimetry.xyz_to_lab(cs.to_xyz(x))
    assert lcms2.delta_e(p.transform(x, "Lab"), want, method="2000").max() < 1.0


def test_strict_refuses_one_way():
    with pytest.raises(ICCWriteError, match="from_xyz"):
        ScannerSpace().to_icc_profile(strict=True)
