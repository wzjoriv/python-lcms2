"""Reading the real-world profiles in profiles/."""

import glob
import os

import numpy as np
import pytest

from lcms2.icc import ICCParseError, ICCProfile
from lcms2.icc._reader import build_transform

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")

KNOWN_BAD = {"bad.icc", "toosmall.icc"}


def all_profile_paths():
    return sorted(glob.glob(os.path.join(PROFILES_DIR, "*.icc"))) + sorted(
        glob.glob(os.path.join(PROFILES_DIR, "*.icm"))
    )


@pytest.mark.parametrize("path", all_profile_paths())
def test_parses_header_and_tags(path):
    name = os.path.basename(path)
    if name in KNOWN_BAD:
        with pytest.raises(ICCParseError):
            ICCProfile.load(path)
        return
    prof = ICCProfile.load(path)
    assert prof.header.version_major in (2, 3, 4, 5)
    assert len(prof.tags) > 0


@pytest.mark.parametrize("path", all_profile_paths())
def test_builds_transform_when_possible(path):
    name = os.path.basename(path)
    if name in KNOWN_BAD:
        pytest.skip("intentionally corrupt fixture")
    prof = ICCProfile.load(path)
    try:
        built = build_transform(prof)
    except ICCParseError:
        pytest.skip(f"{name}: no matrix/TRC, gray, or LUT model found (e.g. a link/abstract profile)")
        return

    n = built.n_channels
    rng = np.random.default_rng(0)
    device = rng.uniform(0, 1, size=(20, n))
    xyz = built.to_xyz(device)
    assert xyz.shape == (20, 3)
    assert np.all(np.isfinite(xyz))

    try:
        back = built.from_xyz(xyz)
        assert back.shape == (20, n)
        assert np.all(np.isfinite(back))
    except ICCParseError:
        pass


def test_srgb_white_maps_near_d50():
    candidates = [
        p
        for p in all_profile_paths()
        if "sRGB" in os.path.basename(p) or "srgb" in os.path.basename(p).lower()
    ]
    found = False
    for path in candidates:
        prof = ICCProfile.load(path)
        try:
            built = build_transform(prof)
        except ICCParseError:
            continue
        if built.kind != "matrix_trc":
            continue
        white_xyz = built.to_xyz(np.array([[1.0, 1.0, 1.0]]))[0]
        assert abs(white_xyz[1] - 1.0) < 0.05, f"{path}: Y={white_xyz[1]}"
        found = True
    assert found, "expected at least one matrix/TRC sRGB-like test profile"
