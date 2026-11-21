"""
Real-world ICC profiles from a folder named by LCMS2_ICC_CORPUS (run monthly in CI; skipped otherwise).
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import lcms2
from lcms2 import Profile
from lcms2.icc import ICCSpace

ROOT = os.environ.get("LCMS2_ICC_CORPUS")
FILES = sorted(p for p in Path(ROOT).rglob("*") if p.suffix.lower() in (".icc", ".icm")) if ROOT else []
DEVICE_CLASSES = ("input", "display", "output", "colorspace")

pytestmark = pytest.mark.skipif(not FILES, reason="set LCMS2_ICC_CORPUS to a folder of ICC profiles")


def _load(path: Path) -> Profile:
    try:
        p = Profile(path)
    except lcms2.CMSError:
        pytest.skip("not a readable ICC profile")
    if str(p.device_class) not in DEVICE_CLASSES:
        pytest.skip(f"{p.device_class} profile")
    return p


def _samples(space: ICCSpace) -> np.ndarray:
    lo, hi = np.array(space.device_domain).T
    return lo + np.random.default_rng(0).random((300, space.n_channels)) * (hi - lo)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_numpy_evaluator_matches_littlecms(path):
    p = _load(path)
    try:
        space = ICCSpace(p.to_bytes())
    except lcms2.ICCParseError:
        pytest.skip("the NumPy evaluator does not read this profile")
    x = _samples(space)
    via_lcms = Profile(path).transform(x, "Lab", intent="relative")
    via_numpy = lcms2.colorimetry.xyz_to_lab(space.to_xyz(x))
    assert lcms2.delta_e(via_lcms, via_numpy, method="2000").max() < 5.0


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_use_does_not_change_colors_and_rewrites_closely(path):
    raw = path.read_bytes()
    p = _load(path)
    try:
        space = ICCSpace(raw)
    except lcms2.ICCParseError:
        pytest.skip("the NumPy evaluator does not read this profile")
    x = _samples(space)
    fresh = Profile.from_bytes(raw).transform(x, "Lab", intent="relative")
    p.transform(x, "Lab", intent="relative")
    saved = Profile.from_bytes(p.to_bytes()).transform(x, "Lab", intent="relative")
    np.testing.assert_allclose(saved, fresh, atol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = space.to_icc_profile()
    rewritten = Profile.from_bytes(result["bytes"])
    want = lcms2.colorimetry.xyz_to_lab(space.to_xyz(x))
    limit = 5.0 if result["method"] == "clut_fallback" else 0.5
    assert lcms2.delta_e(rewritten.transform(x, "Lab", intent="relative"), want, method="2000").max() < limit
