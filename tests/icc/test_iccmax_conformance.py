"""v5 profiles agree with the ICC reference implementation (DemoIccMAX).

Runs when LCMS2_ICCMAX_EVAL names a reference evaluator executable (arguments: profile, direction, n_in, n_out; values on stdin).
"""

import os
import subprocess

import numpy as np
import pytest

import lcms2.spaces as S
from lcms2.icc import ICCSpace

EVAL = os.environ.get("LCMS2_ICCMAX_EVAL")
pytestmark = pytest.mark.skipif(not EVAL, reason="LCMS2_ICCMAX_EVAL is not set")


def _ref(path, direction, x, n_out):
    text = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in x) + "\n"
    out = subprocess.run(
        [EVAL, str(path), direction, str(x.shape[1]), str(n_out)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return np.array([[float(v) for v in line.split()] for line in out.strip().split("\n")])


@pytest.mark.parametrize("force_clut", [False, True])
@pytest.mark.parametrize(
    "cs",
    [S.srgb, S.gray, S.loggray, S.oklab, S.rec2100_pq, S.hsv, S.hwb, S.cmyk, S.lch, S.oklch],
    ids=lambda cs: cs.name,
)
def test_v5_profile_matches_reference(cs, force_clut, tmp_path):
    path = tmp_path / "p.icc"
    cs.to_icc_profile(str(path), version=5, force_clut=force_clut)
    rng = np.random.default_rng(0)
    x = rng.random((100, cs.n_channels))
    xyz = S.srgb.to_xyz(rng.random((100, 3)))
    own = ICCSpace(str(path))
    lo = np.array([d[0] for d in own.device_domain]) if np.ndim(own.device_domain) == 2 else 0.0
    hi = np.array([d[1] for d in own.device_domain]) if np.ndim(own.device_domain) == 2 else 1.0
    ref_xyz = _ref(path, "a2b", x, 3)
    assert np.max(np.abs(ref_xyz - own.to_xyz(x * (hi - lo) + lo))) < 1e-5
    if cs.circular_device_channels:
        return
    ref_dev = _ref(path, "b2a", xyz, cs.n_channels) * (hi - lo) + lo
    assert np.max(np.abs(ref_dev - own.from_xyz(xyz))) < 1e-4 * max(1.0, np.max(np.abs(hi)))
