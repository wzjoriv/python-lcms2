import numpy as np
import pytest

import lcms2
from lcms2._parallel import _MIN_ITEMS
from lcms2.icc import ICCSpace
from lcms2.spaces import SRGBSpace, xyz_d50


@pytest.fixture
def threads():
    yield
    lcms2.set_threads(None)


def test_matrix_trc_space_in_threads_matches_single_thread(threads):
    space = ICCSpace(SRGBSpace().to_icc_profile(version=(4, 4, 0))["bytes"])
    assert space.kind == "matrix_trc"
    rgb = np.random.default_rng(3).random((_MIN_ITEMS + 1000, 3))
    lcms2.set_threads(1)
    ref_xyz = space.to(xyz_d50, rgb)
    ref_rgb = xyz_d50.to(space, ref_xyz)
    lcms2.set_threads(4)
    xyz = space.to(xyz_d50, rgb)
    np.testing.assert_array_equal(xyz, ref_xyz)
    np.testing.assert_array_equal(xyz_d50.to(space, xyz), ref_rgb)
