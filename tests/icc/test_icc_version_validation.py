"""ICC version validation."""

import pytest

from lcms2.icc import ICCWriteError
from lcms2.icc import normalize_icc_version as _normalize_icc_version
from lcms2.spaces import SRGBSpace


@pytest.mark.parametrize(
    "version",
    [
        (4, 99, 99),
        (4, 99, 0),
        (2, 9, 0),
        (2, 0, 1),
        (5, 1, 0),
        (5, 0, 1),
        (3, 0, 0),
    ],
)
def test_unknown_revision_rejected(version):
    with pytest.raises(ICCWriteError):
        _normalize_icc_version(version)


@pytest.mark.parametrize(
    "version",
    [
        (2, 0, 0),
        (2, 1, 0),
        (2, 2, 0),
        (2, 3, 0),
        (2, 4, 0),
        (4, 0, 0),
        (4, 1, 0),
        (4, 2, 0),
        (4, 3, 0),
        (4, 4, 0),
        (5, 0, 0),
    ],
)
def test_real_revisions_accepted(version):
    assert _normalize_icc_version(version) == version


def test_bare_int_shorthand_still_works():
    assert _normalize_icc_version(2) == (2, 4, 0)
    assert _normalize_icc_version(4) == (4, 4, 0)
    assert _normalize_icc_version(5) == (5, 0, 0)


def test_default_version_is_v4():
    space = SRGBSpace()
    result = space.to_icc_profile()
    assert result["method"] == "exact_matrix_trc"


def test_rejected_and_accepted_through_to_icc_profile():
    space = SRGBSpace()
    with pytest.raises(ICCWriteError):
        space.to_icc_profile(version=(4, 99, 99))
    result = space.to_icc_profile(version=(4, 2, 0))
    assert result["method"] == "exact_matrix_trc"
