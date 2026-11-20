"""Channel-count validation of to_xyz and from_xyz."""

import numpy as np
import pytest

from lcms2.spaces import ColorSpace, GraySpace, SRGBSpace, srgb


def test_to_xyz_rejects_too_many_channels_subclass_override():
    with pytest.raises(ValueError, match="last dimension"):
        srgb.to_xyz(np.zeros((5, 4)))


def test_to_xyz_rejects_too_few_channels_subclass_override():
    with pytest.raises(ValueError, match="last dimension"):
        srgb.to_xyz(np.zeros((5, 2)))


def test_from_xyz_rejects_wrong_channel_count_subclass_override():
    with pytest.raises(ValueError, match="last dimension"):
        srgb.from_xyz(np.zeros((4,)))


def test_to_xyz_still_works_for_correct_1d_and_batched_shapes():
    single = srgb.to_xyz(np.array([0.5, 0.5, 0.5]))
    assert single.shape == (3,)
    batched = srgb.to_xyz(np.zeros((5, 3)))
    assert batched.shape == (5, 3)


def test_from_xyz_still_works_for_correct_1d_and_batched_shapes():
    single = srgb.from_xyz(np.array([0.3, 0.3, 0.3]))
    assert single.shape == (3,)
    batched = srgb.from_xyz(np.zeros((5, 3)))
    assert batched.shape == (5, 3)


def test_base_class_with_callables_rejects_wrong_channel_count():
    cs = ColorSpace(n_channels=3, to_xyz=lambda x: x, from_xyz=lambda x: x)
    with pytest.raises(ValueError, match="last dimension"):
        cs.to_xyz(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="last dimension"):
        cs.from_xyz(np.zeros((2, 4)))


def test_base_class_with_callables_still_works_correct_shapes():
    cs = ColorSpace(n_channels=3, to_xyz=lambda x: x, from_xyz=lambda x: x)
    assert cs.to_xyz(np.array([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]
    assert cs.to_xyz(np.zeros((4, 3))).shape == (4, 3)


def test_gray_single_channel_space_rejects_multi_channel_input():
    gray = GraySpace(2.2)
    with pytest.raises(ValueError, match="last dimension"):
        gray.to_xyz(np.zeros((5, 3)))


def test_error_message_names_expected_and_actual_counts():
    with pytest.raises(ValueError) as exc_info:
        srgb.to_xyz(np.zeros((5, 4)))
    msg = str(exc_info.value)
    assert "3" in msg
    assert "4" in msg

    with pytest.raises(ValueError) as exc_info2:
        srgb.to_xyz(np.zeros((5, 2)))
    msg2 = str(exc_info2.value)
    assert "3" in msg2
    assert "2" in msg2


def test_tracing_for_icc_write_is_unaffected_by_validation():
    result = SRGBSpace().to_icc_profile(version=4)
    assert result["method"] == "exact_matrix_trc"
