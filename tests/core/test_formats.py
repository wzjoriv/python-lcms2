"""Format table and inference from arrays."""

import numpy as np
import pytest

import lcms2
from lcms2._formats import as_array, dtype_spec, format_info, parse_format

DTYPES = [np.uint8, np.uint16, np.float16, np.float32, np.float64]


def test_formats_table_has_every_type():
    f = lcms2.FORMATS
    assert len(f) >= 170
    assert f["RGB_8"][1:] == ("uint8", 3, 0, False)
    assert f["RGBA_16"][1:] == ("uint16", 4, 1, False)
    assert f["Lab_DBL"][1] == "float64"
    assert f["RGB_HALF_FLT"][1] == "float16"
    assert f["RGB_8_PLANAR"][4] is True
    assert f["YUVK_8"][0] == f["CMYK_8_REV"][0]


@pytest.mark.parametrize("n", range(5, 13))
def test_cmyk_n_channel_counts(n):
    assert lcms2.FORMATS[f"CMYK{n}_8"][2] == n
    assert lcms2.FORMATS[f"CMYK{n}_16"][2] == n


def test_format_info_and_parse():
    fmt = parse_format("rgba_8")
    assert fmt == lcms2.FORMATS["RGBA_8"][0]
    info = format_info(fmt)
    assert info.channels == 4 and info.extra == 1 and info.dtype == np.uint8
    with pytest.raises(ValueError, match="RGB_8"):
        parse_format("RGB8")


def test_dtype_spec():
    assert dtype_spec(np.float64) == (0, True)
    assert dtype_spec(np.uint8) == (1, False)
    with pytest.raises(TypeError, match="astype"):
        dtype_spec(np.int32)
    with pytest.raises(TypeError):
        dtype_spec(np.dtype(">u2") if np.little_endian else np.dtype("<u2"))


def test_sequence_inference():
    assert as_array([1, 2, 3]).dtype == np.uint8
    assert as_array([1, 2, 300]).dtype == np.uint16
    assert as_array([1, 2.5, 3]).dtype == np.float64
    with pytest.raises(ValueError, match="0..65535"):
        as_array([1, -2, 3])
    with pytest.raises(ValueError):
        as_array([70000])


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_dtype_round_trips(dtype):
    t = lcms2.Transform("sRGB", "sRGB")
    x = np.zeros((4, 3), dtype)
    y = t(x)
    assert y.dtype == dtype and y.shape == (4, 3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_inputs_match_contiguous(dtype):
    t = lcms2.Transform("sRGB", "Lab")
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, (6, 8, 3))
    if np.dtype(dtype).kind == "u":
        base = (base * np.iinfo(dtype).max).round()
    a = base.astype(dtype)
    ref = t(a)
    np.testing.assert_allclose(t(a[::2, 1::3]), ref[::2, 1::3])
    np.testing.assert_allclose(t(a[::-1]), ref[::-1])
    tr = np.ascontiguousarray(a.transpose(1, 0, 2)).transpose(1, 0, 2)
    assert not tr.flags.c_contiguous
    np.testing.assert_allclose(t(tr), ref)
