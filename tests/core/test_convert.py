"""lcms2.convert, the space resolver, constants and module-level helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

import lcms2
from lcms2 import _convert


class FakeLinearSRGB:
    """Linear sRGB adapted to D50 (Bradford), as a ColorSpace would define it."""

    M = np.array(
        [
            [0.4360747, 0.3850649, 0.1430804],
            [0.2225045, 0.7168786, 0.0606169],
            [0.0139322, 0.0971045, 0.7141733],
        ]
    )

    def to_xyz(self, x):
        return np.asarray(x) @ self.M.T

    def from_xyz(self, xyz):
        return np.asarray(xyz) @ np.linalg.inv(self.M).T


def test_convert_basic_and_cache():
    _convert._cache.clear()
    a = lcms2.convert([255, 0, 0], "sRGB", "Lab", intent="relative")
    np.testing.assert_allclose(a, [54.29, 80.80, 69.89], atol=0.02)
    lcms2.convert([255, 0, 0], "srgb", "lab", intent="relative")
    assert len(_convert._cache) == 1
    lcms2.convert([255, 0, 0], "sRGB", "Lab", intent="perceptual")
    assert len(_convert._cache) == 2


def test_convert_out_and_dtype():
    out = np.empty(3, np.float32)
    assert lcms2.convert([1.0, 1.0, 1.0], "sRGB", "Lab", out=out, out_dtype=np.float32) is out


def test_duck_typed_space_routes_through_xyz():
    space = FakeLinearSRGB()
    lin = lcms2.convert([1.0, 0.5, 0.0], "sRGB", space)
    expected = np.array([1.0, ((0.5 + 0.055) / 1.055) ** 2.4, 0.0])
    np.testing.assert_allclose(lin, expected, atol=2e-3)
    back = lcms2.convert(lin, space, "sRGB")
    np.testing.assert_allclose(back, [1.0, 0.5, 0.0], atol=2e-3)
    np.testing.assert_allclose(lcms2.convert([0.2, 0.3, 0.4], space, space), [0.2, 0.3, 0.4])


def test_space_name_lookup():
    from lcms2.spaces import OKLabSpace

    assert isinstance(_convert._resolve_space("OKLab"), OKLabSpace)
    assert isinstance(_convert._resolve_space("sRGB"), lcms2.Profile)  # built-ins win
    with pytest.raises(KeyError, match="Color spaces: .*oklab"):
        _convert._resolve_space("unknown")
    with pytest.raises(FileNotFoundError):
        _convert._resolve_space("missing.icc")


def test_intent_and_flag_parsing():
    assert lcms2.Intent.parse("relative") is lcms2.Intent.RELATIVE_COLORIMETRIC
    assert lcms2.Intent.parse("preserve-k-only perceptual") == 10
    assert lcms2.Intent.parse(3) is lcms2.Intent.ABSOLUTE_COLORIMETRIC
    with pytest.raises(ValueError, match="perceptual"):
        lcms2.Intent.parse(7)
    assert (
        lcms2.Flag.parse("8BITS_DEVICELINK|bpc")
        == lcms2.Flag.EIGHTBITS_DEVICELINK | lcms2.Flag.BLACKPOINTCOMPENSATION
    )
    assert lcms2.Flag.parse("NONE") == 0 and lcms2.Flag.parse(None) == 0
    assert int(lcms2.Flag.grid_points(17)) == 17 << 16
    with pytest.raises(TypeError):
        lcms2.Flag.parse(1.5)


def test_errors_and_version():
    assert issubclass(lcms2.CMSError, ValueError)
    assert lcms2.CMSError is lcms2._lcms2.LCMSError
    assert lcms2.get_version() == "2.20"


def test_data_tables():
    assert lcms2.WHITE_POINTS[10]["D65"]["xyY"][0] == pytest.approx(0.31382)
    for name in ("Display P3", "DCI-P3", "Rec.2020", "Rec.709"):
        assert name in lcms2.RGB_SPACES


def test_format_table_generator_matches_header():
    root = Path(__file__).resolve().parents[2]
    if not (root / "scripts" / "lcms2_tools").exists():
        pytest.skip("scripts/ not available (installed-package run)")
    sys.path.insert(0, str(root / "scripts"))
    try:
        from lcms2_tools import _header, newest_stable_tag
    finally:
        sys.path.pop(0)
    text = (root / "vendor" / "Little-CMS" / "include" / "lcms2.h").read_text(encoding="utf-8")
    assert (root / "lcms2" / "_ext" / "formats_table.h").read_text(encoding="utf-8") == _header.render_table(
        text
    )
    assert set(_header.format_names(text)) == set(lcms2.FORMATS)
    assert "cmsCreateTransform" in _header.cmsapi_functions(text)
    assert newest_stable_tag(["lcms2.9", "lcms2.20", "lcms2.21rc1", "lcms2.20.1", "foo"]) == "lcms2.20.1"
