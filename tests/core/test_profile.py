"""Profile construction, properties and round trips."""

import pickle
import shutil

import numpy as np
import pytest

import lcms2


def is_capsule(h):
    return type(h).__name__ == "PyCapsule"


def test_null_and_builtins():
    p = lcms2.Profile()
    assert "NULL" in p.description and "use freely" in p.copyright and is_capsule(p._handle)
    for name in ("sRGB", "Lab", "XYZ"):
        p = lcms2.Profile(name)
        assert name in p.description and "use freely" in p.copyright


@pytest.mark.parametrize("name", lcms2.Profile.builtins())
def test_every_builtin(name):
    p = lcms2.Profile(name)
    assert p.to_bytes()


@pytest.mark.parametrize(
    "alias,canonical",
    [("adobe rgb", "Adobe RGB (1998)"), ("display p3", "Display P3"), ("prophoto", "ProPhoto RGB"),
     ("SRGB", "sRGB built-in"), ("rec2020", "Rec.2020"), ("bt.709", "Rec.709")],
)
def test_lenient_names(alias, canonical):
    assert lcms2.Profile(alias).description == canonical


def test_srgb_curve_table_matches_builtin():
    lab = lcms2.Profile("Lab")
    v = [50.0, 2.0, -3.0]
    a = lcms2.Transform(lab, "Lab_DBL", lcms2.Profile("sRGB"), "RGB_16").apply(v)
    b = lcms2.Transform(lab, "Lab_DBL", lcms2.Profile("_sRGB"), "RGB_16").apply(v)
    np.testing.assert_allclose(a, b, atol=2)
    np.testing.assert_allclose(
        lcms2.convert([0.2, 0.5, 0.8], "sRGB", "Display P3"),
        lcms2.convert([0.2, 0.5, 0.8], "_sRGB", "Display P3"), atol=2e-3,
    )


def test_unknown_name_and_bad_type():
    with pytest.raises(FileNotFoundError, match="built-in"):
        lcms2.Profile("no such profile")
    with pytest.raises(TypeError, match="ColorSpace"):
        lcms2.Profile(3.5)
    with pytest.raises(ValueError, match="only apply"):
        lcms2.Profile("sRGB", white_point="D65")


def test_files(data_dir, tmp_path):
    p = lcms2.Profile(data_dir / "CMYK.icm")
    assert "CMYK" in p.description and "Offset printing" in p.model and "Public" in p.copyright
    assert p.color_space == "CMYK" and p.channels == 4 and str(p.device_class) == "output"
    assert lcms2.Profile(str(data_dir / "sRGB.icm")).color_space == "RGB"
    with pytest.raises(lcms2.CMSError):
        lcms2.Profile(data_dir / "empty.icm")
    with pytest.raises(FileNotFoundError):
        lcms2.Profile(data_dir / "missing.icm")
    target = tmp_path / "prøfil ü 色.icm"
    shutil.copy(data_dir / "CMYK.icm", target)
    assert lcms2.Profile(target).description == p.description
    assert lcms2.Profile(str(target)).description == p.description


def test_save_and_bytes_round_trip(tmp_path):
    ref = lcms2.Profile("XYZ")
    fn = tmp_path / "x.icc"
    ref.save(fn)
    p = lcms2.Profile(fn)
    assert (p.description, p.model, p.copyright) == (ref.description, ref.model, ref.copyright)
    q = lcms2.Profile.from_bytes(ref.to_bytes())
    assert q.description == ref.description
    assert pickle.loads(pickle.dumps(ref)).description == ref.description


def test_lab_white_points():
    from lcms2._builtins import white_point_from_temperature

    np.testing.assert_allclose(white_point_from_temperature(6504), lcms2.WHITE_POINTS[2]["D65"]["xyY"], atol=1e-4)
    for p in (lcms2.Profile("Lab", white_point="D65"), lcms2.Profile("Lab", temperature=6504),
              lcms2.Profile("Lab", white_point=[0.3127, 0.3290, 1.0]), lcms2.Profile.lab("D65", version=2)):
        assert "Lab" in p.description and p.color_space == "Lab"
    with pytest.raises(ValueError, match="illuminant"):
        lcms2.Profile.lab("D66")


def test_info_header_tags_round_trip():
    p = lcms2.Profile.rgb("D65", gamma=2.2, description="first")
    p.description = "Mïne"
    p.manufacturer = "Acme"
    p.model = "M1"
    p.copyright = "CC0"
    p.rendering_intent = "saturation"
    p.set_header(manufacturer="ACME", model="m1", flags=1, attributes=3)
    p.write_tag_raw("zzzz", b"text\x00\x00\x00\x00hello\x00")
    q = lcms2.Profile.from_bytes(p.to_bytes())
    assert (q.description, q.manufacturer, q.model, q.copyright) == ("Mïne", "Acme", "M1", "CC0")
    assert q.rendering_intent == lcms2.Intent.SATURATION
    q.version = 2.1
    q.description = "v2 text"
    r = lcms2.Profile.from_bytes(q.to_bytes())
    assert r.version == pytest.approx(2.1) and r.description == "v2 text"
    h = q.header
    assert (h["flags"], h["attributes"]) == (1, 3)
    assert "zzzz" in q.tags and q.has_tag("zzzz")
    assert q.read_tag_raw("zzzz") == b"text\x00\x00\x00\x00hello\x00"
    with pytest.raises(KeyError):
        q.read_tag_raw("nope")


def test_profile_id():
    p = lcms2.Profile("sRGB")
    assert p.profile_id == bytes(16)
    pid = p.compute_id()
    assert len(pid) == 16 and pid != bytes(16) and p.profile_id == pid


def test_readers(srgb, cmyk):
    np.testing.assert_allclose(srgb.media_white_point, [0.9642, 1.0, 0.8249], atol=1e-3)
    assert srgb.colorants.shape == (3, 3) and srgb.chromatic_adaptation.shape == (3, 3)
    assert srgb.is_matrix_shaper and not cmyk.is_matrix_shaper
    assert set(srgb.trc) == {"rTRC", "gTRC", "bTRC"}
    assert cmyk.is_clut("perceptual", "output")
    assert 250 < cmyk.tac < 400
    assert srgb.tac == 0
    assert lcms2.Intent.PERCEPTUAL in srgb.supported_intents
    assert srgb.is_intent_supported("relative", "output")
    assert srgb.named_colors is None
    bp = cmyk.black_point("perceptual", destination=True)
    assert bp is not None and 0 <= bp[1] < 0.2
    assert isinstance(srgb.creation_date.year, int)


def test_repr_and_constructors():
    r = repr(lcms2.Profile("sRGB"))
    assert r.startswith("<Profile 'sRGB built-in' RGB display v4.")
    assert lcms2.Profile.xyz().color_space == "XYZ"
    assert lcms2.Profile.srgb().description == "sRGB built-in"
    assert lcms2.Profile.gray("D50", 2.2).color_space == "GRAY"
    assert str(lcms2.Profile.abstract(contrast=1.2).device_class) == "abstract"
    lin = lcms2.Profile.linearization([1.0, 2.0, (1, [1.8])], "RGB")
    assert str(lin.device_class) == "link"
    p3 = lcms2.Profile.rgb("D65", ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)), (4, [2.4, 1 / 1.055, 0.055 / 1.055,
                                                                                    1 / 12.92, 0.04045]))
    np.testing.assert_allclose(lcms2.convert([0.3, 0.6, 0.9], p3, "Display P3"), [0.3, 0.6, 0.9], atol=2e-3)
    with pytest.raises(ValueError, match="parameters"):
        lcms2.Profile.rgb(gamma=(4, [2.4]))
    with pytest.raises(ValueError, match="gamma"):
        lcms2.Profile.rgb(gamma=-1.0)
    with pytest.raises(ValueError, match="ink limit"):
        lcms2.Profile.ink_limiting(500)


class FakeSpace:
    def to_icc_profile(self):
        return lcms2.Profile("Display P3").to_bytes()


def test_from_space_duck_typed():
    assert lcms2.Profile(FakeSpace()).description == "Display P3"
    assert lcms2.Profile.from_space(FakeSpace()).color_space == "RGB"
    with pytest.raises(TypeError, match="to_icc_profile"):
        lcms2.Profile.from_space(object())
