import numpy as np
import pytest
import lcms2

CMYK_ICM = "tests/cms_data/CMYK.icm"


def test_import():
    assert lcms2 is not None


def test_profile_builtin_srgb():
    srgb = lcms2.Profile("sRGB")
    assert srgb.handle is not None


def test_profile_builtin_prophoto():
    p = lcms2.Profile("ProPhoto RGB")
    assert p.handle is not None


def test_profile_all_builtins():
    builtins = [
        "sRGB", "Lab", "XYZ",
        "Adobe RGB (1998)", "Apple RGB", "Best RGB", "Beta RGB", "Bruce RGB",
        "CIE RGB", "ColorMatch RGB", "Don RGB 4", "ECI RGB v2", "Ekta Space PS5",
        "NTSC RGB", "PAL/SECAM RGB", "ProPhoto RGB", "SMPTE-C RGB", "_sRGB",
        "Wide Gamut RGB",
    ]
    for name in builtins:
        p = lcms2.Profile(name)
        assert p.handle is not None, f"Failed for profile: {name}"


def test_profile_from_file():
    p = lcms2.Profile(filename=CMYK_ICM)
    assert p.handle is not None


def test_profile_from_memory():
    with open(CMYK_ICM, "rb") as f:
        data = f.read()
    p = lcms2.Profile(buffer=data)
    assert p.handle is not None


def test_transform_basic():
    lab  = lcms2.Profile("Lab")
    srgb = lcms2.Profile("sRGB")
    t = lcms2.Transform(lab, "Lab_DBL", srgb, "RGB_16")
    assert t is not None


def test_data_types_keys():
    keys = lcms2.DATA_TYPES.keys()
    assert "RGB_8" in keys
    assert "Lab_DBL" in keys
    assert "BGR_8" in keys


def test_transform_with_flags():
    lab  = lcms2.Profile("Lab")
    srgb = lcms2.Profile("sRGB")
    t = lcms2.Transform(lab, "Lab_DBL", srgb, "RGB_16",
                        intent="PERCEPTUAL",
                        flags="GAMUTCHECK,SOFTPROOFING")
    assert t is not None


def test_apply_single_pixel():
    lab  = lcms2.Profile("Lab")
    srgb = lcms2.Profile("sRGB")
    t = lcms2.Transform(lab, "Lab_DBL", srgb, "RGB_16")
    result = t.apply([50.0, 0.0, 0.0])
    assert result.shape == (3,)
    assert result.dtype == np.uint16


def test_apply_full_image():
    srgb     = lcms2.Profile("sRGB")
    prophoto = lcms2.Profile("ProPhoto RGB")
    transform = lcms2.Transform(srgb, "RGB_8", prophoto, "RGB_16")
    image  = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = transform.apply(image)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint16


def test_format_strings_case_insensitive():
    srgb = lcms2.Profile("sRGB")
    lab  = lcms2.Profile("Lab")
    for fmt_in, fmt_out in [("BGR_8", "Lab_FLT"), ("bgr_8", "lab_flt"), ("Bgr_8", "LAB_FLT")]:
        t = lcms2.Transform(srgb, fmt_in, lab, fmt_out)
        assert t is not None


def test_lab_lightness_boost():
    srgb = lcms2.Profile("sRGB")
    lab  = lcms2.Profile("Lab")
    to_lab  = lcms2.Transform(srgb, "bgr_8",   lab,  "lab_flt")
    to_srgb = lcms2.Transform(lab,  "lab_flt", srgb, "rgb_flt")

    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    lab_image = to_lab.apply(image)

    assert lab_image.dtype == np.float32
    assert lab_image.shape == (100, 100, 3)
    assert lab_image[..., 0].min() >= 0.0
    assert lab_image[..., 0].max() <= 100.0

    lab_image[..., 0] = np.clip(lab_image[..., 0] * 1.2, 0, 100)
    result = np.clip(to_srgb.apply(lab_image), 0.0, 1.0)

    assert result.dtype == np.float32
    assert result.shape == (100, 100, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_cv2_save_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")

    srgb = lcms2.Profile("sRGB")
    lab  = lcms2.Profile("Lab")
    to_lab  = lcms2.Transform(srgb, "bgr_8",   lab,  "lab_flt")
    to_srgb = lcms2.Transform(lab,  "lab_flt", srgb, "rgb_flt")

    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    lab_image = to_lab.apply(image)
    lab_image[..., 0] = np.clip(lab_image[..., 0] * 1.2, 0, 100)
    result = to_srgb.apply(lab_image)

    bgr_out = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out_path = str(tmp_path / "output.jpg")
    ok = cv2.imwrite(out_path, bgr_out)

    assert ok
    assert bgr_out.shape == (100, 100, 3)
    assert bgr_out.dtype == np.uint8

    loaded = cv2.imread(out_path)
    assert loaded is not None
    assert loaded.shape == (100, 100, 3)