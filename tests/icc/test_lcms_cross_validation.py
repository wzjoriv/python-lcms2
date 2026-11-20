"""Profiles written by this library agree with Little CMS on sRGB and gray."""

import io

import numpy as np
import pytest

PIL_ImageCms = pytest.importorskip("PIL.ImageCms")
from PIL import Image, ImageCms  # noqa: E402

from lcms2.colorimetry import D50, lab_to_xyz  # noqa: E402
from lcms2.spaces import GraySpace, SRGBSpace  # noqa: E402


def _lcms_lab_from_rgb(profile_bytes_or_path, rgb: np.ndarray) -> np.ndarray:
    if isinstance(profile_bytes_or_path, bytes):
        src = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes_or_path))
    else:
        src = ImageCms.ImageCmsProfile(profile_bytes_or_path)
    lab_profile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransform(
        src,
        lab_profile,
        "RGB",
        "LAB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    flat = np.clip(rgb.reshape(-1, 3), 0, 1)
    px = np.round(flat * 255).astype(np.uint8)
    img = Image.frombytes("RGB", (px.shape[0], 1), px.tobytes())
    out = ImageCms.applyTransform(img, transform)
    raw_u8 = np.frombuffer(out.tobytes(), dtype=np.uint8).reshape(-1, 3)
    L = raw_u8[:, 0].astype(np.float64) * (100.0 / 255.0)
    ab_signed = raw_u8[:, 1:3].astype(np.int8).astype(np.float64)
    a = ab_signed[:, 0]
    b = ab_signed[:, 1]
    lab = np.stack([L, a, b], axis=-1)
    return lab.reshape(rgb.shape[:-1] + (3,))


def _lcms_xyz_from_rgb(profile_bytes_or_path, rgb: np.ndarray) -> np.ndarray:
    lab = _lcms_lab_from_rgb(profile_bytes_or_path, rgb)
    return lab_to_xyz(lab, white=D50)


def test_lcms_loads_crs_written_srgb_v4_profile():
    cs = SRGBSpace()
    result = cs.to_icc_profile(version=(4, 4, 0))
    assert result["method"] == "exact_matrix_trc"
    prof = ImageCms.ImageCmsProfile(io.BytesIO(result["bytes"]))
    name = ImageCms.getProfileName(prof)
    assert isinstance(name, str)


def test_lcms_agrees_with_crs_on_srgb_white_and_primaries():
    cs = SRGBSpace()
    result = cs.to_icc_profile(version=(4, 4, 0))
    assert result["method"] == "exact_matrix_trc"

    samples = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    crs_xyz = cs.to_xyz(samples)
    lcms_xyz = _lcms_xyz_from_rgb(result["bytes"], samples)

    err = np.max(np.abs(crs_xyz - lcms_xyz), axis=-1)
    assert np.all(err < 0.02), f"ours vs lcms2 XYZ mismatch: {err}"


def test_lcms_loads_crs_written_gray_v4_profile():
    cs = GraySpace(gamma=2.2)
    result = cs.to_icc_profile(version=(4, 2, 0))
    assert result["method"] == "exact_gray_trc"
    prof = ImageCms.ImageCmsProfile(io.BytesIO(result["bytes"]))
    assert isinstance(ImageCms.getProfileName(prof), str)


def test_lcms_reads_a_real_v2_and_v4_srgb_profile_the_same_way_crs_does():
    import glob
    import os

    from lcms2.spaces import ICCSpace

    profiles_dir = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
    candidates = list(
        glob.glob(
            os.path.join(
                profiles_dir,
                "*srgb*",
            )
        )
    )
    candidates += glob.glob(os.path.join(profiles_dir, "*sRGB*"))
    tested = 0
    for path in sorted(set(candidates)):
        try:
            crs_space = ICCSpace(path)
        except Exception:
            continue
        if crs_space.kind != "matrix_trc":
            continue
        try:
            ImageCms.ImageCmsProfile(path)
        except Exception:
            continue

        samples = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.2, 0.6, 0.9]])
        crs_xyz = crs_space.to_xyz(samples)
        try:
            lcms_xyz = _lcms_xyz_from_rgb(path, samples)
        except Exception:
            continue
        err = np.max(np.abs(crs_xyz - lcms_xyz), axis=-1)
        assert np.all(err < 0.03), f"{path}: ours vs lcms2 mismatch {err}"
        tested += 1
    assert tested > 0, "expected at least one real sRGB-like matrix/TRC test profile"


def test_pillow_can_apply_a_crs_written_profile_to_a_real_uint8_image():
    cs = SRGBSpace()
    result = cs.to_icc_profile(version=(4, 4, 0))
    assert result["method"] == "exact_matrix_trc"
    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(result["bytes"]))
    dst_profile = ImageCms.createProfile("sRGB")

    width, height = 4, 3
    rng = np.random.default_rng(0)
    pixels_u8 = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels_u8, mode="RGB")
    assert img.mode == "RGB"

    transform = ImageCms.buildTransform(
        src_profile,
        dst_profile,
        "RGB",
        "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    out_img = ImageCms.applyTransform(img, transform)

    assert out_img.mode == "RGB"
    assert out_img.size == img.size
    out_u8 = np.array(out_img)
    assert out_u8.dtype == np.uint8
    assert out_u8.shape == (height, width, 3)

    err = np.max(np.abs(out_u8.astype(np.int16) - pixels_u8.astype(np.int16)))
    assert err <= 3, f"sRGB->sRGB uint8 transform should be near-identity, max diff {err}"
