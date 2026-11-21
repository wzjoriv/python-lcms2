"""Profiles read and written by this library agree with Little CMS."""

import glob
import io
import os

import numpy as np
import pytest

pytest.importorskip("PIL.ImageCms")
from PIL import Image, ImageCms  # noqa: E402

from lcms2 import colorimetry  # noqa: E402
from lcms2 import spaces as S  # noqa: E402
from lcms2.icc import ICCSpace  # noqa: E402
from tests.spaces.toys import TwoChannelChromaSpace  # noqa: E402

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
MODES = {b"RGB ": "RGB", b"CMYK": "CMYK", b"GRAY": "L"}
LAB = ImageCms.createProfile("LAB", 5000)


def _decode_lab(raw: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [raw[..., :1] / 255.0 * 100.0, raw[..., 1:].view(np.int8).astype(np.float64)], axis=-1
    )


def _encode_lab(lab: np.ndarray) -> np.ndarray:
    L = np.round(lab[..., :1] / 100.0 * 255.0)
    ab = np.round(lab[..., 1:]).astype(np.int8).view(np.uint8)
    return np.concatenate([L.astype(np.uint8), ab], axis=-1)


def _lcms_profile(source):
    if isinstance(source, bytes):
        return ImageCms.ImageCmsProfile(io.BytesIO(source))
    return ImageCms.ImageCmsProfile(source)


def _lcms_device_to_lab(source, mode: str, device8: np.ndarray) -> np.ndarray:
    img = Image.fromarray(device8[..., 0] if mode == "L" else device8, mode)
    t = ImageCms.buildTransform(_lcms_profile(source), LAB, mode, "LAB", renderingIntent=1)
    return _decode_lab(np.asarray(ImageCms.applyTransform(img, t)))


def _lcms_lab_to_device(source, mode: str, lab: np.ndarray) -> np.ndarray:
    img = Image.fromarray(_encode_lab(lab), "LAB")
    t = ImageCms.buildTransform(LAB, _lcms_profile(source), "LAB", mode, renderingIntent=1)
    out = np.asarray(ImageCms.applyTransform(img, t)).astype(np.float64) / 255.0
    return out[..., None] if out.ndim == 2 else out


def _range(cs) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.array(cs.device_domain, dtype=np.float64).T
    return lo, hi - lo


def _forward_error(source, cs, mode: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    device8 = rng.integers(0, 255, size=(1, 400, cs.n_channels), dtype=np.uint8)
    lab_lcms = _lcms_device_to_lab(source, mode, device8)
    lo, span = _range(cs)
    lab_ours = colorimetry.xyz_to_lab(cs.to_xyz(lo + device8 / 255.0 * span))
    keep = np.all(np.abs(lab_lcms[..., 1:]) < 120, axis=-1)
    return np.abs(lab_lcms - lab_ours)[keep]


WRITTEN = [
    S.srgb,
    S.display_p3,
    S.a98_rgb,
    S.prophoto_rgb,
    S.rec2020,
    S.srgb_linear,
    S.rec2100_hlg,
    S.rec2100_pq,
    S.gray,
    S.loggray,
    S.cmyk,
]


@pytest.mark.parametrize("version", [2, 4])
@pytest.mark.parametrize("cs", WRITTEN, ids=lambda cs: cs.name)
def test_written_profile_matches_lcms_forward(cs, version):
    result = cs.to_icc_profile(version=version)
    mode = MODES[cs.icc_colour_space_signature or {3: b"RGB ", 1: b"GRAY"}[cs.n_channels]]
    err = _forward_error(result["bytes"], cs, mode)
    limit = 1.5 if result["method"] == "clut_fallback" else 0.5
    assert err.mean() < limit, (result["method"], err.mean())


@pytest.mark.parametrize("version", [2, 4])
@pytest.mark.parametrize("cs", [S.hsv, S.hwb, S.oklab, S.oklch, S.lch, TwoChannelChromaSpace()])
def test_written_profile_loads_in_lcms(cs, version):
    profile = _lcms_profile(cs.to_icc_profile(version=version)["bytes"])
    assert isinstance(ImageCms.getProfileName(profile), str)


@pytest.mark.parametrize("version", [2, 4])
def test_written_cmyk_inverse_matches_lcms(version):
    raw = S.cmyk.to_icc_profile(version=version)["bytes"]
    rng = np.random.default_rng(1)
    rgb = rng.random((200, 3))
    lab = colorimetry.xyz_to_lab(S.srgb.to_xyz(rgb))
    lab_q = _decode_lab(_encode_lab(lab))
    expected = _lcms_lab_to_device(raw, "CMYK", lab_q[None])[0]
    space = ICCSpace(raw)
    lo, span = _range(space)
    got = np.clip((space.from_xyz(colorimetry.lab_to_xyz(lab_q)) - lo) / span, 0, 1)
    assert np.mean(np.abs(expected - got)) * 255 < 1.5


def _real_profiles():
    out = []
    for path in sorted(glob.glob(os.path.join(PROFILES_DIR, "*.icc"))):
        try:
            cs = ICCSpace(path)
        except Exception:
            continue
        mode = MODES.get(cs.icc.header.data_colour_space)
        if mode is not None:
            out.append(pytest.param(path, cs, mode, id=os.path.basename(path)))
    return out


@pytest.mark.parametrize("path,cs,mode", _real_profiles())
def test_real_profile_forward_matches_lcms(path, cs, mode):
    err = _forward_error(path, cs, mode)
    assert err.mean() < 0.5
    assert np.percentile(err, 95) < 1.5


@pytest.mark.parametrize("path,cs,mode", _real_profiles())
def test_real_profile_inverse_matches_lcms(path, cs, mode):
    rng = np.random.default_rng(1)
    lab = np.stack([rng.uniform(10, 90, 300), rng.uniform(-40, 40, 300), rng.uniform(-40, 40, 300)], -1)
    lab_q = _decode_lab(_encode_lab(lab))
    try:
        expected = _lcms_lab_to_device(path, mode, lab_q[None])[0]
    except ImageCms.PyCMSError:
        pytest.skip("Little CMS cannot build an inverse transform for this profile")
    lo, span = _range(cs)
    got = np.clip((cs.from_xyz(colorimetry.lab_to_xyz(lab_q)) - lo) / span, 0, 1)
    assert np.mean(np.abs(expected - got)) * 255 < 1.5
