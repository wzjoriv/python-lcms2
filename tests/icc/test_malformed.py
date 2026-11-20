import os

import numpy as np
import pytest

from lcms2.icc import ICCParseError, ICCProfile

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")


def test_bad_signature_rejected():
    with pytest.raises(ICCParseError):
        ICCProfile.load(os.path.join(PROFILES_DIR, "bad.icc"))


def test_truncated_file_rejected():
    with pytest.raises(ICCParseError):
        ICCProfile.load(os.path.join(PROFILES_DIR, "toosmall.icc"))


def test_empty_bytes_rejected():
    with pytest.raises(ICCParseError):
        ICCProfile.from_bytes(b"")


def test_garbage_bytes_rejected():
    with pytest.raises(ICCParseError):
        ICCProfile.from_bytes(b"\x00" * 200)


def _generated_profiles():
    from lcms2.spaces import cmyk, gray, hsv, oklab, srgb

    return [
        cs.to_icc_profile(version=v)["bytes"]
        for cs in (srgb, hsv, oklab, gray, cmyk)
        for v in (2, 4, 5)
    ]


@pytest.mark.parametrize("tag_index", range(4))
def test_truncated_tags_raise_parse_error(tag_index):
    from lcms2.icc import ICCProfile, ICCSpace

    for raw in _generated_profiles():
        prof = ICCProfile.from_bytes(raw)
        tags = dict(prof.tags)
        sig = sorted(tags)[tag_index % len(tags)]
        tags[sig] = tags[sig][:12]
        broken = ICCProfile.build_bytes(
            version=prof.header.version,
            device_class=prof.header.device_class,
            data_colour_space=prof.header.data_colour_space,
            pcs=prof.header.pcs,
            tags=tags,
        )
        try:
            space = ICCSpace(broken)
            space.to_xyz(np.full(space.n_channels, 0.5))
            space.from_xyz(np.full(3, 0.5))
        except ICCParseError:
            pass


def test_random_corruption_raises_only_parse_error():
    import glob
    import os

    from lcms2.icc import ICCSpace

    root = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
    sources = [
        open(p, "rb").read()
        for p in glob.glob(os.path.join(root, "*.icc"))
        if os.path.basename(p) not in ("bad.icc", "toosmall.icc")
    ] + _generated_profiles()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        raw = bytearray(sources[rng.integers(len(sources))])
        for _ in range(rng.integers(1, 6)):
            raw[rng.integers(128, len(raw))] = rng.integers(256)
        try:
            space = ICCSpace(bytes(raw))
            with np.errstate(all="ignore"):
                space.to_xyz(rng.random((4, space.n_channels)))
                space.from_xyz(rng.random((4, 3)))
        except ICCParseError:
            pass
