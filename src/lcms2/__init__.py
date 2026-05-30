import os
import re
from . import _lcms2
import numpy as np
from .implementation import WHITE_POINTS, CMSError, create_profile, open_profile, profile_from_memory

DATA_TYPES = _lcms2.DATA_TYPES
_DATA_TYPES_UPPER = {k.upper(): k for k in DATA_TYPES}

INTENT = {
    "PERCEPTUAL": 0,
    "RELATIVE_COLORIMETRIC": 1,
    "SATURATION": 2,
    "ABSOLUTE_COLORIMETRIC": 3
}

NON_ICC_INTENT = {
    "PRESERVE_K_ONLY_PERCEPTUAL": 10,
    "PRESERVE_K_ONLY_RELATIVE_COLORIMETRIC":  11,
    "PRESERVE_K_ONLY_SATURATION": 12,
    "PRESERVE_K_PLANE_PERCEPTUAL": 13,
    "PRESERVE_K_PLANE_RELATIVE_COLORIMETRIC": 14,
    "PRESERVE_K_PLANE_SATURATION": 15
}

FLAG = {
    "NONE": 0x0,
    "NOTPRECALC": 0x0100,
    "GAMUTCHECK": 0x1000,
    "SOFTPROOFING": 0x4000,
    "BLACKPOINTCOMPENSATION": 0x2000,
    "PRESERVEBLACK": 0x8000,
    "NULLTRANSFORM": 0x0200,
    "HIGHRESPRECALC": 0x0400,
    "LOWRESPRECALC": 0x0800
}


def get_version():
    ver = str(_lcms2.get_version())
    return f"{ver[0]}.{ver[1:]}"


class Profile:
    def __init__(self, builtin=None, filename=None, buffer=None,
                 white_point=None, degrees=2, temperature=None):
        if builtin is None and filename is None and buffer is None:
            self._assign(profile=_lcms2.create_default_profile())
            return
        if builtin is not None and filename is None and buffer is None:
            self._assign(profile=create_profile(builtin, white_point=white_point, degrees=degrees, temperature=temperature))
            return
        if filename is not None and builtin is None and buffer is None:
            self._assign(profile=open_profile(filename))
            return
        if buffer is not None and builtin is None and filename is None:
            self._assign(profile=profile_from_memory(buffer))
            return
        raise CMSError(
            "Only one source of profile data can be specified: a built-in profile name, filename or bytes object")

    def _assign(self, profile):
        self.handle = profile.handle
        self.name = profile.name
        self.info = profile.info
        self.copyright = profile.copyright

    def to_bytes(self):
        return _lcms2.profile_to_bytes(self)

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(self.to_bytes())


class Transform:
    def __init__(self, src_profile, src_format, dst_profile, dst_format, intent="PERCEPTUAL", flags="NONE",
                 proofing_profile=None, proofing_intent=None):
        if not isinstance(src_profile, (Profile, _lcms2.Profile)):
            raise CMSError(f"Wrong type of src_profile. Expected Profile but got '{type(src_profile)}'")
        src_format_key = _DATA_TYPES_UPPER.get(src_format.upper())
        if src_format_key is None:
            raise CMSError(f"Invalid source data format: '{src_format}'")
        src_format = src_format_key
        if not isinstance(dst_profile, (Profile, _lcms2.Profile)):
            raise CMSError(f"Wrong type of dst_profile. Expected Profile but got '{type(src_profile)}'")
        dst_format_key = _DATA_TYPES_UPPER.get(dst_format.upper())
        if dst_format_key is None:
            raise CMSError(f"Invalid destination data format: '{dst_format}'")
        dst_format = dst_format_key
        intent_key = intent.upper()
        if intent_key not in INTENT:
            raise CMSError(f"Invalid rendering intent: '{intent}'")
        intent = intent_key
        flag_list = [f for f in re.split("[ ,;|]+", flags) if f]
        transform_flags = FLAG["NONE"]
        for f in flag_list:
            f = f.upper()
            if f not in FLAG:
                raise CMSError(f"Invalid flag: '{f}'")
            transform_flags = transform_flags | FLAG[f]

        if proofing_profile is not None:
            if not isinstance(proofing_profile, (Profile, _lcms2.Profile)):
                raise CMSError(f"Wrong type of proofing_profile. Expected Profile but got '{type(src_profile)}'")
            if proofing_intent not in NON_ICC_INTENT.keys():
                raise CMSError(f"Invalid proofing intent: '{proofing_intent}'")

        self.src_format = src_format
        self.dst_format = dst_format
        self.transform = _lcms2.Transform(src_profile, DATA_TYPES[src_format][0],
                                        dst_profile, DATA_TYPES[dst_format][0],
                                        INTENT[intent], transform_flags,
                                        proofing_profile,
                                        NON_ICC_INTENT.get(proofing_intent, 0))

    def apply(self, src):
        _, src_numpy_type, src_channels = DATA_TYPES[self.src_format]
        _, dst_numpy_type, dst_channels = DATA_TYPES[self.dst_format]
        x = src
        if not isinstance(src, np.ndarray):
            x = np.array(src, dtype=src_numpy_type)
        if x.dtype != src_numpy_type:
            raise CMSError(f"Source data type ({x.dtype}) does not match transform input type ({src_numpy_type}")
        if x.shape[-1] != src_channels:
            raise CMSError(f"Wrong number of input channels ({x.shape[-1]}); expected {src_channels}")
        dst_shape = np.empty_like(x.shape)
        dst_shape[:-1] = x.shape[:-1]
        dst_shape[-1] = dst_channels
        dst = np.empty(shape=dst_shape, dtype=dst_numpy_type)
        self.transform.apply(x, dst, x.size // src_channels)
        return dst
