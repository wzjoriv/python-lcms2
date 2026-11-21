"""
# Custom color spaces as ICC profiles

Write a color space as two NumPy functions, turn it into an ICC profile of version 2, 4 or 5, and read it back.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

import lcms2
from lcms2.icc import ICCSpace
from lcms2.spaces import ColorSpace

out = Path(os.environ.get("LCMS2_EXAMPLES_OUT") or tempfile.mkdtemp())

# %% [markdown]
# A color space is `to_xyz` and `from_xyz` on arrays shaped `(..., n_channels)`, with XYZ relative to D50.
# This one is an RGB space with gamma 1.8 and the sRGB primaries.

# %%
M = np.array(
    [[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169], [0.0139322, 0.0971045, 0.7141733]]
)
M_inv = np.linalg.inv(M)


class Gamma18(ColorSpace):
    n_channels = 3

    def to_xyz(self, rgb):
        return (np.asarray(rgb) ** 1.8) @ M.T

    def from_xyz(self, xyz):
        return np.clip(np.asarray(xyz) @ M_inv.T, 0, None) ** (1 / 1.8)


gamma18 = Gamma18()
rgb = np.array([[0.2, 0.5, 0.8], [0.9, 0.1, 0.3]])
print(gamma18.to("srgb", rgb).round(5))

# %% [markdown]
# `to_icc_profile` traces the functions and picks the most exact ICC representation. It returns the file
# bytes, the method chosen and the reasons. `version` is 2, 4 (default) or 5.

# %%
for version in (2, 4, 5):
    info = gamma18.to_icc_profile(version=version)
    print(version, info["method"], len(info["bytes"]), "bytes")
print(info["notes"])

# %% [markdown]
# `Profile(space)` (or `Profile.from_space`) builds a LittleCMS profile directly, ready for transforms and
# for saving. It matches the space to within the precision of a 16-bit ICC curve.

# %%
profile = lcms2.Profile(gamma18)
print(profile, profile.tags)
path = out / "gamma18.icc"
profile.save(path)
print(path.name)
print(lcms2.convert(rgb, profile, "sRGB").round(5))

# %% [markdown]
# Reading the file back gives the same colors. `ICCSpace` evaluates an ICC profile in pure NumPy, so it
# also reads version 5 profiles, which LittleCMS does not.

# %%
space = ICCSpace(path.read_bytes())
print(space)
print(space.to("srgb", rgb).round(5))
v5 = ICCSpace(gamma18.to_icc_profile(version=5)["bytes"])
print(v5.to("srgb", rgb).round(5))

# %% [markdown]
# A space that is not RGB: one channel holding log luminance. `device_domain` is the range of real device
# values; the profile stores it in a `crsD` tag and real values round-trip through it. Through LittleCMS
# use `intent="relative"` for exact XYZ: the default perceptual intent of v4 profiles shifts the black point.


# %%
class LogLuminance(ColorSpace):
    n_channels = 1
    device_domain = (np.log(1e-10), 0.0)

    def to_xyz(self, value):
        y = np.exp(value[..., 0])
        zero = np.zeros_like(y)
        return np.stack([zero, y, zero], axis=-1)

    def from_xyz(self, xyz):
        return np.log(np.maximum(xyz[..., 1:2], 1e-10))


log_lum = LogLuminance()
info = log_lum.to_icc_profile(version=5)
print(info["method"])
print(info["notes"][1])
values = np.array([[-8.0], [-2.0], [-0.5]])
print(log_lum.to("xyz-d50", values).round(5))
print(ICCSpace(info["bytes"]).to("xyz-d50", values).round(5))
print(lcms2.convert(values, lcms2.Profile(log_lum), "xyz-d50", intent="relative").round(5))

# %% [markdown]
# Built-in spaces write profiles too. OKLab cannot be a v2 or v4 matrix profile, so v4 gets a lookup table
# and v5 an exact calculator-style element. Passing `strict=True` raises `ICCWriteError` instead of falling
# back to a sampled table when nothing exact fits.

# %%
oklab = lcms2.spaces.oklab
for version in (4, 5):
    info = oklab.to_icc_profile(version=version)
    print(version, info["method"], len(info["bytes"]))
color = [0.7, 0.1, 0.05]
print(oklab.to("srgb", color).round(6))
print(ICCSpace(info["bytes"]).to("srgb", color).round(6))
