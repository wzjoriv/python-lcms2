"""
# Quick conversions

Convert single colors, CSS strings and arrays between profiles and color spaces with lcms2.convert.
"""

import numpy as np

import lcms2

# %% [markdown]
# `lcms2.convert(data, source, target)` takes profile or color space names. A list of integers is read as
# 8-bit RGB; the result is float for Lab, XYZ and the analytic spaces.

# %%
red = lcms2.convert([255, 0, 0], "sRGB", "Lab", intent="relative")
print(red.round(2))
print(lcms2.convert([1.0, 0.0, 0.0], "sRGB", "oklab").round(4))

# %% [markdown]
# Colors can be CSS strings. `parse_color` returns floats in 0..1 and `to_hex` goes back.

# %%
rgb = lcms2.parse_color("rebeccapurple")
print(rgb.round(3), lcms2.to_hex(rgb))
print(lcms2.convert(rgb, "sRGB", "Display P3").round(4))

# %% [markdown]
# Arrays keep their shape: the last axis is the channels. The input dtype picks the pixel format and the
# output keeps it, so uint8 in gives uint8 out.

# %%
rng = np.random.default_rng(0)
img = rng.integers(0, 256, (2, 3, 3), dtype=np.uint8)
p3 = lcms2.convert(img, "sRGB", "Display P3")
print(img.dtype, "->", p3.dtype, p3.shape)
print(p3[0, 0])

# %% [markdown]
# `out_dtype` asks for another type. Integer input converted to Lab or XYZ comes back as float64 on its own.

# %%
print(lcms2.convert(img, "sRGB", "Display P3", out_dtype=np.float32)[0, 0].round(4))

# %%
lab = lcms2.convert(img, "sRGB", "Lab")
print(lab.dtype)
print(lab[0, 0].round(2))

# %% [markdown]
# Float and 16-bit data work the same way. Here 16-bit values go to linear light through a color space.

# %%
img16 = img.astype(np.uint16) * 257
print(lcms2.convert(img16, "sRGB", "srgb-linear")[0, 0].round(4))

# %% [markdown]
# Names are case-insensitive. `Profile.builtins()` lists the profiles and `ColorSpace.list()` the spaces.

# %%
print(lcms2.Profile.builtins()[:6])
print(lcms2.spaces.ColorSpace.list()[:6])
print(lcms2.get_version())
