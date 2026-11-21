"""
# Color spaces

Convert between OKLab, OKLCh, HSV, Display P3, Rec.2100 and more with lcms2.spaces, mixed with ICC profiles.
"""

import numpy as np

import lcms2
from lcms2 import spaces

# %% [markdown]
# `lcms2.spaces` has color spaces defined as NumPy functions. `ColorSpace.list()` shows the registered
# names; `spaces.get(name)` returns one, and the ready-made instances are module attributes.

# %%
print(spaces.ColorSpace.list())
print(spaces.get("oklab"), spaces.oklab.n_channels, spaces.oklab.device_domain)

# %% [markdown]
# `space.to(target, values)` converts, with the target a name or a space. Values are shaped
# `(..., n_channels)` and in the space's own range. Hue channels (HSV, HWB, LCh, OKLCh) are in degrees, 0..360, as in CSS.

# %%
orange = np.array([1.0, 0.5, 0.0])
for name in ("hsv", "hwb", "oklab", "oklch", "display-p3", "rec2100-pq", "xyz-d65"):
    print(f"{name:11s}", spaces.srgb.to(name, orange).round(4))

# %% [markdown]
# Spaces round-trip, and every conversion goes through XYZ D50, available as `to_xyz` and `from_xyz`.

# %%
hsv = spaces.srgb.to("hsv", orange)
print(spaces.hsv.to("srgb", hsv).round(6))
xyz = spaces.srgb.to_xyz(orange)
print(xyz.round(4), spaces.srgb.from_xyz(xyz).round(6))

# %% [markdown]
# Spaces and ICC profiles mix freely in `lcms2.convert` and in `ColorSpace.to`. When a space is involved
# the conversion runs through XYZ in float64, so intents do not apply.

# %%
print(lcms2.convert([0.7, 0.1, 0.05], "oklab", "Display P3").round(4))
print(spaces.oklch.to("Adobe RGB (1998)", [0.7, 0.15, 0.3]).round(4))
print(lcms2.convert(np.array([200, 100, 50], np.uint8), "sRGB", "oklch").round(4))

# %% [markdown]
# Space and profile link up: `space.profile` is a cached ICC profile of the space, and `profile.space`
# is a space that evaluates a profile.

# %%
print(spaces.oklab.profile)
print(lcms2.Profile("Display P3").space)

# %% [markdown]
# `RGBSpace` is the shortcut for RGB spaces: set a `matrix` (linear RGB to XYZ D50) and optional
# `decode`/`encode` curves. Subclasses register themselves under their name.


# %%
class Linear709(spaces.RGBSpace):
    matrix = spaces.srgb.matrix


print(Linear709().to("srgb", [0.5, 0.5, 0.5]).round(4))
print("linear709" in spaces.ColorSpace.list(), [n for n in spaces.ColorSpace.list() if "709" in n])
