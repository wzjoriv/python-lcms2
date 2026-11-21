"""
# ICC profiles

Open, inspect, edit, create and save ICC profiles.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

import lcms2

out = Path(os.environ.get("LCMS2_EXAMPLES_OUT") or tempfile.mkdtemp())

# %% [markdown]
# `Profile(name)` opens a built-in profile; names are case-insensitive. `Profile.builtins()` lists them.

# %%
srgb = lcms2.Profile("sRGB")
print(len(lcms2.Profile.builtins()), "built-ins")
print(srgb)

# %% [markdown]
# The header and the decoded tags are properties.

# %%
print(srgb.color_space, srgb.pcs, srgb.device_class, srgb.channels, srgb.version)
print(srgb.tags)
print(srgb.is_matrix_shaper, [intent.name for intent in srgb.supported_intents])
print(srgb.media_white_point.round(4))

# %% [markdown]
# Matrix/TRC profiles expose their colorants, one XYZ row per channel, and their tone curves.

# %%
print(srgb.colorants.round(4))
print(srgb.trc["rTRC"])
print(srgb.trc["rTRC"]([0.0, 0.5, 1.0]).round(4))

# %% [markdown]
# Text fields are settable. `save` writes a file and `Profile.open` reads it back; `to_bytes` and
# `from_bytes` do the same in memory, for example for a profile embedded in an image.

# %%
custom = lcms2.Profile("Display P3")
custom.description = "My display"
custom.copyright = "Public domain"
custom.compute_id()
path = out / "my_display.icc"
custom.save(path)
print(path.name, path.stat().st_size, "bytes")

reopened = lcms2.Profile.open(path)
print(reopened.description, reopened.profile_id == custom.profile_id)
print(lcms2.Profile.from_bytes(custom.to_bytes()).description)

# %% [markdown]
# Create profiles from numbers: primaries and a gamma for RGB, or a white point and gamma for gray.
# White points are illuminant names, a `temperature=`, or xyY.

# %%
wide = lcms2.Profile.rgb("D65", ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), 2.2, description="Wide RGB")
gray = lcms2.Profile.gray("D65", 2.2)
print(wide, gray)
print(wide.colorants.round(4))

# %% [markdown]
# `Profile.transform` converts pixels from this profile to another profile or color space name.

# %%
rgb = np.array([[255, 128, 0], [10, 200, 30]], np.uint8)
print(wide.transform(rgb, "sRGB"))
print(srgb.transform(rgb, "Lab").round(2))

# %% [markdown]
# The Lab and XYZ virtual profiles and the sRGB one are also constructors.

# %%
print(lcms2.Profile.lab(white_point="D65", version=4))
print(lcms2.Profile.xyz())
print(lcms2.Profile.srgb().description)
