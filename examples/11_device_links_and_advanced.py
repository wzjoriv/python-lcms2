"""
# Device links and advanced options

Bake a transform into a device link, limit ink, adjust colors with abstract profiles and tune transform flags.
"""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

import lcms2

out = Path(os.environ.get("LCMS2_EXAMPLES_OUT") or tempfile.mkdtemp())
img = np.array([[10, 100, 200], [250, 30, 90]], np.uint8)

# %% [markdown]
# `Transform.to_device_link()` bakes a whole chain into one profile of class `link`. Save it for any CMS,
# or read it back and run it as a one-element chain: `Transform([link])`.

# %%
chain = lcms2.Transform(["sRGB", lcms2.Profile.abstract(contrast=1.2), "Display P3"], intent="relative")
link = chain.to_device_link()
print(link, link.device_class.name)
link.save(out / "warm_p3.icc")
reloaded = lcms2.Profile.open(out / "warm_p3.icc")
print(lcms2.Transform([reloaded])(img))
print(chain(img))

# %% [markdown]
# `Profile.abstract` is a Lab to Lab adjustment (brightness, contrast, hue, saturation) that sits between
# two profiles in a chain.

# %%
vivid = lcms2.Profile.abstract(saturation=20)
print(vivid.device_class.name, vivid.color_space.name)
print(lcms2.Transform(["sRGB", vivid, "sRGB"])(img))

# %% [markdown]
# `Profile.ink_limiting(limit)` is a CMYK device link capping total ink (percent, 0..400). Float CMYK data
# is in percent, 0..100 per channel, unlike float RGB which is 0..1.

# %%
limiter = lcms2.Transform([lcms2.Profile.ink_limiting(200)])
ink = np.array([[100.0, 100, 100, 100], [50, 50, 50, 50], [10, 20, 30, 0]])
limited = limiter(ink)
print(limited.round(1))
print(limited.sum(axis=-1).round(1))

# %% [markdown]
# `optimize`, `precision` and `flags` tune how LittleCMS builds the transform. `lcms2.Flag` members combine
# with `|`; strings such as `"NOOPTIMIZE|HIGHRESPRECALC"` work too. Results agree to a digit or two.

# %%
fast = lcms2.convert(img, "sRGB", "Display P3")
exact = lcms2.convert(img, "sRGB", "Display P3", optimize=False, precision="high")
flagged = lcms2.Transform("sRGB", "Display P3", flags=lcms2.Flag.NOOPTIMIZE | lcms2.Flag.HIGHRESPRECALC)(img)
print(np.abs(fast.astype(int) - exact).max(), np.abs(exact.astype(int) - flagged).max())
print([flag.name for flag in lcms2.Flag][:6])

# %% [markdown]
# In a chain `bpc` takes one entry per profile. The preserve-K intents keep the black channel through
# CMYK conversions.

# %%
steps = lcms2.Transform(["sRGB", "Lab", "Display P3"], intent="relative", bpc=[False, True, False])
print(steps(img))
print([intent.name for intent in lcms2.Intent][4:7])

# %% [markdown]
# Header fields and raw tags are editable, and profiles pickle (as bytes) so they cross process boundaries.

# %%
profile = lcms2.Profile("sRGB")
profile.set_header(rendering_intent=1)
print(profile.rendering_intent, profile.has_tag("wtpt"), len(profile.read_tag_raw("wtpt")), "bytes")
copy = pickle.loads(pickle.dumps(profile))
print(copy, copy.rendering_intent)
print(profile.named_colors, profile.tac)

# %% [markdown]
# `set_adaptation_state(value)` sets how fully absolute colorimetric conversions adapt to the viewing
# white (0..1, process-wide) and returns the previous value.

# %%
previous = lcms2.set_adaptation_state(0.5)
print(previous, lcms2.set_adaptation_state(previous))
