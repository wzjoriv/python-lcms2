"""
# Transforms and images

Reuse a Transform on uint8, uint16 and float images, carry alpha, and control intents and threads.
"""

import numpy as np

import lcms2

# %% [markdown]
# A small seeded gradient plus noise stands in for a photo: an (H, W, 3) uint8 sRGB array.

# %%
rng = np.random.default_rng(0)
ramp = np.linspace(0, 255, 64)
img = np.stack(np.broadcast_arrays(ramp[None, :], ramp[:, None], 128.0), axis=-1)
img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
print(img.shape, img.dtype)

# %% [markdown]
# A `Transform` is built once and called many times. The pixel formats are resolved from each input and
# cached, so the same object takes uint8, uint16 and float arrays.

# %%
to_p3 = lcms2.Transform("sRGB", "Display P3")
print(to_p3(img)[0, -1], to_p3(img).dtype)
print(to_p3(img.astype(np.uint16) * 257)[0, -1], to_p3(img.astype(np.uint16) * 257).dtype)
print(to_p3(img.astype(np.float32) / 255)[0, -1].round(4))

# %% [markdown]
# An extra last channel is alpha and is copied through unchanged. `out=` writes into an array you own.

# %%
rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 200, np.uint8)], axis=-1)
result = to_p3(rgba)
print(result.shape, result[0, 0], rgba[0, 0])
buffer = np.empty_like(img)
to_p3(img, out=buffer)
print(np.array_equal(buffer, to_p3(img)))

# %% [markdown]
# `intent` takes a name or an `lcms2.Intent`. Matrix/TRC profiles render every intent alike; intents matter
# for CLUT profiles such as printers. `bpc=True` scales the source black point to the target's, shown here
# with a profile whose black is lifted to 5% of the signal.

# %%
print([intent.name for intent in lcms2.Intent][:4])
lifted = lcms2.Profile.rgb("D50", gamma=lcms2.ToneCurve.tabulated(np.linspace(0.05, 0.9, 32)))
print("black point", lifted.black_point("relative").round(3))
black = np.array([0, 0, 0], np.uint8)
for bpc in (False, True):
    print("bpc", bpc, lcms2.convert(black, lifted, "sRGB", intent="relative", bpc=bpc))

# %% [markdown]
# A list of profiles is one multi-profile transform, here through a contrast-boosting abstract profile.

# %%
contrast = lcms2.Profile.abstract(contrast=1.3)
chain = lcms2.Transform(["sRGB", contrast, "Display P3"])
print(chain(np.array([[100, 120, 140]], np.uint8)))
print(to_p3(np.array([[100, 120, 140]], np.uint8)))

# %% [markdown]
# Explicit LittleCMS formats replace the automatic ones; `lcms2.FORMATS` has all 178.

# %%
t16 = lcms2.Transform("sRGB", "RGB_8", "Display P3", "RGB_16")
print(t16.input_format, t16.output_format)
print(t16(img)[0, -1])
print(list(lcms2.FORMATS)[:5])

# %% [markdown]
# Large inputs are split across a shared thread pool (262,144 pixels or more). The result does not depend
# on the thread count; `set_threads(1)` turns the pool off, for example inside your own pool.

# %%
big = np.tile(img, (64, 64, 1))
print(big.shape)
default = lcms2.get_threads()
many = to_p3(big)
lcms2.set_threads(1)
single = to_p3(big)
lcms2.set_threads(default)
print(np.array_equal(many, single), lcms2.get_threads() == default)
