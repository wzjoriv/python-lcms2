# %% [markdown]
# # Create, save and load a color space
#
# This tutorial describes how to define a new color space, convert an image
# into it, save the space as an ICC profile, load the profile back, and check
# that the loaded profile gives the same result as the original definition.
#
# A color space is two functions:
#
# - `to_xyz` maps device values, shaped `(..., n_channels)`, to CIE XYZ
#   relative to D50, shaped `(..., 3)`.
# - `from_xyz` maps CIE XYZ back to device values.
#
# Every conversion between two spaces goes through XYZ:
# `source.to(target, x)` computes `target.from_xyz(source.to_xyz(x))`.

# %% [markdown]
# ## Setup
#
# Imports and three small plotting helpers: `show` displays an RGB image,
# `colormap_show` displays a single channel with a color bar, and `show2`
# displays two images side by side.

# %%
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from lcms2.spaces import ColorSpace, ICCSpace, srgb


def show(img, title):
    plt.figure()
    plt.imshow(img)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.show()


def colormap_show(img, title):
    plt.figure()
    im = plt.imshow(img, cmap="grey")
    plt.colorbar(im)
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.show()


def show2(img1, img2, title1, title2):
    fig, axs = plt.subplots(ncols=2, layout="constrained")
    axs[0].imshow(img1)
    axs[1].imshow(img2)

    axs[0].set_title(title1)
    axs[1].set_title(title2)

    axs[0].axis("off")
    axs[1].axis("off")
    plt.show()


# %% [markdown]
# ## Load an image
#
# The astronaut test image comes as 8-bit sRGB. Dividing by 255 gives sRGB
# device values in [0, 1], the range the built-in `srgb` space expects.

# %%
img = data.astronaut().astype(np.float64) / 255

show(img, "Original image")


# %% [markdown]
# ## Define a color space
#
# `LogLuminance` has one channel: the natural logarithm of luminance, `Y`.
#
# - `n_channels = 1` declares the number of device channels.
# - `device_domain` is the range of device values an ICC profile covers.
#   ICC profiles store device values in [0, 1]; the range is mapped onto
#   [0, 1] when saving and back when loading. Here it spans from black,
#   `log(1e-10)`, to white, `log(1) = 0`.
# - `to_xyz` exponentiates the channel to get `Y` and returns XYZ with `X`
#   and `Z` set to zero. `luminance[..., 0]` selects the channel, so the
#   result has shape `(..., 3)`.
# - `from_xyz` takes the logarithm of `Y`. The `1e-10` keeps black finite,
#   so black maps to `log(1e-10)`, about -23.
#
# Defining the subclass registers it by name, so `ColorSpace.list()` shows
# it and `.to("logluminance", ...)` finds it.


# %%
class LogLuminance(ColorSpace):
    n_channels = 1
    device_domain = (np.log(1e-10), 0.0)

    def to_xyz(self, luminance):
        y = np.exp(luminance[..., 0])
        return np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=-1)

    def from_xyz(self, xyz):
        return np.log(xyz[..., 1:2] + 1e-10)


loglum = LogLuminance()

# %% [markdown]
# ## Convert the image into the new space
#
# `srgb.to(loglum, img)` converts every pixel from sRGB to XYZ, then from
# XYZ to log luminance. The result has one channel per pixel.

# %%
img2 = srgb.to(loglum, img)

colormap_show(img2[:, :, 0], "Luminance - Our Space")

# %% [markdown]
# ## Save the space as an ICC profile
#
# `to_icc_profile` writes the space to a file (ICC version 4 by
# default) and returns a dictionary. `result["method"]` names what was
# written:
#
# - `exact_*` methods store the conversion itself: curves, matrices,
#   lookup tables that reproduce it exactly, or, in version 5, calculator
#   programs.
# - `clut_fallback` stores a sampled lookup table, used when no exact form
#   exists for the chosen version.

# %%
icc_path = os.path.join(tempfile.gettempdir(), "loglum.icc")
result = loglum.to_icc_profile(icc_path)
print(result["method"])

# %% [markdown]
# ## Load the profile and convert with it
#
# `ICCSpace(icc_path)` reads the saved file back as a color space. It
# works with `.to()` like any other space and knows nothing about the
# `LogLuminance` class: everything it does comes from the file. Converting
# the image through it repeats the earlier conversion, now using only the
# profile.

# %%
loglum2 = ICCSpace(icc_path)

img3 = srgb.to(loglum2, img)

colormap_show(img3[:, :, 0], "Luminance - From ICC")

# %% [markdown]
# ## Compare
#
# The mean absolute difference between the two results, in log-luminance
# units. Version 4 profiles store 16-bit values, so a small difference
# remains, mostly in the darkest pixels; `to_icc_profile(..., version=5)`
# stores 32-bit floats and brings it close to zero.

# %%
np.mean(np.abs(img2 - img3))
