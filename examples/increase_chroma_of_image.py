# %% [markdown]
# # Increase the chroma of an image
#
# This tutorial makes an image more colorful without changing its lightness
# or hues. It converts the image to CIE LCh, scales the chroma channel, and
# converts back to sRGB.
#
# CIE LCh describes a color with three channels:
#
# - `L`, lightness, from 0 (black) to 100 (white);
# - `C`, chroma, the distance from gray; 0 is gray and larger is more colorful;
# - `h`, hue, as a fraction of a full turn in [0, 1).
#
# Scaling `C` alone changes colorfulness and leaves lightness and hue as
# they are, which is hard to do directly on RGB values.

# %% [markdown]
# ## Setup
#
# Imports and two plotting helpers: `show` displays one image and `show2`
# displays two images side by side. `srgb` and `lch` are ready-made space
# instances from `lcms2.spaces`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from lcms2.spaces import lch, srgb


def show(img, title):
    plt.figure()
    plt.imshow(img)
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
# The astronaut test image as sRGB device values in [0, 1].

# %%
img = data.astronaut().astype(np.float64) / 255

show(img, "Original image")

# %% [markdown]
# ## Convert, scale chroma, convert back
#
# 1. `srgb.to(lch, img)` gives an array of the same shape holding `L`, `C`
#    and `h` for every pixel.
# 2. `mapped_img[..., 1] *= 1.5` multiplies channel 1, chroma, by 1.5.
# 3. `lch.to(srgb, mapped_img)` converts back to sRGB.
#
# lcms2 never clips values, so colors pushed outside the sRGB gamut come
# back below 0 or above 1. `.clip(0.0, 1.0)` limits them to the displayable
# range.

# %%
mapped_img = srgb.to(lch, img)
mapped_img[..., 1] *= 1.5

img2 = lch.to(srgb, mapped_img).clip(0.0, 1.0)
show(img2, "Chroma boosted image")

# %% [markdown]
# ## Compare
#
# The original and the result side by side: grays stay gray, since their
# chroma is 0, and colored regions become more saturated.

# %%
show2(img, img2, "Original", "Enhanced")
