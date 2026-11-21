"""
# Soft proofing and gamut checks

Preview how colors look on a print-like device with proof=, and mark colors a device cannot show with gamut_check.
"""

import numpy as np

import lcms2

# %% [markdown]
# Soft proofing simulates an output device on your display. Here sRGB colors go through a CMYK profile
# (the library's formula-based `lcms2.spaces.cmyk`) and back, so the result is still sRGB data but with
# what the device would print. `proof_intent` sets the intent of the trip to the device.

# %%
press = lcms2.spaces.cmyk.profile
colors = np.array([[0, 255, 0], [255, 0, 0], [128, 128, 128], [230, 120, 60]], np.uint8)
proof = lcms2.Transform("sRGB", "sRGB", proof=press, proof_intent="relative")
print(colors)
print(proof(colors))

# %% [markdown]
# The CMYK values that proof stands for come from an ordinary transform to the device profile.

# %%
to_press = lcms2.Transform("sRGB", press, intent="relative", bpc=True)
print(to_press(colors))

# %% [markdown]
# `gamut_check=True` replaces out-of-gamut pixels with the alarm color, gray by default. Display P3 colors
# checked against sRGB: the saturated red and green do not fit.

# %%
p3 = np.array([[255, 0, 0], [0, 255, 0], [128, 128, 128], [230, 120, 60]], np.uint8)
check = lcms2.Transform("Display P3", "Display P3", proof="sRGB", gamut_check=True)
marked = check(p3)
print(marked)
print("out of gamut:", (marked != p3).any(axis=-1))

# %% [markdown]
# `set_alarm_codes` changes the marker; the codes are 16-bit values per channel, global, and used by
# transforms created after the call.

# %%
print(lcms2.get_alarm_codes()[:3])
lcms2.set_alarm_codes([65535, 0, 65535])
magenta = lcms2.Transform("Display P3", "Display P3", proof="sRGB", gamut_check=True)
print(magenta(p3))
lcms2.set_alarm_codes([32512, 32512, 32512])

# %% [markdown]
# `lcms2.convert` and `Profile.transform` accept the same `proof=` and `gamut_check=` options, and float
# data keeps full precision.

# %%
wide = np.array([[0.9, 0.1, 0.1], [0.4, 0.4, 0.4]])
print(lcms2.convert(wide, "ProPhoto RGB", "ProPhoto RGB", proof="Display P3", gamut_check=True).round(3))
print(lcms2.Profile("Display P3").transform(p3, "Display P3", proof="sRGB", gamut_check=True)[:2])
