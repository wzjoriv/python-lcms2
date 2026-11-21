"""
# Colorimetry

Color differences, XYZ/Lab/LCh/xyY conversions, white points, chromatic adaptation and CIECAM02.
"""

import numpy as np

import lcms2
from lcms2 import colorimetry as cm

# %% [markdown]
# `delta_e(lab1, lab2, method=...)` measures color differences. It is vectorized over `(..., 3)` arrays
# and broadcasts. The default is CIEDE2000; `76`, `94`, `cmc` and `bfd` are also available.

# %%
pair = ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485])
print("2000", lcms2.delta_e(*pair).round(4))
for method in ("76", "94", "bfd"):
    print(method.ljust(4), lcms2.delta_e(*pair, method=method).round(4))
print("cmc ", lcms2.delta_e(*pair, method="cmc", l=2, c=1).round(4))
print("2000 Kc=2", lcms2.delta_e(*pair, Kc=2).round(4))

# %% [markdown]
# One reference against many colors, converted to Lab first.

# %%
rng = np.random.default_rng(0)
palette = rng.integers(0, 256, (5, 3), dtype=np.uint8)
lab = lcms2.convert(palette, "sRGB", "Lab")
print(lcms2.delta_e(lab, lab[0]).round(2))

# %% [markdown]
# XYZ, xyY, Lab and LCh conversions, each with its inverse. `xyz_to_lab` takes an optional white (default D50). Hue in `lab_to_lch` is in degrees.

# %%
xyz = np.array([0.2, 0.3, 0.4])
lab = cm.xyz_to_lab(xyz)
print(lab.round(3), cm.lab_to_xyz(lab).round(6))
print(cm.lab_to_lch(lab).round(3), cm.xyz_to_xyy(xyz).round(4))
print(cm.xyz_to_lab(xyz, cm.D65).round(3))
print(cm.D50, cm.D65)

# %% [markdown]
# White points: named illuminants live in `lcms2.WHITE_POINTS` (per observer angle), and color temperature
# converts to and from a white point in xyY.

# %%
print(lcms2.WHITE_POINTS[2]["D65"])
white = cm.white_point_from_temperature(5000)
print(white.round(4), round(float(cm.temperature_from_white_point(white))))
print(round(float(cm.temperature_from_white_point(cm.D65_xyY))))

# %% [markdown]
# `adapt` moves XYZ between white points with Bradford (default) or another adaptation matrix.

# %%
d65_white = np.array([0.95047, 1.0, 1.08883])
color = np.array([0.4, 0.3, 0.2])
print(cm.adapt(d65_white, cm.D65, cm.D50).round(4))
for method in ("bradford", "von_kries", "xyz_scaling"):
    print(method.ljust(11), cm.adapt(color, cm.D65, cm.D50, method=method).round(4))
print(cm.adaptation_matrix(cm.D65, cm.D50).round(4))

# %% [markdown]
# CIECAM02 maps XYZ to lightness, chroma and hue for a viewing condition, and back.

# %%
model = lcms2.CIECAM02(white_point=d65_white * 100, adapting_luminance=64, background=20, surround="average")
jch = model.forward(np.array([19.01, 20.0, 21.78]))
print(jch.round(3))
print(model.reverse(jch).round(3))
