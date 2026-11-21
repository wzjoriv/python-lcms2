"""
# Tone curves

Build, evaluate, invert and compose ToneCurve objects, and use them to make RGB and gray profiles.
"""

import numpy as np

import lcms2
from lcms2 import ToneCurve

x = np.linspace(0, 1, 5)

# %% [markdown]
# Curves come from a gamma, a parametric formula, a table of values or a standard (`srgb`, `rec709`). A
# curve is called on floats in 0..1 (or on uint16 data).

# %%
gamma = ToneCurve.gamma(2.2)
srgb = ToneCurve.srgb()
print(gamma)
print(gamma(x).round(4))
print(srgb(x).round(4))

# %% [markdown]
# `parametric(type, params)` takes the ICC types; type 4 with these five numbers is the sRGB curve.
# `tabulated` interpolates between the samples you give.

# %%
formula = ToneCurve.parametric(4, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
print(np.abs(formula(x) - srgb(x)).max().round(6))
table = ToneCurve.tabulated([0, 0.1, 0.5, 1])
print(table, table(x).round(4))

# %% [markdown]
# Inspect a curve: gamma estimate, shape flags and the sampled table (uint16). `estimate_gamma` raises when
# the curve is not close enough to a power law for `precision` (default 0.01), so loosen it for sRGB.

# %%
print(round(gamma.estimate_gamma(), 3), round(srgb.estimate_gamma(precision=0.5), 3))
print(gamma.is_monotonic, gamma.is_descending, gamma.is_linear, ToneCurve.linear().is_linear)
print(len(gamma.table), gamma.table[[0, 2048, 4095]])

# %% [markdown]
# `inverse()` undoes a curve. `a.then(b)` computes `b(a(x))` and `a.join(b)` computes `b.inverse()(a(x))`;
# both are sampled at 4096 points, so results carry a small sampling error.

# %%
print(gamma.inverse()(gamma(x)).round(4))
print(ToneCurve.gamma(2).then(ToneCurve.gamma(1.5))(x).round(4))
print(gamma.join(gamma)(x).round(3))

# %% [markdown]
# Curves build profiles: `gamma=` on `Profile.rgb` and `Profile.gray` takes a number, a `ToneCurve`, a
# `(type, params)` tuple, or one of these per channel.

# %%
rgb = lcms2.Profile.rgb("D65", gamma=srgb, description="Curve RGB")
gray = lcms2.Profile.gray("D50", gamma=gamma)
print(rgb.trc["gTRC"], gray)
print(lcms2.convert([0.5, 0.5, 0.5], rgb, "srgb-linear").round(4))

# %% [markdown]
# A linearization profile applies per-channel curves as a device link. A device link is used as a
# one-element chain: `Transform([link])`.

# %%
link = lcms2.Profile.linearization([gamma] * 3, color_space="RGB")
print(link, link.device_class)
print(lcms2.Transform([link])(np.array([[0.0, 0.5, 1.0]])).round(4))
