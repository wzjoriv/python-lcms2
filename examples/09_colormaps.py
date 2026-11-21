"""
# Colormaps

Build perceptually uniform colormaps, look values up in them, check them, simulate color blindness and save them.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

import lcms2
from lcms2 import Colormap

out = Path(os.environ.get("LCMS2_EXAMPLES_OUT") or tempfile.mkdtemp())

# %% [markdown]
# `Colormap.from_colors` interpolates between colors in OKLab, so equal steps in data look like equal steps
# in color. Colors are hex strings, CSS names, `rgb(...)` strings or numbers.

# %%
cmap = Colormap.from_colors(["#0b1a3a", "#f08a24", "#fff6d5"], name="dusk")
print(cmap)
print(cmap.hex()[:3], cmap.colors.shape)

# %% [markdown]
# `positions` places the colors unevenly; here the middle color sits a quarter of the way in. `space`
# changes the interpolation space and `n` the table size.

# %%
skewed = Colormap.from_colors(["#0b1a3a", "#f08a24", "#fff6d5"], positions=[0, 0.25, 1], n=8)
print(skewed.hex())
print(Colormap.from_colors(["black", "white"], space="srgb", n=5).hex())

# %% [markdown]
# Built-ins and generators: `Colormap.get(name)`, `lcms2.colormap.list()`, and `sequential`, `diverging`,
# `cyclic` and `qualitative` constructors.

# %%
print(lcms2.colormap.list())
maps = [
    Colormap.get("glacier"),
    Colormap.sequential(250),
    Colormap.diverging("#2b5fb4", "#b4352b"),
    Colormap.cyclic(),
    Colormap.qualitative(5),
]
for m in maps:
    print(f"{m.name:15s} monotonic L: {m.is_lightness_monotonic()!s:5s} step spread: {m.uniformity():.3f}")

# %% [markdown]
# Calling a colormap looks values up in an array of any shape. Values are scaled by `vmin` and `vmax` (0..1
# by default), `bytes=True` returns uint8, `alpha` adds a channel and `nan_color` replaces NaN.

# %%
values = np.array([0.0, 0.25, 0.5, 1.0, np.nan])
print(cmap(values, nan_color="#ff00ff").round(3))
print(cmap(values, bytes=True, alpha=0.8)[1])
print(cmap(np.array([[0, 5], [10, 20]]), vmin=0, vmax=10).shape)

# %% [markdown]
# Reverse, resample and slice to derive new maps.

# %%
print(cmap.reversed()(0.0).round(3))
print(cmap.resampled(16))
print(cmap[64:192])

# %% [markdown]
# `simulate` previews a map as seen with a color vision deficiency; `severity` goes from 0 to 1.

# %%
print(cmap.simulate("deuteranopia")(0.5).round(3))
print(cmap.simulate("protanopia", severity=0.5)(0.5).round(3))

# %% [markdown]
# Save to PNG, CSV, JSON or TXT and load a map back. `to_matplotlib()` converts to a matplotlib colormap
# and `register()` makes it available by name (matplotlib is optional).

# %%
for suffix in ("png", "csv", "json"):
    print(cmap.save(out / f"dusk.{suffix}").name)
print(Colormap.load(out / "dusk.json"))
try:
    mpl = cmap.to_matplotlib()
    print(type(mpl).__name__, mpl.name)
except ImportError:
    print("matplotlib is not installed")
