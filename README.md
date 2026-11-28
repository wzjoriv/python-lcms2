# python-lcms2

Color management for NumPy. Convert images between ICC profiles and color spaces in one call, measure color
differences, build exact ICC profiles from your own NumPy color math, and make perceptually uniform colormaps,
all backed by [LittleCMS 2](https://github.com/mm2/Little-CMS).

[![Tests](https://github.com/wzjoriv/python-lcms2/actions/workflows/ci.yml/badge.svg)](https://github.com/wzjoriv/python-lcms2/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-lcms2)](https://pypi.org/project/python-lcms2/)
[![Python](https://img.shields.io/pypi/pyversions/python-lcms2)](https://pypi.org/project/python-lcms2/)
[![Free-threaded](https://img.shields.io/badge/free--threaded-3.14t%20%7C%203.15t-blue)](https://pypi.org/project/python-lcms2/#files)
[![PyPy](https://img.shields.io/badge/PyPy-3.11-blue)](https://pypi.org/project/python-lcms2/#files)
[![Pyodide](https://img.shields.io/badge/Pyodide-3.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/python-lcms2/#files)
[![LittleCMS 2.20](https://img.shields.io/badge/LittleCMS-2.20-blue)](https://github.com/mm2/Little-CMS)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-wzjoriv.github.io%2Fpython--lcms2-blue)](https://wzjoriv.github.io/python-lcms2)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

| Feature | What you get |
|---|---|
| One-call conversion | `lcms2.convert(img, "sRGB", "Display P3")`: pixel formats come from the array's dtype and shape. |
| Profiles | 24 built-ins (sRGB, Display P3, Adobe RGB, ProPhoto, Rec.2020, Lab, XYZ, …), ICC files, bytes; read and edit header, text and tags. |
| Transforms | Rendering intents, black-point compensation, soft proofing with gamut check, multi-profile chains, device links. |
| Fast and safe | The GIL is released during transforms; any contiguity or stride; alpha channels carried through. |
| Color spaces | OKLab, OKLCh, CIE LCh, HSV, HWB, Display P3, Rec.2100 PQ/HLG, … as NumPy functions, mixed freely with ICC profiles. |
| ICC authoring | `lcms2.Profile(my_space)` traces your NumPy `to_xyz`/`from_xyz` into an exact ICC v4/v5 profile. |
| Colorimetry | ΔE 76/94/2000/CMC/BFD, CIECAM02, white point ⇄ color temperature, chromatic adaptation. |
| Tone curves | Gamma, parametric and tabulated curves; evaluate, invert, join, estimate gamma. |
| Colormaps | Perceptually uniform colormaps in OKLab, gamut-mapped, color-vision-deficiency simulation, matplotlib export. |
| Measurement data | Read and write CGATS / IT8 files. |

Wheels for Windows, Linux and macOS (x86-64 and ARM64) on Python 3.10 and later, including free-threaded 3.14t and 3.15t, and PyPy 3.11; Pyodide wheels on PyPI. NumPy is the only dependency.

## Install

```bash
pip install python-lcms2
pip install "python-lcms2[plot]"      # + matplotlib for colormap export
```

## Use

```python
import numpy as np
import lcms2

img = np.asarray(...)                                  # (H, W, 3) uint8, e.g. from Pillow or OpenCV

p3  = lcms2.convert(img, "sRGB", "Display P3")         # uint8 in, uint8 out
lab = lcms2.Profile("sRGB").transform(img, "Lab")      # float64 L*a*b*

lcms2.convert([255, 0, 0], "sRGB", "Lab", intent="relative")
# array([54.29, 80.81, 69.89])
lcms2.convert([255, 0, 0], "sRGB", "oklab")
# array([0.6279, 0.2249, 0.1258])
```

Reuse a `Transform` when converting many images; formats are resolved per input and cached. `press.icc` stands
for any CMYK profile, e.g. from your printer or from [ECI](https://www.eci.org/en/downloads):

```python
to_cmyk = lcms2.Transform("sRGB", lcms2.Profile("press.icc"), intent="relative", bpc=True)
cmyk = to_cmyk(img)                                    # (H, W, 4) uint8
```

### Profiles

```python
srgb = lcms2.Profile("sRGB")                  # built-in (case-insensitive: "display p3", "adobe rgb", …)
cmyk = lcms2.Profile("press.icc")             # a file
web  = lcms2.Profile.from_bytes(icc_bytes)    # bytes, e.g. embedded in an image

cmyk                                          # <Profile 'Fogra27L CMYK Coated Press' CMYK output v2.2>
cmyk.color_space, cmyk.device_class, cmyk.tags
cmyk.description = "Press proof"
cmyk.save("press-v2.icc")
```

### Soft proofing

```python
proof = lcms2.Transform(srgb, srgb, proof=cmyk, gamut_check=True)
proof(img)                                    # how img prints; out-of-gamut pixels marked gray
```

### Your own color space, as an ICC profile

```python
from lcms2.spaces import ColorSpace

M = np.array([[0.4360747, 0.3850649, 0.1430804],   # sRGB primaries → XYZ (D50)
              [0.2225045, 0.7168786, 0.0606169],
              [0.0139322, 0.0971045, 0.7141733]])
M_inv = np.linalg.inv(M)

class Gamma18(ColorSpace):
    n_channels = 3
    def to_xyz(self, rgb):
        return np.asarray(rgb) ** 1.8 @ M.T
    def from_xyz(self, xyz):
        return np.clip(np.asarray(xyz) @ M_inv.T, 0, None) ** (1 / 1.8)

profile = lcms2.Profile(Gamma18())            # exact matrix/TRC profile, usable in any CMS
profile.save("gamma18.icc")
lcms2.convert(img, profile, "sRGB")
```

Built-in analytic spaces (`lcms2.spaces.oklab`, `oklch`, `hsv`, `display_p3`, `rec2100_pq`, …) work the same
way: `lcms2.spaces.oklab.profile` is an ICC profile of OKLab.

### Color differences

```python
lcms2.delta_e(lab1, lab2)                     # CIEDE2000, vectorized over (..., 3)
lcms2.delta_e(lab1, lab2, method="cmc", l=2, c=1)
```

### Colormaps

```python
import matplotlib.pyplot as plt

cm = lcms2.Colormap.from_colors(["#0b1a3a", "#f08a24", "#fff6d5"])   # interpolated in OKLab
cm = lcms2.Colormap.get("glacier")            # built-ins: lcms2.colormap.list()

values = np.linspace(0, 1, 256)
rgb = cm(values)                              # (256, 3) floats; bytes=True for uint8
cm.is_lightness_monotonic(), cm.uniformity()
cm.simulate("deuteranopia")                   # color-vision-deficiency preview
plt.imshow(np.random.rand(64, 64), cmap=cm.to_matplotlib())
```

## Examples

A numbered learning path from first conversions to ICC authoring, each as a script and a notebook with outputs: [examples/](examples/README.md).

## Documentation

Guides and the API reference: <https://wzjoriv.github.io/python-lcms2>.

## Contributing

Building from source and the developer commands are in [docs/contributing.rst](docs/contributing.rst).

## License

MIT. LittleCMS is © Marti Maria Saguer, MIT license (`vendor/Little-CMS/LICENSE`).
