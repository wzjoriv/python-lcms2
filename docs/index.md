# python-lcms2

Color management for NumPy, built on [LittleCMS 2](https://github.com/mm2/Little-CMS).

```python
import lcms2

p3  = lcms2.convert(img, "sRGB", "Display P3")      # uint8 in, uint8 out
lab = lcms2.Profile("sRGB").transform(img, "Lab")    # float64 L*a*b*
lcms2.delta_e(lab_a, lab_b)                          # CIEDE2000, vectorized
lcms2.Profile(lcms2.spaces.oklab).save("oklab.icc")  # an ICC profile of OKLab
```

| You want to… | Read |
|---|---|
| Install and convert a first image | {doc}`start/quickstart` |
| Open, inspect, edit and create ICC profiles | {doc}`guides/profiles` |
| Control intents, proofing, chains and device links | {doc}`guides/transforms` |
| Mix ICC profiles with OKLab, HSV, PQ… | {doc}`guides/spaces` |
| Turn your own NumPy color math into an ICC profile | {doc}`guides/icc-authoring` |
| Measure color differences, CIECAM02, white points | {doc}`guides/colorimetry` |
| Build perceptually uniform colormaps | {doc}`guides/colormaps` |
| Read and write measurement (CGATS / IT8) files | {doc}`guides/cgats` |
| Know how dtypes and shapes become pixel formats | {doc}`explanation/formats` |

```{toctree}
:hidden:
:caption: Start

start/installation
start/quickstart
```

```{toctree}
:hidden:
:caption: Guides

guides/profiles
guides/transforms
guides/spaces
guides/icc-authoring
guides/colorimetry
guides/colormaps
guides/cgats
```

```{toctree}
:hidden:
:caption: Explanation

explanation/formats
explanation/architecture
```

```{toctree}
:hidden:
:caption: Reference

reference/api
contributing
```
