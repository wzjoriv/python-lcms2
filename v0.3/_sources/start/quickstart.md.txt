# Quickstart

## Convert an image

```python
import numpy as np
from PIL import Image
import lcms2

img = np.asarray(Image.open("photo.jpg"))          # (H, W, 3) uint8, sRGB

p3 = lcms2.convert(img, "sRGB", "Display P3")      # (H, W, 3) uint8
```

There are no format strings: the input dtype (`uint8`, `uint16`, `float16`, `float32`, `float64`) and the
last axis (channels, plus one optional alpha channel) decide the pixel format. The output keeps the input
dtype, except that integer data converted to Lab or XYZ comes back as `float64`. Ask for another with
`out_dtype=`. See {doc}`../explanation/formats`.

## Profiles

```python
srgb = lcms2.Profile("sRGB")                  # built-in name, case-insensitive
cmyk = lcms2.Profile("press.icc")             # file
lcms2.Profile.builtins()                      # the 25 built-in names
```

`Profile.transform` converts from that profile to another:

```python
lab = srgb.transform(img, "Lab")              # float64 L*a*b* (D50)
```

## Reuse a transform

`lcms2.convert` caches recent transforms. For explicit control, create a `Transform` once and call it:

```python
to_print = lcms2.Transform(srgb, cmyk, intent="relative", bpc=True)
ink = to_print(img)                           # (H, W, 4) uint8
```

## Pixels, lists and single colors

```python
>>> lcms2.convert([255, 0, 0], "sRGB", "Lab", intent="relative").round(2)
array([54.29, 80.81, 69.89])
>>> lcms2.convert([1.0, 0.0, 0.0], "sRGB", "oklab").round(4)
array([0.6279, 0.2249, 0.1258])
```

Python lists of integers are read as 8-bit (16-bit when a value exceeds 255); lists with floats as `float64`.

## Next

- {doc}`../guides/transforms`: intents, proofing, chains
- {doc}`../guides/spaces`: OKLab, HSV, PQ and friends
- {doc}`../guides/colormaps`: colormaps
