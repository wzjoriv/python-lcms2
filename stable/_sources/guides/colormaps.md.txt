# Colormaps

{class}`lcms2.Colormap` builds colormaps by interpolating in a perceptual space (OKLab by default), so equal
steps in data look like equal steps in color.

```python
cm = lcms2.Colormap.from_colors(["#0b1a3a", "#f08a24", "#fff6d5"])
cm = lcms2.Colormap.sequential(250)                       # one hue, lightness ramp
cm = lcms2.Colormap.diverging("#2b5fb4", "#b4352b")
cm = lcms2.Colormap.cyclic()                              # constant lightness, hue rotation
cm = lcms2.Colormap.qualitative(8)                        # distinct categories
cm = lcms2.Colormap.get("glacier")                        # built-ins: lcms2.colormap.list()
```

Colors may be hex strings, CSS names, `rgb(...)` strings, 0–1 floats or 0–255 integers
({func}`lcms2.parse_colors`).

Colors that interpolate outside sRGB are brought back by reducing chroma at constant lightness and hue
(`gamut="chroma"`, default); `gamut="clip"` clips.

## Using one

```python
rgb  = cm(values, vmin=0, vmax=10)           # (..., 3) floats
rgba = cm(values, alpha=0.8, bytes=True)     # (..., 4) uint8
cm.reversed(), cm.resampled(16), cm[64:192]
plt.imshow(data, cmap=cm.to_matplotlib())    # or cm.register() then cmap="name"
cm.save("map.png")                           # .png, .csv, .json, .txt
```

In Jupyter a colormap displays as a swatch.

## Checking one

```python
cm.is_lightness_monotonic()                  # sequential maps should be
cm.uniformity()                              # spread of perceptual step sizes; lower is better
cm.simulate("deuteranopia")                  # also protanopia, tritanopia; severity=0..1
```
