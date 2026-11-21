# python-lcms2 examples

A numbered learning path from first conversions to advanced profiles. Run one with `python examples/<name>.py` (set `LCMS2_EXAMPLES_OUT` to choose where files are written), or open the notebook beside it to see every step with its output.

| # | Example | Notebook | What it shows | Key APIs |
|--:|---|---|---|---|
| 01 | [01_quick_conversions.py](01_quick_conversions.py) | [ipynb](01_quick_conversions.ipynb) | One-call conversion of single colors, CSS strings and arrays; dtypes and `out_dtype`. | `lcms2.convert`, `parse_color`, `to_hex`, `out_dtype` |
| 02 | [02_profiles.py](02_profiles.py) | [ipynb](02_profiles.ipynb) | Built-in and file profiles, header and tags, editing text, saving, bytes, creating RGB and gray profiles. | `Profile`, `Profile.open`, `from_bytes`, `Profile.rgb`, `Profile.transform` |
| 03 | [03_transforms_and_images.py](03_transforms_and_images.py) | [ipynb](03_transforms_and_images.ipynb) | A reusable `Transform` on uint8, uint16 and float images, alpha, `out=`, intents, black-point compensation, chains, formats and threads. | `Transform`, `Intent`, `bpc`, `FORMATS`, `set_threads` |
| 04 | [04_soft_proofing_and_gamut.py](04_soft_proofing_and_gamut.py) | [ipynb](04_soft_proofing_and_gamut.ipynb) | Soft proofing through a CMYK profile and gamut checks with a custom alarm color. | `proof=`, `gamut_check`, `set_alarm_codes` |
| 05 | [05_color_spaces.py](05_color_spaces.py) | [ipynb](05_color_spaces.ipynb) | OKLab, OKLCh, HSV, Display P3, Rec.2100 and friends as NumPy functions, mixed with ICC profiles. | `lcms2.spaces`, `ColorSpace.to`, `ColorSpace.list`, `RGBSpace` |
| 06 | [06_custom_color_spaces_to_icc.py](06_custom_color_spaces_to_icc.py) | [ipynb](06_custom_color_spaces_to_icc.ipynb) | Your own color space as an ICC v2, v4 or v5 profile, read back and round-tripped with real device values. | `ColorSpace`, `to_icc_profile`, `Profile.from_space`, `ICCSpace` |
| 07 | [07_colorimetry.py](07_colorimetry.py) | [ipynb](07_colorimetry.ipynb) | Color differences, XYZ/Lab/LCh/xyY, white points and temperature, chromatic adaptation, CIECAM02. | `delta_e`, `colorimetry.adapt`, `WHITE_POINTS`, `CIECAM02` |
| 08 | [08_tone_curves.py](08_tone_curves.py) | [ipynb](08_tone_curves.ipynb) | Gamma, parametric and tabulated curves; invert, compose, and build RGB, gray and linearization profiles. | `ToneCurve`, `then`, `join`, `Profile.gray`, `Profile.linearization` |
| 09 | [09_colormaps.py](09_colormaps.py) | [ipynb](09_colormaps.ipynb) | Perceptual colormaps from colors or generators, lookup on arrays, checks, color-blindness preview and saving. | `Colormap.from_colors`, `Colormap.get`, `simulate`, `save` |
| 10 | [10_cgats_and_measurements.py](10_cgats_and_measurements.py) | [ipynb](10_cgats_and_measurements.ipynb) | Create, save, load and edit CGATS / IT8 files and compare measurements with predictions. | `CGATS.from_columns`, `CGATS.load`, `column`, `set` |
| 11 | [11_device_links_and_advanced.py](11_device_links_and_advanced.py) | [ipynb](11_device_links_and_advanced.ipynb) | Device links, abstract and ink-limiting profiles, transform flags, per-profile options and pickling. | `to_device_link`, `Profile.abstract`, `ink_limiting`, `Flag` |
