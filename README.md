# python-lcms2

Python binding for the [LittleCMS2](https://github.com/mm2/Little-CMS) color management library.

## Installation

```
pip install python-lcms2
```

Pre-built wheels are available on PyPI for Windows (x86-64) and Linux (x86-64) on Python 3.10 and later. For other platforms, see [Building from source](#building-from-source).

## Usage

```python
import lcms2
```

### Profiles

A `Profile` can be created from a built-in name, a file on disk, or raw bytes in memory:

```python
srgb    = lcms2.Profile("sRGB")
prophoto = lcms2.Profile("ProPhoto RGB")
from_file = lcms2.Profile(filename="CMYK.icm")

with open("CMYK.icm", "rb") as f:
    from_bytes = lcms2.Profile(buffer=f.read())
```

Built-in profiles provided by LittleCMS: `sRGB`, `Lab`, `XYZ`.

Additional built-in RGB profiles: `Adobe RGB (1998)`, `Apple RGB`, `Best RGB`, `Beta RGB`, `Bruce RGB`, `CIE RGB`, `ColorMatch RGB`, `Don RGB 4`, `ECI RGB v2`, `Ekta Space PS5`, `NTSC RGB`, `PAL/SECAM RGB`, `ProPhoto RGB`, `SMPTE-C RGB`, `_sRGB`, `Wide Gamut RGB`.

### Transforms

A `Transform` converts pixel data from one profile and format to another:

```python
lab  = lcms2.Profile("Lab")
srgb = lcms2.Profile("sRGB")

transform = lcms2.Transform(lab, "Lab_DBL", srgb, "RGB_16")
```

The constructor signature is:

```
Transform(src_profile, src_format, dst_profile, dst_format, intent="PERCEPTUAL", flags="NONE")
```

All supported format strings:

```python
lcms2.DATA_TYPES.keys()
```

Rendering intents (`lcms2.INTENT`): `PERCEPTUAL`, `RELATIVE_COLORIMETRIC`, `SATURATION`, `ABSOLUTE_COLORIMETRIC`.

Flags (`lcms2.FLAG`) can be combined with commas, spaces, semicolons, or `|`:

```python
transform = lcms2.Transform(lab, "Lab_DBL", srgb, "RGB_16",
                            intent="PERCEPTUAL",
                            flags="GAMUTCHECK,SOFTPROOFING")
```

### Applying a transform

`Transform.apply` accepts a NumPy array or any array-compatible object and returns a transformed NumPy array:

```python
result = transform.apply([50.0, 0.0, 0.0])   # L*a*b* neutral grey
```

The input type must match the source format declared at construction time. When passing lists or other convertibles, the data is cast automatically where possible.

## Building from source

Clone the repository with its submodules:

```
git clone --recurse-submodules https://github.com/wzjoriv/python-lcms2.git
cd python-lcms2
```

Install the build tool if needed:

```
pip install build --upgrade
```

Build a wheel:

```
python -m build --wheel
```

The wheel is written to `dist/`. Install it with:

```
pip install dist/python_lcms2*.whl
```

Windows path detection (Windows SDK include and library directories) is handled automatically via the registry in `setup.py`. If the build fails on your Windows installation, the paths may need to be adjusted manually.

For bugs, questions, and feature requests use [GitHub Issues](https://github.com/wzjoriv/python-lcms2/issues).

## Acknowledgements

This project is a continuation of work by prior maintainers. The lineage of the repository is:
[sk1-project](https://github.com/sk1-project) → [RomanKosobrodov](https://github.com/RomanKosobrodov/python-lcms2) → [wzjoriv](https://github.com/wzjoriv/python-lcms2).