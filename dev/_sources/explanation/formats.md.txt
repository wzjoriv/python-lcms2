# How arrays become pixel formats

LittleCMS describes pixels with a 32-bit format word: color space, channels, extra (alpha) channels, bytes
per sample, float or integer, planar or interleaved. python-lcms2 derives it from your array so you never
write one.

| Array | Format |
|---|---|
| dtype `uint8` / `uint16` | 1 / 2 bytes per sample, integer |
| dtype `float16` / `float32` / `float64` | half / single / double float |
| other dtypes | `TypeError` with the accepted list and an `.astype(...)` hint |
| last axis = profile channels | no alpha |
| last axis = profile channels + 1 | one extra channel, copied to the output unchanged |
| 1-channel (gray) profile | `(H, W)` works: each element is a pixel; an explicit trailing axis of 1 is kept |

Python lists: all integers → `uint8` (or `uint16` past 255); any float → `float64`.

**Value ranges** follow LittleCMS: integer RGB/CMYK use the full integer range; float RGB is 0–1; float CMYK is
0–100 (percent ink); float Lab is real L\*a\*b\* (L 0–100); float XYZ has Y = 1 for the white.

**Output dtype**: the input dtype, except integer input converted to Lab or XYZ gives `float64`, since encoded
integer Lab is rarely what you want. `out_dtype=` overrides.

**Memory layout**: any strides work; non-contiguous input is copied once. Output is C-contiguous with the
input's leading shape.

A `Transform` without explicit formats creates one LittleCMS transform per distinct input format and caches
it, so calling the same `Transform` on `uint8` and `float32` images both work.
