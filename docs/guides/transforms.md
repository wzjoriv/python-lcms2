# Transforms

`lcms2.convert(data, src, dst, **options)` is the one-call form; {class}`lcms2.Transform` is the reusable one.
Both take the same options.

## Options

| Option | Meaning |
|---|---|
| `intent` | `"perceptual"` (default), `"relative"`, `"saturation"`, `"absolute"`, the K-preserving intents, or {class}`lcms2.Intent`. |
| `bpc` | Black-point compensation. |
| `proof`, `proof_intent`, `gamut_check` | Soft proofing through a third profile; `gamut_check=True` paints out-of-gamut pixels with the alarm color. |
| `optimize` | `False` disables LittleCMS optimization (slower, sometimes more exact). |
| `precision` | `"auto"`, `"high"` or `"low"` precalculation. |
| `flags` | Any {class}`lcms2.Flag` combination, e.g. `"NOWHITEONWHITEFIXUP|GUESSDEVICECLASS"`. |
| `out_dtype`, `out` | Output dtype, or a preallocated output array. |

```python
proof = lcms2.Transform("sRGB", "sRGB", proof=lcms2.Profile("press.icc"), gamut_check=True)
lcms2.set_alarm_codes([0xFFFF, 0, 0xFFFF])       # magenta instead of gray
preview = proof(img)
```

## Chains

A list of profiles makes one multi-profile transform; per-profile `intent`, `bpc` and `adaptation` lists
build an extended transform.

```python
t = lcms2.Transform(["sRGB", lcms2.Profile.abstract(contrast=1.2), "press.icc"])
```

## Device links

```python
link = t.to_device_link()        # a Profile; save and use it in any CMS
```

## Explicit formats

When you need a specific LittleCMS pixel layout (planar, swapped bytes, premultiplied alpha, …) pass format
names; see `lcms2.FORMATS` for all 178:

```python
t = lcms2.Transform(srgb, "RGB_8", prophoto, "RGB_16")
```

## Threads

The GIL is released while pixels are converted and transforms keep no per-call state, so one `Transform` can
be called from several threads at once.

Large inputs are also split for you: a transform, `lcms2.delta_e` or a colormap lookup over 262,144
pixels or more runs in parallel chunks on a shared pool of `min(os.cpu_count(), 16)` threads. The result
is identical to a single-threaded run; on a 20-core machine a 4-megapixel float64 sRGB → Display P3 conversion drops from about
730 ms to 110 ms. Use `lcms2.set_threads(1)` when you already parallelise at a higher level, and
`lcms2.get_threads()` to read the setting.
