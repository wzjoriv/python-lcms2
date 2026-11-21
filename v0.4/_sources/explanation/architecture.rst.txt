Architecture
============

.. code-block:: text

   lcms2/
     _ext/          C extension: one file per LittleCMS area, handles passed as capsules
     _profile.py    Profile            _transform.py  Transform        _convert.py  convert
     _formats.py    format inference   _constants.py  Intent, Flag     _curves.py   ToneCurve
     colorimetry    ΔE, CIECAM02, Lab/LCh/xyY, white points
     cgats.py       CGATS / IT8
     spaces/        ColorSpace and the analytic spaces (pure NumPy)
     icc/           pure-Python ICC reader, writer and tracer
     colormap/      Colormap

**Binding.** The extension is hand-written C. Each LittleCMS object (profile, transform, tone curve, IT8
handle) is a capsule owned by a small Python class; there are no C types with mutable state. A LittleCMS error
handler records the last message per thread, so failures raise :class:`lcms2.CMSError` with LittleCMS's own
explanation. Transforms run with the GIL released, in chunks, and are created with ``cmsFLAGS_NOCACHE`` so
they can be shared between threads.

**Two color models.** ICC profiles are evaluated by LittleCMS; analytic color spaces are NumPy functions
through an XYZ (D50) hub. ``lcms2.convert`` resolves each side (a built-in profile name wins over a color-space
name) and routes profile↔profile through LittleCMS, anything involving a color space through XYZ.

**Why C and not Cython.** LittleCMS does the work in one ``cmsDoTransform`` call per chunk, so the wrapper is
thin; plain C keeps the build dependency-free and gives direct control over the GIL and object lifetimes.

**LittleCMS updates.** LittleCMS is a git submodule in ``vendor/Little-CMS``. Dependabot opens a pull request
for new upstream commits; ``python scripts/dev.py lcms2 update`` checks out the newest release, regenerates the
pixel-format table and lists API functions added or removed.
