ICC authoring
=============

``lcms2.Profile(space)`` turns any :class:`~lcms2.spaces.ColorSpace` into an ICC profile that other color
management systems (Photoshop, browsers, OS color pipelines) can use.

How it works
------------

The space's ``to_xyz`` / ``from_xyz`` are run once on a tracing array that records every NumPy operation. The
recorded graph is then matched against ICC building blocks, in order of exactness:

1. **matrix/TRC**: a 3x3 matrix with per-channel curves (classic RGB display profiles);
2. **gray TRC**: one curve;
3. **LUT** (``mAB``/``mBA``): curves, matrix and CLUT stages in exact composition;
4. **v5 multiProcessElement / calculator**: exact formulas for spaces like OKLab that no v2/v4 element can express;
5. **CLUT fallback**: a sampled grid, used when nothing exact fits.

The result is verified by evaluating the written profile against your functions.

.. code-block:: python

   info = lcms2.spaces.oklab.to_icc_profile()
   info["method"], info["notes"]          # which representation was chosen, and why

When a function cannot be traced (Python ``if`` on array values, ``float(x)``, iteration, calls into other
libraries), the writer falls back to a CLUT and says why in ``notes``. Pass ``strict=True`` to get an
``ICCWriteError`` instead.

.. note::

   Spaces with a circular hue channel (HSV, HSL, HWB, LCh, OKLCh) are sampled into a CLUT. ICC
   interpolation is linear and does not wrap hue, so colours whose hue lies between the last grid point
   and 360° are interpolated through the whole hue range. Expect small errors near hue 0/360 in such
   profiles. Converting with the space itself (``lcms2.convert(x, "sRGB", "oklch")``) is exact.

Versions
--------

``version=4`` (default) writes v4.4; ``version=2`` writes v2.4 for older software; ``version=5`` allows the exact
calculator elements. LittleCMS reads v2 and v4; v5 profiles are read by :class:`lcms2.icc.ICCSpace`.

Reading back without LittleCMS
------------------------------

:class:`lcms2.icc.ICCSpace` evaluates ICC profiles (v2, v4, v5) in pure NumPy, which is useful for checking a
profile or for v5 elements LittleCMS does not implement.
