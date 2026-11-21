Colorimetry
===========

All functions are vectorized over ``(..., 3)`` arrays and broadcast between operands.

Color differences
-----------------

.. code-block:: python

   >>> lcms2.delta_e([50, 2.6772, -79.7751], [50, 0, -82.7485])          # CIEDE2000
   2.0425
   >>> lcms2.delta_e(lab1, lab2, method="76")                             # Euclidean
   >>> lcms2.delta_e(lab1, lab2, method="94")
   >>> lcms2.delta_e(lab1, lab2, method="cmc", l=2, c=1)
   >>> lcms2.delta_e(lab1, lab2, method="bfd")
   >>> lcms2.delta_e(lab1, lab2, Kl=2)                                    # CIEDE2000 weights

CIECAM02
--------

.. code-block:: python

   cam = lcms2.CIECAM02(white_point=xyz_white, adapting_luminance=64, background=20, surround="average")
   jch = cam.forward(xyz)       # lightness J, chroma C, hue h
   xyz = cam.reverse(jch)

Conversions and white points
----------------------------

:mod:`lcms2.colorimetry` has ``xyz_to_lab``, ``lab_to_xyz``, ``lab_to_lch``, ``lch_to_lab``, ``xyz_to_xyy``,
``xyy_to_xyz``, ``adapt(xyz, src_white, dst_white, method="bradford")``, the ``D50``/``D65`` constants, and
``white_point_from_temperature`` / ``temperature_from_white_point``:

.. code-block:: python

   >>> from lcms2.colorimetry import temperature_from_white_point, D65_xyY
   >>> round(float(temperature_from_white_point(D65_xyY)))
   6504

Tone curves
-----------

:class:`lcms2.ToneCurve` wraps LittleCMS curves: ``ToneCurve.gamma(2.2)``, ``ToneCurve.parametric(type, params)``,
``ToneCurve.tabulated(values)``, ``ToneCurve.srgb()``. Call a curve on an array to evaluate it; ``inverse()``,
``estimate_gamma()``, ``is_monotonic``, ``table``.
