Color spaces
============

:mod:`lcms2.spaces` defines color spaces as NumPy functions to and from CIE XYZ (D50). They need no ICC
profile and mix freely with profiles.

.. code-block:: python

   >>> from lcms2.spaces import srgb, oklab, oklch
   >>> srgb.to("oklch", [1.0, 0.0, 0.0]).round(4)       # names or instances
   >>> oklab.to("Display P3", lab)                      # profile names work too

Available: ``srgb``, ``srgb_linear``, ``display_p3``, ``display_p3_linear``, ``a98_rgb``, ``prophoto_rgb``, ``rec2020``,
``rec2100_linear``, ``rec2100_pq``, ``rec2100_hlg``, ``xyz_d50``, ``xyz_d65``, ``oklab``, ``oklch``, ``lch``, ``hsv``, ``hwb``,
``cmyk`` (a formula, not an ink model), ``gray``, ``loggray``. ``ColorSpace.list()`` lists the registered names;
``lcms2.spaces.get(name)`` returns one.

Spaces and profiles
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - From
     - To
     - Call
   * - space
     - profile
     - ``lcms2.Profile(space)`` or ``space.profile`` (cached)
   * - profile
     - space
     - ``profile.space``
   * - anything
     - anything
     - ``lcms2.convert(x, a, b)`` / ``space.to(b, x)``

When either side of a conversion is a color space, the conversion goes through XYZ D50 in ``float64``;
rendering intents then do not apply (they are an ICC concept). Profile-to-profile conversions stay in LittleCMS.

Your own space
--------------

Subclass :class:`~lcms2.spaces.ColorSpace` and implement ``to_xyz`` / ``from_xyz`` on ``(..., n_channels)``
arrays. Subclasses register themselves by ``name`` (default: the class name).

.. code-block:: python

   import numpy as np
   from lcms2.spaces import ColorSpace

   class Gamma18(ColorSpace):
       n_channels = 3
       def to_xyz(self, rgb):
           return np.asarray(rgb) ** 1.8 @ M.T
       def from_xyz(self, xyz):
           return np.clip(np.asarray(xyz) @ M_inv.T, 0, None) ** (1 / 1.8)

:class:`~lcms2.spaces.RGBSpace` is a shortcut for RGB spaces: give it a ``matrix`` and per-channel
``decode``/``encode``. ``device_domain`` declares the valid range per channel and ``circular_device_channels`` marks
hue channels.
