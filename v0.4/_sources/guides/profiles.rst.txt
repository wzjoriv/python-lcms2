Profiles
========

A :class:`lcms2.Profile` wraps one ICC profile.

Getting one
-----------

.. code-block:: python

   import lcms2

   lcms2.Profile("sRGB")                     # built-in; also "Display P3", "Adobe RGB (1998)", "Lab", "XYZ", ...
   lcms2.Profile("display p3")               # names are case-insensitive with common aliases
   lcms2.Profile("press.icc")                # a path (str or pathlib.Path)
   lcms2.Profile.from_bytes(data)            # e.g. an ICC profile embedded in a PNG/JPEG/TIFF
   lcms2.Profile(lcms2.spaces.oklab)         # an analytic color space, see the ICC authoring guide

Unknown names raise ``KeyError`` listing every built-in profile and color space.

Creating profiles
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Constructor
     - Builds
   * - ``Profile.srgb()``, ``Profile.lab(white_point="D50", version=4)``, ``Profile.xyz()``
     - The LittleCMS virtual profiles.
   * - ``Profile.rgb(white_point, primaries, gamma)``
     - A matrix/TRC RGB profile; ``gamma`` is a number, a :class:`~lcms2.ToneCurve`, a ``(type, params)`` parametric tuple, or three of those.
   * - ``Profile.gray(white_point, gamma)``
     - A gray profile.
   * - ``Profile.abstract(brightness, contrast, hue, saturation)``
     - A Lab abstract profile (photographic adjustments).
   * - ``Profile.ink_limiting(color_space, limit)``
     - A CMYK ink-limiting device link.
   * - ``Profile.linearization(color_space, curves)``
     - A per-channel curve device link.

White points accept an illuminant name (``"D65"``, ``"A"``, ``"F2"``, ...), a temperature via ``temperature=``, or ``xyY``.

Inspecting
----------

.. code-block:: python

   >>> p = lcms2.Profile("FOGRA39L_coated.icc")
   >>> p
   <Profile 'FOGRA39L Coated' CMYK output v2.2>
   >>> p.color_space, p.pcs, p.device_class, p.channels
   ('CMYK', 'Lab', 'output', 4)

Header: ``version``, ``device_class``, ``color_space``, ``pcs``, ``rendering_intent``, ``creation_date``, ``profile_id``,
``header`` (everything at once; ``set_header(**fields)`` to change several).
Text: ``description``, ``manufacturer``, ``model``, ``copyright`` (all settable).
Tags: ``tags``, ``has_tag(sig)``, ``read_tag_raw(sig)``, ``write_tag_raw(sig, data)``, plus decoded readers
``media_white_point``, ``chromatic_adaptation``, ``colorants``, ``luminance``, ``trc`` (as :class:`~lcms2.ToneCurve`).
Capabilities: ``is_matrix_shaper``, ``is_clut(intent, direction)``, ``is_intent_supported(intent)``,
``supported_intents``, ``black_point(intent)``, ``tac`` (total area coverage), ``named_colors``.

Saving
------

.. code-block:: python

   p.description = "Press proof, 2026"
   p.compute_id()                 # refresh the MD5 profile ID
   p.save("proof.icc")            # or p.to_bytes()

Profiles pickle (as their bytes), so they work with ``multiprocessing``.
