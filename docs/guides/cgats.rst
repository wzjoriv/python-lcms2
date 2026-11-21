Measurement files (CGATS / IT8)
===============================

:class:`lcms2.CGATS` reads and writes the text format used by spectrophotometers and profiling tools.

.. code-block:: python

   it8 = lcms2.CGATS.load("measurements.txt")
   it8.columns                    # ['SAMPLE_ID', 'RGB_R', 'RGB_G', 'RGB_B', 'LAB_L', 'LAB_A', 'LAB_B']
   it8.patches                    # patch names
   L = it8.column("LAB_L")        # a numeric column as an ndarray
   it8.data                       # every column, as a dict
   it8["A1", "LAB_L"]             # one cell
   it8.properties["ORIGINATOR"]   # header keywords

Create one from columns and save it:

.. code-block:: python

   out = lcms2.CGATS.from_columns({"SAMPLE_ID": names, "LAB_L": L, "LAB_A": a, "LAB_B": b})
   out.save("out.txt")
