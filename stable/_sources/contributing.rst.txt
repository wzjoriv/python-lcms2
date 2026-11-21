Contributing
============

Setup
-----

.. code-block:: bash

   git clone --recurse-submodules https://github.com/wzjoriv/python-lcms2.git
   cd python-lcms2
   pip install -e ".[dev]"

After changing C code in ``lcms2/_ext``, rebuild with ``pip install -e . --no-build-isolation``.

Developer commands
------------------

``scripts/dev.py`` wraps the day-to-day tasks (``python scripts/dev.py --help``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - What it does
   * - ``dev.py check``
     - Lint, tests and the docs build without warnings; non-zero when one fails.
   * - ``dev.py test [quick|cov]``
     - The test suite, quiet or with coverage.
   * - ``dev.py docs [--open] [--clean]``
     - Builds the HTML docs into ``docs/_build/html``.
   * - ``dev.py stub generate|check``
     - Type stubs with stubpy.
   * - ``dev.py version show|bump|set``
     - Reads or changes ``lcms2.__version__``.
   * - ``dev.py lcms2 update``
     - Moves ``vendor/Little-CMS`` to the newest LittleCMS release and reports API changes.
   * - ``dev.py lcms2 formats``
     - Regenerates ``lcms2/_ext/formats_table.h`` from ``lcms2.h``.

Run ``dev.py check`` before opening a pull request, and add the ``full-tests`` label to run every OS and Python.

Conventions
-----------

- Private modules start with ``_``; public names are listed in ``__all__``.
- Public functions have NumPy-style docstrings with runnable examples.
- C functions start their docstring with a ``name(arg) -> Ret`` line; every LittleCMS failure goes through
  ``lcms2_raise``.
- Tests mirror the package (``tests/core``, ``tests/spaces``, ...).
