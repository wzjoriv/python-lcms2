Installation
============

.. code-block:: bash

   pip install python-lcms2
   pip install "python-lcms2[plot]"     # adds matplotlib for colormap export

Wheels are published for Windows, Linux and macOS on x86-64 and ARM64: CPython 3.10 and later (including the
free-threaded 3.14t and 3.15t builds) and PyPy 3.11. NumPy (2.0+) is the only runtime dependency. LittleCMS is compiled into the extension; nothing else needs to be installed.
python-lcms2 is released under the MIT license, like the LittleCMS it bundles.

In the browser (Pyodide), ``await micropip.install("python-lcms2")`` installs the Pyodide wheel from PyPI; it is
also attached to each `GitHub release <https://github.com/wzjoriv/python-lcms2/releases>`_.

Check what you got:

.. code-block:: python

   >>> import lcms2
   >>> lcms2.__version__, lcms2.get_version()   # package, bundled LittleCMS
   ('0.3.0', '2.20')

From source
-----------

.. code-block:: bash

   git clone --recurse-submodules https://github.com/wzjoriv/python-lcms2.git
   cd python-lcms2
   pip install -e ".[dev]"

A C compiler is required: MSVC (Visual Studio Build Tools) on Windows, gcc or clang elsewhere.
