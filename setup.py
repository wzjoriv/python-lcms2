"""Build the ``lcms2._lcms2`` extension from the vendored LittleCMS sources.

Project metadata lives in ``pyproject.toml``; this file only declares the C extension.
"""

from glob import glob
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

ROOT = Path(__file__).parent
LCMS = Path("vendor", "Little-CMS")

if not (ROOT / LCMS / "src" / "cmsxform.c").exists():
    raise RuntimeError(
        "LittleCMS sources are missing. Clone with --recurse-submodules or run "
        "`git submodule update --init`."
    )

SOURCES = sorted(glob("lcms2/_ext/*.c")) + sorted(glob(str(LCMS / "src" / "*.c")))

setup(
    ext_modules=[
        Extension(
            name="lcms2._lcms2",
            language="c",
            sources=SOURCES,
            include_dirs=[str(LCMS / "include"), str(LCMS / "src"), "lcms2/_ext", np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ]
)
