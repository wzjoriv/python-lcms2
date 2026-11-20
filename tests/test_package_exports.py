"""The public surface of the lcms2 package: `__all__` lists, public modules and docstrings."""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import lcms2
from tests.test_package_boundaries import MODULES, _is_private


def _public_modules() -> list[str]:
    """Dotted names of the modules no segment of which is private."""
    return sorted(m for m in MODULES if not any(_is_private(s) for s in m.split(".")))


PUBLIC_MODULES = _public_modules()


@pytest.mark.parametrize("name", lcms2.__all__)
def test_all_names_are_importable(name):
    assert hasattr(lcms2, name)
    namespace: dict = {}
    exec(f"from lcms2 import {name}", namespace)
    assert namespace[name] is getattr(lcms2, name)


def test_all_has_no_duplicates():
    assert len(lcms2.__all__) == len(set(lcms2.__all__))


def test_metadata():
    assert re.fullmatch(r"\d+\.\d+\.\d+", lcms2.__version__)
    assert lcms2.__author__ == "Josue N Rivera"


def test_errors_are_exported_and_share_a_base():
    for name in ("CMSError", "ICCParseError", "ICCWriteError"):
        assert name in lcms2.__all__
    assert issubclass(lcms2.ICCParseError, lcms2.CMSError)
    assert issubclass(lcms2.ICCWriteError, lcms2.CMSError)
    assert lcms2.icc.ICCParseError is lcms2.ICCParseError


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_every_public_module_has_an_all_that_exists(module):
    mod = importlib.import_module(module)
    names = getattr(mod, "__all__", None)
    assert names is not None, f"{module} has no __all__"
    assert len(names) == len(set(names)), f"{module}.__all__ has duplicates"
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module}.__all__ names missing attributes: {missing}"


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_public_modules_have_docstrings(module):
    assert (inspect.getdoc(importlib.import_module(module)) or "").strip()


@pytest.mark.parametrize(
    "package",
    [
        "lcms2.icc._tracer.tracing",
        "lcms2.icc._io.mpe",
        "lcms2.colorimetry",
    ],
)
def test_split_packages_reexport_explicitly(package):
    """Packages made from a long module list their public names; none uses ``import *``."""
    mod = importlib.import_module(package)
    init = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import *" not in init
    assert (inspect.getdoc(mod) or "").strip()
    assert mod.__all__ and all(hasattr(mod, n) for n in mod.__all__)
    # `__all__` holds only names the package re-exports, not its submodules
    submodules = {m.rpartition(".")[2] for m in MODULES if m.startswith(package + ".")}
    assert not submodules & set(mod.__all__)


def test_no_module_uses_star_imports():
    for module, (path, _) in MODULES.items():
        assert "import *" not in path.read_text(encoding="utf-8"), module


def test_public_callables_have_numpy_docstrings():
    for obj in (lcms2.convert, lcms2.delta_e, lcms2.Profile.from_space, lcms2.spaces.ColorSpace):
        doc = inspect.getdoc(obj) or ""
        assert "Parameters" in doc.splitlines(), obj
    assert "Parameters" in (inspect.getdoc(lcms2.spaces.ColorSpace.to_icc_profile) or "")


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocess on Emscripten")
def test_importing_the_package_does_not_import_numpy_patches():
    code = "import numpy as np, lcms2; print(np.asarray.__module__, np.random.random.__module__)"
    out = subprocess.run(
        [sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True
    ).stdout
    assert "lcms2" not in out
