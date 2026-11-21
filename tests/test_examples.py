"""
Every script in ``examples/`` runs cleanly and prints something.
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


def _run(script: Path, cwd: Path) -> str:
    out = io.StringIO()
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous)
    return out.getvalue()


def test_examples_exist():
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_example_runs(script, tmp_path, monkeypatch):
    monkeypatch.setenv("LCMS2_EXAMPLES_OUT", str(tmp_path))
    assert _run(script, tmp_path).strip()


def test_notebooks_sit_next_to_scripts():
    missing = [path.name for path in SCRIPTS if not path.with_suffix(".ipynb").exists()]
    assert not missing
