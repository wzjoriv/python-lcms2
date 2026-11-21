"""Shared fixtures: sample data paths and profiles."""

from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    return DATA


@pytest.fixture
def cmyk():
    import lcms2

    return lcms2.Profile(DATA / "profiles" / "test1.icc")


@pytest.fixture
def srgb():
    import lcms2

    return lcms2.Profile("sRGB")
