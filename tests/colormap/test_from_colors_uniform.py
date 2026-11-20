import numpy as np
import pytest

from lcms2.colormap import Colormap
from lcms2.spaces import oklab, srgb


def test_uniform_positions_give_even_perceptual_steps():
    cm = Colormap.from_colors(["#000000", "#202020", "#ffffff"], positions="uniform", gamut="none", n=64)
    steps = np.linalg.norm(np.diff(srgb.to(oklab, cm.colors), axis=0), axis=1)
    assert steps.max() / steps.min() < 1.05


def test_default_positions_are_even():
    cm = Colormap.from_colors(["#000", "#f80", "#fff"], n=3)
    assert cm.hex() == ["#000000", "#ff8800", "#ffffff"]


def test_uniform_rejects_repeated_colours_and_bad_strings():
    with pytest.raises(ValueError, match="differ"):
        Colormap.from_colors(["#000", "#000", "#fff"], positions="uniform")
    with pytest.raises(ValueError, match="uniform"):
        Colormap.from_colors(["#000", "#fff"], positions="even")
