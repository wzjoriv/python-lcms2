import io

import numpy as np
import pytest

from lcms2.colormap import Colormap
from lcms2.colormap._export import png_bytes

SIG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def cm():
    return Colormap.from_colors(["#102030", "#f0a020", "#ffffe0"], n=16, name="demo")


def test_to_image(cm):
    img = cm.to_image(64, 8)
    assert img.shape == (8, 64, 3) and img.dtype == np.uint8
    assert img[0, 0].tolist() == [16, 32, 48] and np.array_equal(img[0], img[-1])
    v = cm.to_image(4, 40, vertical=True)
    assert v.shape == (40, 4, 3)
    assert v[-1, 0].tolist() == [16, 32, 48] and v[0, 0].tolist() == [255, 255, 224]
    with pytest.raises(ValueError):
        cm.to_image(0, 3)


@pytest.mark.parametrize("pillow", [True, False])
def test_png_bytes(cm, pillow):
    data = png_bytes(cm.to_image(32, 4), use_pillow=pillow)
    assert data.startswith(SIG) and data.endswith(b"IEND\xaeB`\x82")
    assert png_bytes(np.zeros((2, 2, 4), np.uint8), use_pillow=pillow).startswith(SIG)


def test_pure_png_decodes():
    pil = pytest.importorskip("PIL.Image")
    img = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    back = np.asarray(pil.open(io.BytesIO(png_bytes(img, use_pillow=False))))
    assert np.array_equal(back, img)


def test_notebook_reprs(cm):
    assert cm._repr_png_().startswith(SIG)
    html = cm._repr_html_()
    assert "demo" in html and "linear-gradient" in html and "#102030" in html
    assert "&lt;" in Colormap(cm.colors, name="<b>")._repr_html_()


@pytest.mark.parametrize("ext", [".csv", ".json", ".txt"])
def test_save_load_round_trip(cm, tmp_path, ext):
    path = cm.save(tmp_path / f"map{ext}")
    back = Colormap.load(path)
    assert len(back) == len(cm)
    assert np.allclose(back.colors, cm.colors, atol=1 / 255 if ext == ".txt" else 1e-12)
    if ext == ".json":
        assert back == cm and back.name == "demo"


def test_save_png_and_errors(cm, tmp_path):
    p = cm.save(str(tmp_path / "map.PNG"))
    assert p.read_bytes().startswith(SIG)
    with pytest.raises(ValueError, match="cannot save"):
        cm.save(tmp_path / "map.xyz")
    with pytest.raises(ValueError, match="cannot load"):
        Colormap.load(tmp_path / "map.png")


def test_load_txt_accepts_names(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("red\n#00ff00\n\nrgb(0,0,255)\n")
    assert Colormap.load(p).hex() == ["#ff0000", "#00ff00", "#0000ff"]


def test_matplotlib_round_trip(cm):
    pytest.importorskip("matplotlib")
    cmap = cm.to_matplotlib()
    assert cmap.N == len(cm) and cmap.name == "demo"
    assert np.allclose(cmap(np.linspace(0, 1, 16))[:, :3], cm.colors)
    back = Colormap.from_matplotlib(cmap, n=16)
    assert np.allclose(back.colors, cm.colors, atol=1e-6)
    assert len(Colormap.from_matplotlib("gray", n=9)) == 9


def test_matplotlib_register(cm):
    mpl = pytest.importorskip("matplotlib")
    cmap = cm.register("lcms2-test-map")
    assert mpl.colormaps["lcms2-test-map"].N == len(cm)
    assert cmap.name == "lcms2-test-map"
    cm.register()
    assert "demo" in mpl.colormaps


def test_matplotlib_missing_hint(cm, monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("no matplotlib")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"pip install python-lcms2\[plot\]"):
        cm.to_matplotlib()
