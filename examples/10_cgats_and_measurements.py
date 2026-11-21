"""
# CGATS measurement files

Create, save, read and edit CGATS / IT8 measurement files, and compare measured colors with predictions.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

import lcms2
from lcms2 import CGATS

out = Path(os.environ.get("LCMS2_EXAMPLES_OUT") or tempfile.mkdtemp())

# %% [markdown]
# `CGATS.from_columns` builds a file from named columns of equal length. Numbers are stored as numbers and
# anything else as text. Here are three patches with their RGB values and a simulated Lab measurement.

# %%
rng = np.random.default_rng(0)
names = ["A1", "A2", "A3", "A4"]
rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], np.uint8)
lab = lcms2.convert(rgb, "sRGB", "Lab")
measured = (lab + rng.normal(0, 0.5, lab.shape)).round(2)
it8 = CGATS.from_columns(
    {
        "SAMPLE_ID": names,
        "RGB_R": rgb[:, 0],
        "RGB_G": rgb[:, 1],
        "RGB_B": rgb[:, 2],
        "LAB_L": measured[:, 0],
        "LAB_A": measured[:, 1],
        "LAB_B": measured[:, 2],
    },
    properties={"ORIGINATOR": "examples"},
)
print(it8.columns)
print(it8.patches, it8.n_rows, it8.sheet_type)

# %% [markdown]
# `save` writes the text file and `CGATS.load` reads it back; `to_bytes` and `from_bytes` work in memory.

# %%
path = out / "measurements.txt"
it8.save(path)
print(path.name)
loaded = CGATS.load(path)
print(loaded.properties)
print(CGATS.from_bytes(loaded.to_bytes()).patches)

# %% [markdown]
# Read data as arrays or cell by cell: `column` gives a float array (or strings when the column is text),
# `data` every column in a dict, and `loaded["A2", "LAB_L"]` one cell by patch and column name.

# %%
L = loaded.column("LAB_L")
print(L)
print(loaded.column("SAMPLE_ID"))
print(list(loaded.data))
print(loaded["A2", "LAB_L"], loaded.get(1, "LAB_A"))

# %% [markdown]
# Edit a cell with `set` and compare the measurements with what sRGB predicts using a color difference.

# %%
loaded.set("A4", "LAB_L", 55.0)
lab_measured = np.stack([loaded.column(c) for c in ("LAB_L", "LAB_A", "LAB_B")], axis=-1)
rgb_columns = np.stack([loaded.column(c) for c in ("RGB_R", "RGB_G", "RGB_B")], axis=-1).astype(np.uint8)
predicted = lcms2.convert(rgb_columns, "sRGB", "Lab")
print(lcms2.delta_e(lab_measured, predicted).round(2))
