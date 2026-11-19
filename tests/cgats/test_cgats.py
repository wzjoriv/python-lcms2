"""CGATS / IT8 reading, writing and editing."""

import numpy as np
import pytest
from lcms2._lcms2 import LCMSError

from lcms2.cgats import CGATS

SAMPLE = b"""IT8.7/2
ORIGINATOR "Test lab"
DESCRIPTOR "Two patch chart"
NUMBER_OF_FIELDS 4
NUMBER_OF_SETS 3
BEGIN_DATA_FORMAT
SAMPLE_ID LAB_L LAB_A LAB_B
END_DATA_FORMAT
BEGIN_DATA
A1 95.5 -0.5 1.25
A2 50.0 10.0 -20.0
A3 5.0 0.0 0.0
END_DATA
"""


def test_load_bytes_and_inspect():
    it8 = CGATS.from_bytes(SAMPLE)
    assert it8.sheet_type == "IT8.7/2"
    assert len(it8) == 1
    assert it8.columns == ["SAMPLE_ID", "LAB_L", "LAB_A", "LAB_B"]
    assert it8.n_rows == 3
    assert it8.patches == ["A1", "A2", "A3"]
    assert it8.properties["ORIGINATOR"] == "Test lab"
    assert "DESCRIPTOR" in it8.properties
    assert it8.column("LAB_L").tolist() == [95.5, 50.0, 5.0]
    assert it8.column("SAMPLE_ID").tolist() == ["A1", "A2", "A3"]
    data = it8.data
    assert set(data) == set(it8.columns)
    assert data["LAB_B"].dtype == np.float64


def test_cells():
    it8 = CGATS.from_bytes(SAMPLE)
    assert it8["A2", "LAB_A"] == 10.0
    assert it8.get(0, "lab_b") == 1.25
    assert it8.get("A3", 0) == "A3"
    it8["A2", "LAB_A"] = 12.5
    assert it8["A2", "LAB_A"] == 12.5
    it8.set("A1", "SAMPLE_ID", "W1")
    assert it8.patches[0] == "W1"
    with pytest.raises(KeyError, match="no column named"):
        it8["A2", "NOPE"]
    with pytest.raises(KeyError, match="no patch named"):
        it8["Z9", "LAB_L"]
    with pytest.raises(IndexError):
        it8.get(9, 0)


def test_save_load_round_trip(tmp_path):
    it8 = CGATS.from_bytes(SAMPLE)
    it8.properties["EXTRA"] = "hello world"
    it8.properties["COUNT"] = 7
    it8["A2", "LAB_L"] = 51.125
    path = tmp_path / "out.it8"
    it8.save(path)
    again = CGATS.load(path)
    assert again.to_dict() == it8.to_dict()
    assert again.properties["EXTRA"] == "hello world"
    assert again.properties["COUNT"] == "7"
    assert again["A2", "LAB_L"] == 51.125
    assert CGATS.from_bytes(again.to_bytes()).to_dict() == it8.to_dict()


def test_from_columns_and_properties_mapping():
    it8 = CGATS.from_columns(
        {"SAMPLE_ID": ["a", "b"], "R": [0, 255], "G": [0.5, 1.5]},
        sheet_type="CGATS.17",
        properties={"K": "v"},
    )
    assert it8.columns == ["SAMPLE_ID", "R", "G"]
    assert it8.n_rows == 2
    np.testing.assert_array_equal(it8.column("R"), [0.0, 255.0])
    assert dict(it8.properties)["K"] == "v"
    assert len(it8.properties) == len(list(it8.properties))
    with pytest.raises(TypeError, match="cannot be removed"):
        del it8.properties["K"]
    with pytest.raises(KeyError):
        it8.properties["MISSING"]
    with pytest.raises(ValueError, match="same length"):
        CGATS.from_columns({"a": [1], "b": [1, 2]})


def test_tables_and_sheet_type():
    it8 = CGATS()
    assert it8.sheet_type == "CGATS.17"
    it8.sheet_type = "IT8.7/1"
    it8.define_table(["SAMPLE_ID", "X"], 1)
    it8[0, "SAMPLE_ID"] = "p"
    it8[0, "X"] = 3
    second = it8.add_table()
    assert second == 1 and len(it8) == 2
    it8.define_table(["Y"], 2)
    it8[0, 0] = 1.5
    it8[1, 0] = 2.5
    data = it8.to_dict()
    assert [t["columns"] for t in data["tables"]] == [
        {"SAMPLE_ID": ["p"], "X": [3.0]},
        {"Y": [1.5, 2.5]},
    ]
    reloaded = CGATS.from_bytes(it8.to_bytes())
    assert len(reloaded) == 2
    reloaded.select(1)
    assert reloaded.column("Y").tolist() == [1.5, 2.5]
    reloaded.table = 0
    assert reloaded.table == 0 and reloaded.columns == ["SAMPLE_ID", "X"]
    with pytest.raises(IndexError, match="table 5 out of range"):
        reloaded.select(5)
    assert "CGATS(tables=2" in repr(reloaded)


def test_errors():
    with pytest.raises(LCMSError):
        CGATS.from_bytes(b"this is not a CGATS file at all")
    with pytest.raises(ValueError):
        CGATS.from_bytes(b"")
    it8 = CGATS.from_bytes(SAMPLE)
    with pytest.raises(LCMSError):
        it8.define_table(["A"], 1)
    with pytest.raises(FileNotFoundError):
        CGATS.load("definitely/missing.it8")
