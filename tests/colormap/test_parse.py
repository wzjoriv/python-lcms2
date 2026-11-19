import numpy as np
import pytest

from lcms2._parse import CSS_NAMES, parse_color, parse_colors, to_hex


@pytest.mark.parametrize(
    "text, expected",
    [
        ("#f00", [1, 0, 0]),
        ("#F00", [1, 0, 0]),
        ("#f008", [1, 0, 0, 8 * 17 / 255]),
        ("#ff8000", [1, 128 / 255, 0]),
        ("#ff800080", [1, 128 / 255, 0, 128 / 255]),
        ("red", [1, 0, 0]),
        ("  ReBeccaPurple ", [0.4, 0.2, 0.6]),
        ("rgb(255, 0, 0)", [1, 0, 0]),
        ("rgb(100%, 50%, 0%)", [1, 0.5, 0]),
        ("RGB(0 0 255)", [0, 0, 1]),
        ("rgba(0, 0, 0, 0.5)", [0, 0, 0, 0.5]),
        ("rgba(0, 0, 0, 50%)", [0, 0, 0, 0.5]),
        ("rgb(0 0 0 / 25%)", [0, 0, 0, 0.25]),
    ],
)
def test_strings(text, expected):
    assert np.allclose(parse_color(text), expected)


def test_css_table_complete():
    assert len(CSS_NAMES) == 148
    assert parse_color("grey").tolist() == parse_color("gray").tolist()
    assert "darkslategrey" in CSS_NAMES and "lightgoldenrodyellow" in CSS_NAMES


def test_numeric_forms():
    assert np.allclose(parse_color((1.0, 0.5, 0.0)), [1, 0.5, 0])
    assert np.allclose(parse_color([255, 0, 0]), [1, 0, 0])
    assert np.allclose(parse_color(np.array([0, 128, 255], dtype=np.uint8)), [0, 128 / 255, 1])
    assert parse_color((0.1, 0.2, 0.3, 0.4)).shape == (4,)
    arr = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(parse_color(arr), arr)


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        "#12",
        "#12345",
        "#ggg",
        "rgb(1, 2)",
        "rgb(a, b, c)",
        (1, 2),
        (2.0, 0, 0),
        (300, 0, 0),
        (float("nan"), 0, 0),
        ["a", "b", "c"],
        b"red",
        None,
    ],
)
def test_errors_name_the_value(bad):
    with pytest.raises(ValueError, match="cannot parse colour"):
        parse_color(bad)


def test_parse_colors():
    out = parse_colors(["red", "#00f", (0, 255, 0)])
    assert out.shape == (3, 3)
    assert parse_colors(np.array([[0, 0, 0], [1, 1, 1.0]])).shape == (2, 3)
    assert parse_colors(["#f008", "red"][:1]).shape == (1, 4)
    with pytest.raises(ValueError, match="mix"):
        parse_colors(["#f008", "red"])
    with pytest.raises(ValueError):
        parse_colors("red")
    with pytest.raises(ValueError):
        parse_colors([])


def test_to_hex():
    assert to_hex([1, 0.5, 0]) == "#ff8000"
    assert to_hex([0, 0, 0, 1]) == "#000000ff"
    assert to_hex([[0, 0, 0], [2, 2, 2]]) == ["#000000", "#ffffff"]
    assert parse_color(to_hex(parse_color("#12ab9f"))).tolist() == parse_color("#12ab9f").tolist()
    with pytest.raises(ValueError):
        to_hex([1, 2])
