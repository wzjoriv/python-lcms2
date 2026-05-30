import lcms2
from lcms2.implementation import WHITE_POINTS
from lcms2 import _lcms2
import numpy as np

def is_valid_lab(profile):
    t = type(profile.handle)
    return "Lab" in profile.name and  "use freely" in profile.copyright and t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'


def test_temperature_to_white_point():
    t = 6504
    wp_list = _lcms2.white_point_from_temperature(t)
    wp = np.array(wp_list)
    reference = np.array(WHITE_POINTS[2]["D65"]["xyY"])
    assert np.allclose(wp, reference, atol=0.0001)


def test_lab_from_illuminant():
    p = lcms2.Profile("Lab", white_point="D65")
    assert is_valid_lab(p)

def test_lab_from_temperature():
    p = lcms2.Profile("Lab", temperature=6504)
    assert is_valid_lab(p)


def test_lab_from_white_point():
    wp = [0.31271405688264753, 0.3291190991371872, 1.0]
    p = lcms2.Profile("Lab", white_point=wp)
    assert is_valid_lab(p)


def test_default_lab():
    p = lcms2.Profile("Lab")
    assert is_valid_lab(p)



if __name__ == "__main__":
    test_temperature_to_white_point()
    test_lab_from_illuminant()
    test_lab_from_temperature()
    test_lab_from_white_point()
    test_default_lab()