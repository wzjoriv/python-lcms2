import lcms2
import numpy as np


def test_srgb_profile():
    reference = lcms2.Profile("sRGB")
    p = lcms2.implementation.create_profile("_sRGB")
    lab = lcms2.Profile("Lab")
    t1 = lcms2.Transform(lab, "Lab_DBL", reference, "RGB_16")
    t2 = lcms2.Transform(lab, "Lab_DBL", p, "RGB_16")
    v = [50.0, 2.0, -3.0]
    z1 = t1.apply(v)
    z2 = t2.apply(v)
    assert np.allclose(z1, z2, atol=1)


if __name__ == "__main__":
    test_srgb_profile()
