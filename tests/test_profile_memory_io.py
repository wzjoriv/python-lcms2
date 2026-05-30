import lcms2
import os

def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__),
                        'cms_data',
                        filename)


def test_profile_from_memory():
    fn = get_filepath("CMYK.icm")
    reference = lcms2.open_profile(fn)
    with open(fn, "rb") as fid:
        buffer = fid.read()
        p = lcms2.Profile(buffer=buffer)
    assert reference.name == p.name
    assert reference.info == p.info
    assert reference.copyright == p.copyright


def test_memory_round_trip():
    reference = lcms2.Profile("XYZ")
    buffer = reference.to_bytes()
    p = lcms2.Profile(buffer=buffer)
    assert reference.name == p.name
    assert reference.info == p.info
    assert reference.copyright == p.copyright



if __name__ == "__main__":
    test_profile_from_memory()
    