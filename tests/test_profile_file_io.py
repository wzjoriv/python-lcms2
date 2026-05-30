import os
import lcms2


def get_filepath(filename):
    return os.path.join(os.path.dirname(__file__),
                        'cms_data',
                        filename)


def test_invalid_profile():
    try:
        filename = get_filepath('empty.icm')
        lcms2.Profile(filename=filename)
    except Exception as e:
        return
    assert False


def test_missing_profile():
    try:
        filename = get_filepath('this_profile_does_not_exist.icm')
        lcms2.Profile(filename=filename)
    except Exception as e:
        return
    assert False


def test_valid_cmyk_profile():
    filename = get_filepath("CMYK.icm")
    p = lcms2.Profile(filename=filename)
    assert "CMYK" in p.name
    assert "Offset printing" in p.info
    assert "Public" in p.copyright


def test_file_round_trip():
    reference = lcms2.Profile("Lab")
    fn = get_filepath("tmp_profile.icm")
    reference.save(fn)
    p = lcms2.Profile(filename=fn)
    assert reference.name == p.name
    assert reference.info == p.info
    assert reference.copyright == p.copyright
    p.handle = None  # close CMS profile and release the file 
    os.remove(fn)


if __name__ == "__main__":
    test_invalid_profile()
    test_missing_profile()
    test_valid_cmyk_profile()
    test_file_round_trip()