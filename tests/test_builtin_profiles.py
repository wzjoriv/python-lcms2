import lcms2

def test_version():
    assert len(lcms2.get_version()) >= 3


def is_valid_handle(h):
    t = type(h)
    return t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'


def test_default_profile():
    p = lcms2.Profile()
    assert "NULL" in p.name
    assert "use freely" in p.copyright
    assert is_valid_handle(p.handle)


def test_builtins():
    for profile_name in ("sRGB", "Lab", "XYZ"):
        p = lcms2.Profile(profile_name)
        assert profile_name in p.name
        assert "use freely" in p.copyright
        assert is_valid_handle(p.handle)


if __name__ == "__main__":
    test_default_profile()
    test_builtins()