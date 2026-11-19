from lcms2.icc._io import binary as bn


def test_s15fixed16_roundtrip():
    for v in [0.0, 1.0, -1.0, 0.9642, -32768.0, 32767.99998]:
        b = bn.write_s15Fixed16(v)
        assert abs(bn.read_s15Fixed16(b, 0) - v) < 1e-4


def test_u16fixed16_roundtrip():
    for v in [0.0, 1.0, 65535.99998]:
        b = bn.write_u16Fixed16(v)
        assert abs(bn.read_u16Fixed16(b, 0) - v) < 1e-3


def test_xyznumber_roundtrip():
    xyz = (0.9642, 1.0, 0.8249)
    b = bn.write_XYZNumber(xyz)
    got = bn.read_XYZNumber(b, 0)
    for a, c in zip(xyz, got, strict=False):
        assert abs(a - c) < 1e-4


def test_datetime_roundtrip():
    import datetime

    dt = datetime.datetime(2024, 3, 14, 15, 9, 26)
    b = bn.write_dateTime(dt)
    got = bn.read_dateTime(b, 0)
    assert got == dt
