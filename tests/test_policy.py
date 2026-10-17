from raidflash.policy import adapt_code, depth_next, enabled


def test_flag_off_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("RAIDFLASH", raising=False)
    monkeypatch.delenv("GLM53_TF_RAIDFLASH", raising=False)
    monkeypatch.setenv("RAIDFLASH_FLAG", str(tmp_path / "missing.on"))
    code = [16, 7, 300000, 0]
    out, serial = adapt_code(code, 80, None, None)
    assert out == code and serial is False
    assert enabled() is False


def test_depth_next_most3_identical_to_vendor():
    assert depth_next(3, 0.50) == 1
    assert depth_next(3, 0.85) == 2
    assert depth_next(3, 0.95) == 3
    assert depth_next(3, 0.95, fixed=True) == 3


def test_depth_next_most7_heats_to_seven():
    assert depth_next(7, 0.50) == 1
    assert depth_next(7, 0.85) == 4
    assert depth_next(7, 0.95) == 7
    assert depth_next(7, 0.1, fixed=True) == 7
