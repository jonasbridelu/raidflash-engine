from raidflash.policy import _n_active, adapt_code, bind_d, depth_next, enabled, most_for_lanes


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


def test_most_for_lanes_window32():
    assert most_for_lanes(1, window=32, d_most=7) == 7
    assert most_for_lanes(4, window=32, d_most=7) == 7
    assert most_for_lanes(8, window=32, d_most=7) == 4
    assert most_for_lanes(16, window=32, d_most=7) == 2


def test_bind_d_k8_keeps_four_not_three(monkeypatch):
    monkeypatch.setenv("RAIDFLASH", "1")
    monkeypatch.setenv("RAIDFLASH_TURBO", "1")
    monkeypatch.setenv("RAIDFLASH_WINDOW", "32")
    monkeypatch.setenv("RAIDFLASH_D_MOST", "7")

    class Lane:
        decoding = True
        s = object()

    class Multi:
        lanes = {i: Lane() for i in range(8)}

        def live(self):
            return 8

    class Pol:
        most = 7
        fixed = True

    p = bind_d(Pol(), Multi())
    assert p.most == 4
    assert p.fixed is False


def test_bind_d_k4_keeps_seven(monkeypatch):
    monkeypatch.setenv("RAIDFLASH", "1")
    monkeypatch.setenv("RAIDFLASH_TURBO", "1")
    monkeypatch.setenv("RAIDFLASH_WINDOW", "32")

    class Lane:
        decoding = True
        s = object()

    class Multi:
        lanes = {i: Lane() for i in range(4)}

        def live(self):
            return 4

    class Pol:
        most = 3
        fixed = True

    p = bind_d(Pol(), Multi())
    assert p.most == 7


def test_n_active_reads_multi_lanes_not_engine():
    class Lane:
        def __init__(self):
            self.s = object()
            self.decoding = True

    class Multi:
        def __init__(self):
            self.lanes = {i: Lane() for i in range(4)}

        def live(self):
            return len(self.lanes)

    class Engine:
        def __init__(self):
            self.multi = Multi()

    assert _n_active(None) == 0
    assert _n_active(Engine()) == 4
    assert _n_active(Engine().multi) == 4
