"""EFI injector inside the in-engine overlay (no TensorFold import)."""
import importlib.util
from pathlib import Path


def _pack():
    p = Path(__file__).resolve().parents[1] / "patches" / "tensorfold" / "raidflash_pack.py"
    spec = importlib.util.spec_from_file_location("raidflash_pack_overlay", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_user_fills_the_die():
    p = _pack()
    assert p.k_for(8, n_users=1, parallel=8) == 8
    assert p.k_for(3, n_users=1, parallel=8) == 3


def test_three_users_split_eight():
    p = _pack()
    assert p.split_lanes(3, 8, 8) == [3, 3, 2]
    assert p.k_for(8, n_users=3, user_index=0, parallel=8) == 3
    assert p.k_for(8, n_users=3, user_index=2, parallel=8) == 2


def test_p_cold_one_lane():
    p = _pack()
    assert p.k_for(8, n_users=3, user_index=0, phase="P_cold", parallel=8) == 1


def test_more_sections_than_codo_still_caps():
    p = _pack()
    assert p.k_for(64, n_users=1, phase="D", parallel=8) == 8


def test_empty_sections_fills_the_die():
    p = _pack()
    assert p.k_for(0, n_users=1, parallel=8) == 8
    assert p.default_sections(8) == [f"parte-{i}" for i in range(1, 9)]


def test_idle_cap_leaves_room_for_other_users():
    p = _pack()
    assert p.k_for(0, n_users=1, parallel=8, n_active=0) == 8
    assert p.k_for(0, n_users=1, parallel=8, n_active=3) == 5
    assert p.k_for(8, n_users=3, user_index=0, parallel=8, n_active=6) == 2
    assert p.k_for(0, n_users=1, parallel=8, n_active=8) == 1


def test_n_active_walks_engine_multi_lanes():
    p = _pack()

    class Lane:
        def __init__(self):
            self.s = object()
            self.decoding = True

    class Multi:
        def __init__(self):
            self.lanes = {0: Lane(), 1: Lane(), 2: Lane()}

        def live(self):
            return len(self.lanes)

    class Engine:
        def __init__(self):
            self.multi = Multi()

    class App:
        def __init__(self):
            self.engine = Engine()

    assert p.n_active(None) == 0
    assert p.n_active(App()) == 3
    assert p.n_active(Engine()) == 3
    assert p.idle_slots(App(), parallel=8) == 5


def test_document_of_complete_and_chat():
    p = _pack()
    assert p.document_of({"document": "acta plana"}) == "acta plana"
    assert p.document_of({"raidflash": {"document": "nested"}}) == "nested"
    assert p.document_of({"messages": [{"role": "user", "content": "hola"}]}) == "hola"


def test_wants_pack_document_without_sections():
    p = _pack()
    assert p.wants_pack({"raidflash": {"document": "acta"}}) is True
    assert p.wants_pack({"raidflash": {"sections": ["a"]}}) is True
    assert p.wants_pack({"messages": [{"role": "user", "content": "hola"}], "max_tokens": 16}) is False
    assert p.wants_pack({"stream": True, "raidflash": {"document": "x"}}) is False
    assert p.wants_pack({"tools": [{}], "raidflash": {"document": "x"}}) is False


def test_wants_pack_auto_long_chat(monkeypatch):
    p = _pack()
    monkeypatch.setenv("RAIDFLASH_AUTO", "1")
    long_doc = "acta " * 200
    assert len(long_doc) >= 512
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 1024}) is True
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 16}) is False
    monkeypatch.delenv("RAIDFLASH_AUTO", raising=False)
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 1024}) is False
