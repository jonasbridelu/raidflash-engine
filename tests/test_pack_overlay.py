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
    assert p.default_sections(8) == [f"part-{i}" for i in range(1, 9)]


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
    assert p.document_of({"document": "plain minutes"}) == "plain minutes"
    assert p.document_of({"raidflash": {"document": "nested"}}) == "nested"
    assert p.document_of({"messages": [{"role": "user", "content": "hola"}]}) == "hola"


def test_wants_pack_document_without_sections():
    p = _pack()
    assert p.wants_pack({"raidflash": {"document": "minutes"}}) is True
    assert p.wants_pack({"raidflash": {"sections": ["a"]}}) is True
    assert p.wants_pack({"messages": [{"role": "user", "content": "hola"}], "max_tokens": 16}) is False
    assert p.wants_pack({"stream": True, "raidflash": {"document": "x"}}) is False
    assert p.wants_pack({"tools": [{}], "raidflash": {"document": "x"}}) is False


def test_complete_outline_sisters_see_spine():
    p = _pack()

    class App:
        served_name = "glm"

        def __init__(self):
            self.calls = []

        def run(self, body, *_a, **_k):
            content = body["messages"][0]["content"]
            self.calls.append(content)
            n = len(self.calls)
            if n == 1:
                return {
                    "content": "1. alpha\n2. beta FIN0",
                    "completion_tokens": 16,
                    "prompt_tokens": 40,
                    "cached_tokens": 0,
                    "stats": {},
                }
            return {
                "content": f"body-{n} FIN{n - 1}",
                "completion_tokens": 8,
                "prompt_tokens": 50,
                "cached_tokens": 45,
                "stats": {},
            }

    app = App()
    out = p.complete(app, "Doc text", ["alpha", "beta"], stripe_tokens=8, seed_tokens=16)
    assert out["mode"] == "raidflash-1.1"
    assert out["outline"] is True
    assert out["rounds"] == 1
    assert out["k"] == 2
    assert "Write ONLY a numbered outline" in app.calls[0]
    assert "1. alpha" in app.calls[0]
    for sister in app.calls[1:]:
        assert sister.startswith(app.calls[0])
        assert "OUTLINE:" in sister
        assert "1. alpha" in sister
    assert "body-2" in out["text"] and "body-3" in out["text"]


def test_complete_rounds_second_wave_sees_so_far():
    p = _pack()

    class App:
        served_name = "glm"

        def __init__(self):
            self.calls = []

        def run(self, body, *_a, **_k):
            content = body["messages"][0]["content"]
            self.calls.append(content)
            n = len(self.calls)
            if n == 1:
                return {
                    "content": "1. A\n2. B FIN0",
                    "completion_tokens": 8,
                    "prompt_tokens": 20,
                    "cached_tokens": 0,
                    "stats": {},
                }
            return {
                "content": f"r{n} FIN{(n - 2) % 2 + 1}",
                "completion_tokens": 4,
                "prompt_tokens": 30,
                "cached_tokens": 20,
                "stats": {},
            }

    app = App()
    out = p.complete(
        app, "Doc", ["A", "B"], stripe_tokens=8, seed_tokens=16, rounds=2, outline=True
    )
    assert out["rounds"] == 2
    assert len(app.calls) == 1 + 4  # seed + 2 sisters × 2 rounds
    second_wave = app.calls[3:]
    assert all("SO_FAR:" in c for c in second_wave)


def test_wants_pack_auto_long_chat(monkeypatch):
    p = _pack()
    monkeypatch.setenv("RAIDFLASH_AUTO", "1")
    long_doc = "minutes " * 200
    assert len(long_doc) >= 512
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 1024}) is True
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 16}) is False
    monkeypatch.delenv("RAIDFLASH_AUTO", raising=False)
    assert p.wants_pack({"messages": [{"role": "user", "content": long_doc}], "max_tokens": 1024}) is False


def test_continue_ids_appends_leader():
    p = _pack()
    assert p.continue_ids([1, 2, 3], [9, 8]) == [1, 2, 3, 9, 8]


def test_wants_pack_fork():
    p = _pack()
    assert p.wants_pack({"raidflash": {"fork": True, "document": "essay"}}) is True


def test_run_fork_keeps_leader_in_kv_not_user_text():
    from types import SimpleNamespace

    p = _pack()

    class App:
        served = "m"

        def __init__(self):
            self.calls = []

        def prepare(self, body, chat):
            return SimpleNamespace(prompt=[10, 11, 12, 13], max_tokens=body.get("max_tokens", 8), sampling=None)

        def run_job_lanes(self, bodies, cancelled=None):
            self.calls.append(bodies)
            ntok = int(bodies[0].get("max_tokens") or 4)
            out = []
            for i, b in enumerate(bodies):
                tokens = list(range(200 + i * 20, 200 + i * 20 + ntok))
                out.append({
                    "text": f"gen{i}-" + "x" * ntok,
                    "content": f"gen{i}",
                    "reasoning": "",
                    "finish": "stop",
                    "prompt_tokens": len(b.get("prompt_ids") or [10, 11, 12, 13]),
                    "completion_tokens": ntok,
                    "cached_tokens": 0,
                    "cached_pct": 0.0,
                    "prefill_s": 0.01,
                    "decode_s": 0.1,
                    "stats": {},
                    "job": b.get("tf_raidflash"),
                    "tokens": tokens,
                })
            return out

    app = App()
    out = p.run_fork(app, "Write a Cusco essay", k=3, stripe_tokens=5, leader_tokens=8)
    assert out["mode"] == "raidflash-fork"
    assert out["kv_continue"] is True
    assert out["share_len"] == 12
    assert len(app.calls) == 2
    sister_bodies = app.calls[1]
    expected = [10, 11, 12, 13] + list(range(200, 208))
    for b in sister_bodies:
        assert b["prompt_ids"] == expected
        assert "gen0" not in b["messages"][0]["content"]
        assert b["tf_raidflash"]["share_len"] == 12
