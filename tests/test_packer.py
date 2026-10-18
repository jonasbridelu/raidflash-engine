import json
from email.message import EmailMessage
from io import BytesIO

import urllib.error

from raidflash.packer import Packer
from raidflash.prefill import thought_sections


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_complete_prefers_engine(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return _Resp({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 80, "prompt_tokens_details": {"cached_tokens": 51200}},
            "raidflash": {
                "k": 2, "mode": "raidflash-engine", "wall_s": 1.2,
                "agg_tok_s": 70.0, "cached_pct_min": 99.1, "seed": {}, "sisters": [{}, {}],
            },
        })

    monkeypatch.setattr("raidflash.packer.urllib.request.urlopen", fake_urlopen)
    out = Packer(base_url="http://engine").complete("doc", ["a", "b"])
    assert calls == ["http://engine/v1/raidflash/complete"]
    assert out["mode"] == "raidflash-engine"
    assert out["k"] == 2
    assert out["cached_pct_min"] == 99.1


def test_complete_http_fallback_on_404(monkeypatch):
    n = {"i": 0}

    def fake_urlopen(req, timeout=0):
        n["i"] += 1
        if req.full_url.endswith("/v1/raidflash/complete"):
            raise urllib.error.HTTPError(
                req.full_url, 404, "no", EmailMessage(), BytesIO(b"{}")
            )
        return _Resp({
            "choices": [{"message": {"content": f"t{n['i']} FIN{n['i']-1}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 16, "prompt_tokens_details": {"cached_tokens": 90}},
        })

    monkeypatch.setattr("raidflash.packer.urllib.request.urlopen", fake_urlopen)
    out = Packer(base_url="http://plain").complete("doc", ["a", "b"], stripe_tokens=16)
    assert out["mode"] == "raidflash-1.1"
    assert out["k"] == 2
    assert n["i"] >= 3  # 404 + seed + 2 sisters


def test_three_users_send_three_sisters(monkeypatch):
    bodies = []

    def fake_urlopen(req, timeout=0):
        bodies.append(json.loads(req.data.decode()))
        return _Resp({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 16},
            "raidflash": {"k": 3, "mode": "raidflash-engine", "wall_s": 1.0,
                          "agg_tok_s": 50.0, "cached_pct_min": 99.0, "seed": {}, "sisters": [{}, {}, {}]},
        })

    monkeypatch.setattr("raidflash.packer.urllib.request.urlopen", fake_urlopen)
    secs = [f"s{i}" for i in range(8)]
    Packer(base_url="http://engine", n_users=3, user_index=0).complete("doc", secs)
    sent = bodies[0]["raidflash"]["sections"]
    assert len(sent) == 3
    Packer(base_url="http://engine", n_users=3, user_index=2).complete("doc", secs)
    assert len(bodies[1]["raidflash"]["sections"]) == 2  # [3,3,2]
    assert bodies[0]["raidflash"]["n_users"] == 3
    assert bodies[0]["raidflash"]["user_index"] == 0
    assert bodies[1]["raidflash"]["user_index"] == 2
    assert bodies[0]["raidflash"]["phase"] == "P_hot"


def test_thought_sections_are_prefix_sisters():
    secs = thought_sections("why is 2+2=4", k=4)
    assert len(secs) == 4
    assert all("why is 2+2=4" in s for s in secs)
    assert len(set(secs)) == 4


def test_think_opens_four_sisters(monkeypatch):
    bodies = []

    def fake_urlopen(req, timeout=0):
        bodies.append(json.loads(req.data.decode()))
        return _Resp({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 16},
            "raidflash": {"k": 4, "mode": "raidflash-engine", "wall_s": 1.0,
                          "agg_tok_s": 90.0, "cached_pct_min": 99.0, "seed": {},
                          "sisters": [{}, {}, {}, {}]},
        })

    monkeypatch.setattr("raidflash.packer.urllib.request.urlopen", fake_urlopen)
    out = Packer(base_url="http://engine").think("por que el cielo es azul")
    sent = bodies[0]["raidflash"]["sections"]
    assert len(sent) == 4
    assert out["k"] == 4


def test_fork_posts_kv_continue_body(monkeypatch):
    bodies = []

    def fake_urlopen(req, timeout=0):
        bodies.append(json.loads(req.data.decode()))
        return _Resp({
            "choices": [{"message": {"content": "lane text"}}],
            "usage": {"completion_tokens": 648},
            "raidflash": {
                "k": 6, "mode": "raidflash-fork", "wall_s": 9.7,
                "agg_tok_s": 66.6, "kv_continue": True, "share_len": 97,
                "lanes": ["a", "b", "c", "d", "e", "f"], "leader_text": "worn",
            },
        })

    monkeypatch.setattr("raidflash.packer.urllib.request.urlopen", fake_urlopen)
    out = Packer(base_url="http://engine").fork("Cusco essay", k=6, leader_tokens=48)
    rf = bodies[0]["raidflash"]
    assert rf["fork"] is True
    assert rf["leader_tokens"] == 48
    assert rf["k"] == 6
    assert out["kv_continue"] is True
    assert out["share_len"] == 97
    assert len(out["lanes"]) == 6
