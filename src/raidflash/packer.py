"""v0.1 packer: one user POST → seed (P) + k sister streams.

Prefers in-engine `/v1/raidflash/complete` (one POST, App.run lanes).
Falls back to OpenAI chat fan-out when the overlay is absent.
k = min(sections, occupancy.p_open()) — dynamic, not a hardcoded 8.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .efi import EfiMap
from .occupancy import Occupancy


def _load_efi() -> EfiMap:
    path = (os.environ.get("RAIDFLASH_EFI") or "").strip()
    if path:
        try:
            return EfiMap.load(path)
        except OSError:
            pass
    return EfiMap.gb10()
from .prefill import SEED_MIN_TOKENS, merge_sisters, seed_prompt, sister_prompts


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return default


class Packer:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        occupancy: Occupancy | None = None,
        timeout: float = 300.0,
        engine: bool = True,
        efi: EfiMap | None = None,
        n_users: int | None = None,
        user_index: int = 0,
    ):
        self.base_url = (base_url or os.environ.get("RAIDFLASH_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.model = model or os.environ.get("RAIDFLASH_MODEL", "GLM-5.3-Flash-EXL3")
        self.occ = occupancy or Occupancy.from_env()
        self.timeout = timeout
        self.engine = engine
        self.efi = efi or _load_efi()
        self.n_users = n_users if n_users is not None else _env_int("RAIDFLASH_USERS", default=1)
        self.user_index = max(0, int(user_index))

    def _k(self, n_sections: int, phase: str = "P_hot") -> int:
        """Sisters for this user: EFI split, then occupancy cap, then section count."""
        ks = self.efi.allocate(self.n_users, phase if phase in ("D", "P_cold", "P_hot") else "P_hot")
        mine = ks[self.user_index] if self.user_index < len(ks) else 0
        if mine <= 0:
            mine = 1
        cap = max(1, self.occ.p_open() or self.occ.parallel)
        return min(n_sections, mine, cap)

    def _post(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        max_tokens = int(max_tokens)
        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode())
        wall = time.time() - t0
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        us = resp.get("usage") or {}
        tf = resp.get("tensorfold") or {}
        pt = us.get("prompt_tokens") or 0
        cached = (us.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return {
            "text": text,
            "wall_s": round(wall, 3),
            "prefill_s": tf.get("prefill_s"),
            "decode_s": tf.get("decode_s"),
            "prompt_tokens": pt,
            "cached_tokens": cached,
            "cached_pct": round(100.0 * cached / pt, 1) if pt else 0.0,
            "completion_tokens": us.get("completion_tokens") or 0,
        }

    def complete(
        self,
        document: str,
        sections: list[str],
        stripe_tokens: int = 80,
        seed_tokens: int = SEED_MIN_TOKENS,
    ) -> dict[str, Any]:
        """P seed then k sisters. k = occupancy.p_open() after seed (dynamic).

        Tries in-engine `/v1/raidflash/complete` first. HTTP fan-out if 404.
        """
        if not sections:
            raise ValueError("sections required")
        k = self._k(len(sections), "P_hot")
        secs = sections[:k]
        if self.engine:
            try:
                return self._complete_engine(document, secs, stripe_tokens, seed_tokens)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        return self._complete_http(document, secs, stripe_tokens, seed_tokens)

    def _complete_engine(
        self,
        document: str,
        secs: list[str],
        stripe_tokens: int,
        seed_tokens: int,
    ) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "raidflash": {
                "document": document,
                "sections": secs,
                "stripe_tokens": int(stripe_tokens),
                "seed_tokens": max(SEED_MIN_TOKENS, int(seed_tokens)),
                "n_users": int(self.n_users),
                "user_index": int(self.user_index),
                "phase": "P_hot",
            },
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/raidflash/complete",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode())
        wall = time.time() - t0
        rf = resp.get("raidflash") or {}
        us = resp.get("usage") or {}
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return {
            "version": "0.1.0",
            "mode": rf.get("mode") or "raidflash-engine",
            "k": rf.get("k") or len(secs),
            "text": text,
            "wall_s": rf.get("wall_s") or round(wall, 3),
            "seed": rf.get("seed") or {},
            "sisters": rf.get("sisters") or [],
            "completion_total": us.get("completion_tokens") or rf.get("completion_total"),
            "agg_tok_s": rf.get("agg_tok_s"),
            "cached_pct_min": rf.get("cached_pct_min"),
            "cached_tokens": (us.get("prompt_tokens_details") or {}).get("cached_tokens")
            or rf.get("cached_tokens"),
        }

    def _complete_http(
        self,
        document: str,
        secs: list[str],
        stripe_tokens: int,
        seed_tokens: int,
    ) -> dict[str, Any]:
        k = len(secs)
        t0 = time.time()
        seed = self._post(seed_prompt(document), max(SEED_MIN_TOKENS, int(seed_tokens)))
        prompts = sister_prompts(document, secs)
        results: list[dict | None] = [None] * k

        def worker(i: int, prompt: str) -> None:
            results[i] = self._post(prompt, int(stripe_tokens))

        threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        sisters = [r for r in results if r is not None]
        if len(sisters) != k:
            raise RuntimeError("sister stream failed")
        text = merge_sisters([s["text"] for s in sisters])
        wall = time.time() - t0
        out_tok = seed["completion_tokens"] + sum(s["completion_tokens"] for s in sisters)
        return {
            "version": "0.1.0",
            "mode": "raidflash-P",
            "k": k,
            "text": text,
            "wall_s": round(wall, 3),
            "seed": seed,
            "sisters": sisters,
            "completion_total": out_tok,
            "agg_tok_s": round(out_tok / wall, 1) if wall else None,
            "cached_pct_min": min(s["cached_pct"] for s in sisters),
        }
