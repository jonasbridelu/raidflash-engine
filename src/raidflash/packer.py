"""Packer: one user POST → seed (P) + k sister streams.

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
from .choose import choose_k
from .prefill import (
    OUTLINE_SEED_TOKENS,
    SEED_MIN_TOKENS,
    SO_FAR_MARK,
    bind_outline,
    merge_sisters,
    outline_seed_prompt,
    seed_prompt,
    sister_prompts,
    strip_fin,
    thought_sections,
)


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
        stripe_tokens: int = 256,
        seed_tokens: int = SEED_MIN_TOKENS,
        outline: bool = True,
        rounds: int = 1,
        polish: bool = False,
    ) -> dict[str, Any]:
        """P seed then k sisters. k = occupancy.p_open() after seed (dynamic).

        Tries in-engine `/v1/raidflash/complete` first. HTTP fan-out if 404.
        v1.1: outline seed (default), optional sync rounds, optional polish.
        """
        if not sections:
            raise ValueError("sections required")
        k = self._k(len(sections), "P_hot")
        secs = sections[:k]
        if self.engine:
            try:
                return self._complete_engine(
                    document, secs, stripe_tokens, seed_tokens,
                    outline=outline, rounds=rounds, polish=polish,
                )
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        return self._complete_http(
            document, secs, stripe_tokens, seed_tokens,
            outline=outline, rounds=rounds, polish=polish,
        )

    def think(
        self,
        question: str,
        k: int | None = None,
        stripe_tokens: int = 256,
        seed_tokens: int = SEED_MIN_TOKENS,
    ) -> dict[str, Any]:
        """One question → seed + k thought-sisters (P-hot). Default k=4 (thinking peak)."""
        q = (question or "").strip()
        if not q:
            raise ValueError("question required")
        if k is None:
            k = choose_k(kind="decode", thinking=True, max_tokens=80)
        k = max(1, int(k))
        return self.complete(q, thought_sections(q, k), stripe_tokens=stripe_tokens, seed_tokens=seed_tokens)

    def fork(
        self,
        document: str,
        k: int = 6,
        stripe_tokens: int = 100,
        leader_tokens: int = 48,
        temperature: float | None = 0.8,
    ) -> dict[str, Any]:
        """Leader tokens stay on the prompt; k lanes continue that chain."""
        doc = (document or "").strip()
        if not doc:
            raise ValueError("document required")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": doc}],
            "max_tokens": int(stripe_tokens),
            "temperature": 0.8 if temperature is None else float(temperature),
            "raidflash": {
                "fork": True,
                "document": doc,
                "k": max(1, int(k)),
                "stripe_tokens": int(stripe_tokens),
                "leader_tokens": max(0, int(leader_tokens)),
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
            "version": "0.3.1",
            "mode": rf.get("mode") or "raidflash-fork",
            "k": rf.get("k") or k,
            "text": text,
            "lanes": rf.get("lanes") or [],
            "leader_text": rf.get("leader_text") or "",
            "share_len": rf.get("share_len"),
            "kv_continue": rf.get("kv_continue"),
            "wall_s": rf.get("wall_s") or round(wall, 3),
            "completion_total": us.get("completion_tokens") or rf.get("completion_total"),
            "agg_tok_s": rf.get("agg_tok_s"),
            "cached_pct_min": rf.get("cached_pct_min"),
        }

    def _complete_engine(
        self,
        document: str,
        secs: list[str],
        stripe_tokens: int,
        seed_tokens: int,
        outline: bool = True,
        rounds: int = 1,
        polish: bool = False,
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
                "outline": bool(outline),
                "rounds": max(1, int(rounds)),
                "polish": bool(polish),
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
            "version": "0.3.1",
            "mode": rf.get("mode") or "raidflash-1.1",
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
        outline: bool = True,
        rounds: int = 1,
        polish: bool = False,
    ) -> dict[str, Any]:
        k = len(secs)
        t0 = time.time()
        rounds = max(1, int(rounds))
        per_round = max(1, (int(stripe_tokens) + rounds - 1) // rounds)
        if outline:
            src = outline_seed_prompt(document, secs)
            seed = self._post(src, max(OUTLINE_SEED_TOKENS, int(seed_tokens), SEED_MIN_TOKENS))
            shared = bind_outline(src, seed["text"])
        else:
            seed = self._post(seed_prompt(document), max(SEED_MIN_TOKENS, int(seed_tokens)))
            shared = document
        accumulated = [""] * k
        sisters: list[dict] = []
        so_far = ""
        for _ in range(rounds):
            prefix = shared + SO_FAR_MARK + so_far if so_far else shared
            prompts = sister_prompts(prefix, secs)
            results: list[dict | None] = [None] * k

            def worker(i: int, prompt: str) -> None:
                results[i] = self._post(prompt, per_round)

            threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            sisters = [r for r in results if r is not None]
            if len(sisters) != k:
                raise RuntimeError("sister stream failed")
            for i, row in enumerate(sisters):
                chunk = strip_fin(row["text"], f"FIN{i + 1}")
                accumulated[i] = f"{accumulated[i]}\n{chunk}".strip() if accumulated[i] else chunk
            so_far = merge_sisters(accumulated)
        text = so_far
        if polish:
            polish_src = (
                document
                + "\n\nRewrite the draft below into one coherent document. "
                "Keep every fact. Do not add sections.\n\nDRAFT:\n"
                + text
            )
            polished = self._post(polish_src, max(32, int(stripe_tokens)))
            text = strip_fin(polished["text"], "FIN0") or text
        wall = time.time() - t0
        out_tok = seed["completion_tokens"] + sum(s["completion_tokens"] for s in sisters)
        return {
            "version": "0.3.1",
            "mode": "raidflash-1.1",
            "k": k,
            "text": text,
            "wall_s": round(wall, 3),
            "seed": seed,
            "sisters": sisters,
            "completion_total": out_tok,
            "agg_tok_s": round(out_tok / wall, 1) if wall else None,
            "cached_pct_min": min(s["cached_pct"] for s in sisters),
            "outline": bool(outline),
            "rounds": rounds,
            "polish": bool(polish),
        }
