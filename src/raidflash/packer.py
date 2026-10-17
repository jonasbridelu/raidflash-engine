"""v0.1 packer: one user POST → seed (P) + k sister streams (slots of --parallel).

This is the product API. The engine stays OpenAI-compatible. RAIDFLASH fills
C_max with that user's stripes instead of leaving slots for other tenants.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .occupancy import Occupancy
from .prefill import SEED_MIN_TOKENS, merge_sisters, seed_prompt, sister_prompts


class Packer:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        occupancy: Occupancy | None = None,
        timeout: float = 300.0,
    ):
        self.base_url = (base_url or os.environ.get("RAIDFLASH_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.model = model or os.environ.get("RAIDFLASH_MODEL", "GLM-5.3-Flash-EXL3")
        self.occ = occupancy or Occupancy(parallel=int(os.environ.get("RAIDFLASH_PARALLEL", "4")))
        self.timeout = timeout

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
        text = msg.get("content") or ""
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
        """P seed then k sisters in parallel. k = occupancy.p_open() after seed.

        Returns merged text plus per-lane metrics. v0.1 does not split a generic
        chat; it packs a document+sections job into all free slots.
        """
        if not sections:
            raise ValueError("sections required")
        k = min(len(sections), max(1, self.occ.p_open() or self.occ.parallel))
        secs = sections[:k]
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
