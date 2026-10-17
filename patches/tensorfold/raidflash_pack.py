"""RAIDFLASH in-engine packer: one request → seed (P) + k sister streams (D/P occupancy).

Uses App.run on the CUDA engine thread pool. No HTTP loopback.
Seed must be drafted (max_tokens >= 16) so DFlash keep is resumable.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

SEED_MIN_TOKENS = 16


def _parallel() -> int:
    for key in ("RAIDFLASH_PARALLEL", "RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY"):
        raw = (os.environ.get(key) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 8


def sister_prompt(document: str, i: int, k: int, section: str) -> str:
    return (
        document
        + f"\n\nTarea {i}/{k}: escribe SOLO la seccion '{section}' en 2 oraciones "
        f"sobre el acta anterior. NO escribas las otras secciones ni introduccion. "
        f"Termina con FIN{i}."
    )


def merge_texts(texts: list[str]) -> str:
    parts = []
    for i, t in enumerate(texts, 1):
        body = (t or "").strip()
        marker = f"FIN{i}"
        if body.endswith(marker):
            body = body[: -len(marker)].rstrip()
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def _lane(app, model: str, prompt: str, max_tokens: int, cancelled: Callable[[], bool] | None) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
    }
    result = app.run(body, True, lambda delta: True, cancelled=cancelled)
    stats = result.get("stats") or {}
    pt = int(result.get("prompt_tokens") or 0)
    cached = int(result.get("cached_tokens") or 0)
    text = result.get("content") or result.get("reasoning") or ""
    return {
        "text": text,
        "reasoning": result.get("reasoning") or "",
        "content": result.get("content") or "",
        "finish": result.get("finish"),
        "prompt_tokens": pt,
        "completion_tokens": int(result.get("completion_tokens") or 0),
        "cached_tokens": cached,
        "cached_pct": round(100.0 * cached / pt, 1) if pt else 0.0,
        "prefill_s": stats.get("prefill_s"),
        "decode_s": stats.get("decode_s"),
        "stats": stats,
    }


def complete(
    app,
    document: str,
    sections: list[str],
    *,
    stripe_tokens: int = 80,
    seed_tokens: int = SEED_MIN_TOKENS,
    model: str = "",
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if not document or not str(document).strip():
        raise ValueError("raidflash: document required")
    if not sections:
        raise ValueError("raidflash: sections required")
    k = min(len(sections), max(1, _parallel()))
    secs = [str(s) for s in sections[:k]]
    seed_tokens = max(SEED_MIN_TOKENS, int(seed_tokens))
    stripe_tokens = max(1, int(stripe_tokens))
    model = model or getattr(app, "served_name", "") or ""
    t0 = time.perf_counter()
    seed = _lane(app, model, document, seed_tokens, cancelled)
    rows: list[dict[str, Any] | None] = [None] * k
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            rows[i] = _lane(app, model, sister_prompt(document, i + 1, k, secs[i]), stripe_tokens, cancelled)
        except BaseException as exc:  # noqa: BLE001 — surface to waiter
            errors.append(exc)

    ths = [threading.Thread(target=worker, args=(i,), name=f"raidflash-P-{i}") for i in range(k)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errors:
        raise errors[0]
    sisters = [r for r in rows if r is not None]
    if len(sisters) != k:
        raise RuntimeError("raidflash: sister stream failed")
    wall = time.perf_counter() - t0
    texts = [s["text"] for s in sisters]
    out_tok = seed["completion_tokens"] + sum(s["completion_tokens"] for s in sisters)
    pt = seed["prompt_tokens"] + sum(s["prompt_tokens"] for s in sisters)
    cached = sum(s["cached_tokens"] for s in sisters)
    return {
        "k": k,
        "mode": "raidflash-engine",
        "wall_s": round(wall, 3),
        "text": merge_texts(texts),
        "seed": seed,
        "sisters": sisters,
        "sections": secs,
        "completion_total": out_tok,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "cached_pct_min": min(s["cached_pct"] for s in sisters),
        "agg_tok_s": round(out_tok / wall, 1) if wall else None,
        "seed_tokens": seed_tokens,
        "stripe_tokens": stripe_tokens,
    }
