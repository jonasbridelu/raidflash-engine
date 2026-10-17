"""RAIDFLASH in-engine packer: one request → seed (P) + k sister streams (D/P occupancy).

Uses App.run on the CUDA engine thread pool. No HTTP loopback.
Seed must be drafted (max_tokens >= 16) so DFlash keep is resumable.

k is the EFI injector: one user fills min(PARALLEL, k_codo_phase);
n users split the filled die (3 users on 8 D lanes → [3, 3, 2]).
P-cold extra k does not pay; P-hot sisters inherit the seed.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

SEED_MIN_TOKENS = 16


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return default


def _parallel() -> int:
    return _env_int("RAIDFLASH_PARALLEL", "RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY", default=8)


def split_lanes(n_users: int, parallel: int, k_codo: int) -> list[int]:
    """Fill paying lanes, cap each user at k_codo, split leftover even."""
    n = max(1, int(n_users))
    cap = max(1, int(parallel))
    codo = max(1, int(k_codo))
    active = min(n, cap)
    usable = min(cap, active * codo)
    if usable < active:
        active = usable
    base, rem = divmod(usable, active) if active else (0, 0)
    out = [base + (1 if i < rem else 0) for i in range(active)]
    out.extend([0] * (n - active))
    return out


def k_codo(phase: str, parallel: int) -> int:
    """GB10 live EFI 2026-10-17: D still pays at 8, P-cold at 1, P-hot at 8."""
    if phase == "P_cold":
        return _env_int("RAIDFLASH_K_CODO_P_COLD", default=1)
    if phase == "D":
        return _env_int("RAIDFLASH_K_CODO_D", default=parallel)
    return _env_int("RAIDFLASH_K_CODO_P", "RAIDFLASH_K_CODO_P_HOT", default=parallel)


def k_for(
    n_sections: int,
    n_users: int | None = None,
    user_index: int = 0,
    phase: str = "P_hot",
    parallel: int | None = None,
) -> int:
    """Sisters for this user: EFI split, then section count, then slot cap."""
    cap = max(1, int(parallel if parallel is not None else _parallel()))
    n = n_users if n_users is not None else _env_int("RAIDFLASH_USERS", default=1)
    ph = phase if phase in ("D", "P_cold", "P_hot") else "P_hot"
    ks = split_lanes(n, cap, k_codo(ph, cap))
    ui = max(0, int(user_index))
    mine = ks[ui] if ui < len(ks) else 0
    if mine <= 0:
        mine = 1
    return min(max(1, int(n_sections)), mine, cap)


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
    n_users: int | None = None,
    user_index: int = 0,
    phase: str = "P_hot",
) -> dict[str, Any]:
    if not document or not str(document).strip():
        raise ValueError("raidflash: document required")
    if not sections:
        raise ValueError("raidflash: sections required")
    k = k_for(len(sections), n_users=n_users, user_index=user_index, phase=phase)
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
    n = n_users if n_users is not None else _env_int("RAIDFLASH_USERS", default=1)
    ph = phase if phase in ("D", "P_cold", "P_hot") else "P_hot"
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
        "n_users": int(n),
        "user_index": max(0, int(user_index)),
        "phase": ph,
        "k_plan": split_lanes(n, _parallel(), k_codo(ph, _parallel())),
    }
