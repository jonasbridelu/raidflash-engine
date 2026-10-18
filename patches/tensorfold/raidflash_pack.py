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
OUTLINE_SEED_TOKENS = 64
OUTLINE_MARK = "\n\nOUTLINE:\n"
SO_FAR_MARK = "\n\nSO_FAR:\n"


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return default


def _parallel() -> int:
    return _env_int("RAIDFLASH_PARALLEL", "RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY", default=8)


def n_active(obj) -> int:
    """Live GPU streams. Lanes sit on MultiDecoder, not GlmEngine.

    Accepts MultiDecoder, GlmEngine, or App (``app.engine.multi``).
    """
    seen: set[int] = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        live = getattr(cur, "live", None)
        if callable(live):
            try:
                return max(0, int(live()))
            except Exception:
                pass
        lanes = getattr(cur, "lanes", None)
        if lanes:
            return len(lanes)
        nxt = getattr(cur, "multi", None)
        if nxt is None:
            nxt = getattr(cur, "engine", None)
        cur = nxt
    return 0


def idle_slots(obj=None, parallel: int | None = None) -> int:
    cap = max(1, int(parallel if parallel is not None else _parallel()))
    used = n_active(obj) if obj is not None else 0
    return max(0, cap - used)


def _auto_on() -> bool:
    raw = (os.environ.get("RAIDFLASH_AUTO") or os.environ.get("GLM53_TF_RAIDFLASH_AUTO") or "").strip().lower()
    return raw in ("1", "true", "on", "yes")


def document_of(body: dict) -> str:
    rf = body.get("raidflash") if isinstance(body.get("raidflash"), dict) else None
    if isinstance(rf, dict) and rf.get("document"):
        return str(rf.get("document"))
    if body.get("document"):
        return str(body.get("document"))
    msgs = body.get("messages") or []
    if msgs and isinstance(msgs[-1], dict):
        content = msgs[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in (None, "text"):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            if parts:
                return "\n".join(parts)
    prompt = body.get("prompt")
    return str(prompt) if prompt else ""


def wants_pack(body: dict | None) -> bool:
    """Chat enters the packer: raidflash.document/sections/auto, or AUTO long chat."""
    if not isinstance(body, dict):
        return False
    if body.get("stream"):
        return False
    if body.get("tools") or body.get("functions"):
        return False
    rf = body.get("raidflash")
    if isinstance(rf, dict):
        if rf.get("off") or rf.get("enabled") is False:
            return False
        if rf.get("sections") or rf.get("document") or rf.get("auto"):
            return True
    if not _auto_on():
        return False
    min_mt = _env_int("RAIDFLASH_AUTO_MIN_TOKENS", default=256)
    mt = int(body.get("max_tokens") or 0)
    if mt < min_mt:
        return False
    min_chars = _env_int("RAIDFLASH_AUTO_MIN_CHARS", default=512)
    return len(document_of(body)) >= min_chars


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
    n_active: int | None = None,
) -> int:
    """Sisters for this user: EFI split, section count, slot cap, idle leftover.

    n_active=None means the die is empty (tests / one-user idle). A full die
    still returns 1 so the job queues instead of failing.
    """
    cap = max(1, int(parallel if parallel is not None else _parallel()))
    n = n_users if n_users is not None else _env_int("RAIDFLASH_USERS", default=1)
    ph = phase if phase in ("D", "P_cold", "P_hot") else "P_hot"
    ks = split_lanes(n, cap, k_codo(ph, cap))
    ui = max(0, int(user_index))
    mine = ks[ui] if ui < len(ks) else 0
    if mine <= 0:
        mine = 1
    want = cap if int(n_sections) <= 0 else int(n_sections)
    k = min(max(1, want), mine, cap)
    if n_active is not None:
        idle = max(0, cap - max(0, int(n_active)))
        k = min(k, idle) if idle > 0 else 1
    return k


def default_sections(k: int) -> list[str]:
    return [f"part-{i+1}" for i in range(max(1, int(k)))]


def outline_seed_prompt(document: str, sections: list[str]) -> str:
    titles = "\n".join(f"{i}. {s}" for i, s in enumerate(sections, 1))
    return (
        document
        + "\n\nWrite ONLY a numbered outline with exactly these headings, "
        "one line each, no body:\n"
        + titles
        + "\nEnd with FIN0."
    )


def strip_fin(text: str, mark: str) -> str:
    body = (text or "").strip()
    if body.endswith(mark):
        body = body[: -len(mark)].rstrip()
    return body


def bind_outline(seed_prompt_text: str, outline_text: str) -> str:
    return seed_prompt_text + OUTLINE_MARK + strip_fin(outline_text, "FIN0")


def sister_prompt(prefix: str, i: int, k: int, section: str) -> str:
    return (
        prefix
        + f"\n\nTask {i}/{k} section '{section}': continue from the outline and text above. "
        f"Write ONLY this section, up to the token limit. No introduction, no other sections. "
        f"End with FIN{i}."
    )


def merge_texts(texts: list[str]) -> str:
    parts = []
    for i, t in enumerate(texts, 1):
        body = strip_fin(t, f"FIN{i}")
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


def _run_k(app, model, prompts: list[str], max_tokens: int, cancelled) -> list[dict[str, Any]]:
    k = len(prompts)
    rows: list[dict[str, Any] | None] = [None] * k
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            rows[i] = _lane(app, model, prompts[i], max_tokens, cancelled)
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
    return sisters


def complete(
    app,
    document: str,
    sections: list[str],
    *,
    stripe_tokens: int = 256,
    seed_tokens: int = SEED_MIN_TOKENS,
    model: str = "",
    cancelled: Callable[[], bool] | None = None,
    n_users: int | None = None,
    user_index: int = 0,
    phase: str = "P_hot",
    outline: bool = True,
    rounds: int = 1,
    polish: bool = False,
) -> dict[str, Any]:
    if not document or not str(document).strip():
        raise ValueError("raidflash: document required")
    cap = _parallel()
    used0 = n_active(app)
    k = k_for(
        len(sections or []),
        n_users=n_users,
        user_index=user_index,
        phase=phase,
        parallel=cap,
        n_active=used0,
    )
    secs = [str(s) for s in (sections or [])[:k]]
    if len(secs) < k:
        secs.extend(default_sections(k)[len(secs):])
    stripe_tokens = max(1, int(stripe_tokens))
    rounds = max(1, int(rounds))
    per_round = max(1, (stripe_tokens + rounds - 1) // rounds)
    model = model or getattr(app, "served_name", "") or getattr(app, "served", "") or ""
    t0 = time.perf_counter()
    if outline:
        seed_tokens = max(OUTLINE_SEED_TOKENS, int(seed_tokens), SEED_MIN_TOKENS)
        seed_src = outline_seed_prompt(document, secs)
    else:
        seed_tokens = max(SEED_MIN_TOKENS, int(seed_tokens))
        seed_src = document
    seed = _lane(app, model, seed_src, seed_tokens, cancelled)
    used1 = n_active(app)
    k2 = k_for(
        len(secs),
        n_users=n_users,
        user_index=user_index,
        phase=phase,
        parallel=cap,
        n_active=used1,
    )
    if k2 < k:
        k = k2
        secs = secs[:k]
    shared = bind_outline(seed_src, seed["text"]) if outline else document
    accumulated = [""] * k
    sisters: list[dict[str, Any]] = []
    out_tok = seed["completion_tokens"]
    pt = seed["prompt_tokens"]
    cached = 0
    so_far = ""
    for _ in range(rounds):
        prefix = shared + SO_FAR_MARK + so_far if so_far else shared
        prompts = [sister_prompt(prefix, i + 1, k, secs[i]) for i in range(k)]
        sisters = _run_k(app, model, prompts, per_round, cancelled)
        for i, row in enumerate(sisters):
            chunk = strip_fin(row["text"], f"FIN{i + 1}")
            accumulated[i] = f"{accumulated[i]}\n{chunk}".strip() if accumulated[i] else chunk
            out_tok += row["completion_tokens"]
            pt += row["prompt_tokens"]
            cached += row["cached_tokens"]
        so_far = merge_texts(accumulated)
    text = so_far
    polish_row = None
    if polish:
        polish_src = (
            document
            + "\n\nRewrite the draft below into one coherent document. "
            "Keep every fact. Do not add sections.\n\nDRAFT:\n"
            + text
        )
        polish_row = _lane(app, model, polish_src, max(32, stripe_tokens), cancelled)
        text = strip_fin(polish_row["text"], "FIN0") or text
        out_tok += polish_row["completion_tokens"]
        pt += polish_row["prompt_tokens"]
        cached += polish_row["cached_tokens"]
    wall = time.perf_counter() - t0
    n = n_users if n_users is not None else _env_int("RAIDFLASH_USERS", default=1)
    ph = phase if phase in ("D", "P_cold", "P_hot") else "P_hot"
    cache_pcts = [s["cached_pct"] for s in sisters] if sisters else [0.0]
    return {
        "k": k,
        "mode": "raidflash-1.1",
        "wall_s": round(wall, 3),
        "text": text,
        "seed": seed,
        "sisters": sisters,
        "sections": secs,
        "completion_total": out_tok,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "cached_pct_min": min(cache_pcts),
        "agg_tok_s": round(out_tok / wall, 1) if wall else None,
        "seed_tokens": seed_tokens,
        "stripe_tokens": stripe_tokens,
        "outline": bool(outline),
        "rounds": rounds,
        "polish": bool(polish),
        "n_users": int(n),
        "user_index": max(0, int(user_index)),
        "phase": ph,
        "k_plan": split_lanes(n, cap, k_codo(ph, cap)),
        "n_active": used1,
        "idle": max(0, cap - used1),
        "polish_lane": polish_row,
    }
