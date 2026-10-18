"""Pick k from measured GB10 8-seat data. This is what you implement.

One document  → seed k=1 (P-cold sequential), then k=8 sisters (P-hot).
Many cold docs → k=8 (prefill tok/s 2.3k → 8.8k → 15.7k).
Decode        → k=8 even with fat KV (8k history still 59 tok/s vs C1 ~12–34).
Short thinking (mt≤80) → k=4 (peak 115 tok/s; k=8 cuts `most` via WINDOW/n).
Long thinking → k=8 occupancy.
3 users       → [3,3,2] on D/P-hot/many; [1,1,1] on one-doc seed.
"""
from __future__ import annotations

from typing import Literal

from .efi import split_lanes

Kind = Literal["decode", "prefill_one", "prefill_many", "packer"]

# Expression grid 2026-10-16: thinking k=4 mt=80 = 115.4 tok/s (peak);
# k=8 mt=80 = 110.7. Long thinking (256+) occupancy k=8 wins (115 @256).
SHORT_THINK_TOKENS = 80
SHORT_THINK_K = 4


def k_codo_for(
    kind: Kind,
    parallel: int = 8,
    *,
    thinking: bool = False,
    max_tokens: int | None = None,
) -> int:
    if kind == "prefill_one":
        return 1
    cap = max(1, int(parallel))
    if kind == "decode" and thinking:
        mt = cap * 32 if max_tokens is None else int(max_tokens)
        if mt <= SHORT_THINK_TOKENS:
            return min(SHORT_THINK_K, cap)
    return cap


def choose_k(
    *,
    kind: Kind,
    n_users: int = 1,
    user_index: int = 0,
    parallel: int = 8,
    n_work: int | None = None,
    thinking: bool = False,
    max_tokens: int | None = None,
) -> int:
    """Lanes this user should open. Cap by remaining work items if given."""
    cap = max(1, int(parallel))
    codo = k_codo_for(kind, cap, thinking=thinking, max_tokens=max_tokens)
    ks = split_lanes(n_users, cap, codo)
    ui = max(0, int(user_index))
    mine = ks[ui] if ui < len(ks) else 0
    if mine <= 0:
        mine = 1
    if n_work is not None:
        mine = min(mine, max(1, int(n_work)))
    return min(mine, cap, codo)


def seed_then_sisters(*, n_users: int = 1, user_index: int = 0, parallel: int = 8, n_sections: int = 8) -> dict:
    """The real RAIDFLASH job: one cold seed, then P-hot sisters."""
    return {
        "seed_k": choose_k(kind="prefill_one", n_users=n_users, user_index=user_index, parallel=parallel),
        "sister_k": choose_k(
            kind="packer",
            n_users=n_users,
            user_index=user_index,
            parallel=parallel,
            n_work=n_sections,
        ),
        "seed_tokens": 16,
    }
