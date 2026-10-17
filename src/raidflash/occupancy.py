"""Occupancy: fill free streams of ONE user until adding another does not scale.

Recipe knobs (--parallel, dflash most) are CAPS. CUDA already parallelizes
inside the kernel. RAIDFLASH spends unused stream slots on that user's job.
"""
from __future__ import annotations

from dataclasses import dataclass


def k_target(parallel: int, n_active: int, k_codo: int) -> int:
    """How many NEW streams this user should open right now."""
    want = min(max(1, parallel), max(1, k_codo))
    used = max(0, n_active)
    if used >= want:
        return 0
    return want - used


@dataclass
class Occupancy:
    parallel: int = 4
    k_codo_d: int = 4
    k_codo_p: int = 4  # v0.1: P sisters bounded by --parallel (knee of the slot cap)
    n_active: int = 0

    def idle(self) -> int:
        return max(0, self.parallel - self.n_active)

    def d_open(self) -> int:
        return k_target(self.parallel, self.n_active, self.k_codo_d)

    def p_open(self) -> int:
        return k_target(self.parallel, self.n_active, min(self.k_codo_p, self.parallel))

    def filling(self) -> bool:
        return self.idle() == 0
