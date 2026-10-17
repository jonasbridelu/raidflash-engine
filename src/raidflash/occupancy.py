"""Occupancy: fill free streams of ONE user until adding another does not scale.

`--parallel` is a CAP (engine slots). k is DYNAMIC: P and D have different
knees, and the knee moves with the die (GB10 ~8, a B300 may want 32/64/120).
Open lanes while aggregate tok/s still pays; stop at k_codo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def k_target(parallel: int, n_active: int, k_codo: int) -> int:
    """How many NEW streams this user should open right now."""
    want = min(max(1, parallel), max(1, k_codo))
    used = max(0, n_active)
    if used >= want:
        return 0
    return want - used


def pays(prev_agg: float, new_agg: float, eps: float = 0.05) -> bool:
    """True if a wider k still raises aggregate tok/s by at least eps."""
    if prev_agg <= 0:
        return new_agg > 0
    return new_agg >= prev_agg * (1.0 + eps)


def knee(ks: list[int], aggs: list[float], eps: float = 0.05) -> int:
    """Last k whose step from the previous measurement still paid.

    `ks` increasing (e.g. [1, 4, 8] or [1, 32, 64, 120]). P and D series
    are separate: compute-bound prefill often knees at 1; decode later.
    """
    if not ks:
        return 1
    best = ks[0]
    for i in range(1, min(len(ks), len(aggs))):
        if pays(aggs[i - 1], aggs[i], eps):
            best = ks[i]
        else:
            return best
    return best


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return default


@dataclass
class Occupancy:
    parallel: int = 8
    k_codo_d: int = 4
    k_codo_p: int = 8
    n_active: int = 0

    @classmethod
    def from_env(cls) -> Occupancy:
        parallel = _env_int("RAIDFLASH_PARALLEL", "RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY", default=8)
        return cls(
            parallel=parallel,
            k_codo_d=_env_int("RAIDFLASH_K_CODO_D", default=4),
            k_codo_p=_env_int("RAIDFLASH_K_CODO_P", default=parallel),
        )

    def idle(self) -> int:
        return max(0, self.parallel - self.n_active)

    def d_open(self) -> int:
        return k_target(self.parallel, self.n_active, self.k_codo_d)

    def p_open(self) -> int:
        return k_target(self.parallel, self.n_active, min(self.k_codo_p, self.parallel))

    def filling(self) -> bool:
        return self.idle() == 0

    def set_knees(self, *, k_codo_d: int | None = None, k_codo_p: int | None = None) -> None:
        """Update measured knees. Cap stays `parallel`."""
        if k_codo_d is not None:
            self.k_codo_d = max(1, int(k_codo_d))
        if k_codo_p is not None:
            self.k_codo_p = max(1, int(k_codo_p))

    def assign(self, n_users: int, phase: str = "D") -> list[int]:
        """k per user from the EFI splitter. See raidflash.efi.EfiMap."""
        from .efi import EfiMap, Phase

        m = EfiMap(
            parallel=self.parallel,
            k_codo_d=self.k_codo_d,
            k_codo_p_cold=1,
            k_codo_p_hot=self.k_codo_p,
        )
        ph: Phase = phase if phase in ("D", "P_cold", "P_hot") else "D"
        return m.allocate(n_users, ph)
