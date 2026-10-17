"""EFI map: measured tok/s vs k, then a per-user injector.

k is RPM. P-cold / P-hot / D is load. Aggregate tok/s is power.
The computer: given n users and free slots, pick k per user so the die
stays at the paying knee and nobody sits on C1 while lanes idle.

1 user  → that user gets min(PARALLEL, k_codo).
3 users → even split of the filled die, e.g. 8 lanes → [3, 3, 2].
P-cold  → extra k does not raise agg; one lane per user.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .occupancy import knee, pays

Phase = Literal["D", "P_cold", "P_hot"]


def _lerp(points: dict[int, float], k: int) -> float:
    if not points:
        return 0.0
    k = max(1, int(k))
    if k in points:
        return float(points[k])
    ks = sorted(points)
    if k <= ks[0]:
        return float(points[ks[0]])
    if k >= ks[-1]:
        return float(points[ks[-1]])
    lo = max(x for x in ks if x < k)
    hi = min(x for x in ks if x > k)
    t = (k - lo) / (hi - lo)
    return points[lo] + t * (points[hi] - points[lo])


def split_lanes(n_users: int, parallel: int, k_codo: int) -> list[int]:
    """Fill the die, cap each user at k_codo, split leftover lanes even.

    Returns one k per user (zeros if more users than slots).
    """
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


@dataclass
class EfiMap:
    """Power curve of one engine. Load from a live EFI JSON when present."""

    parallel: int = 8
    d: dict[int, float] = field(default_factory=dict)
    p_cold: dict[int, float] = field(default_factory=dict)
    p_hot: dict[int, float] = field(default_factory=dict)  # speedup vs serial
    k_codo_d: int = 8
    k_codo_p_cold: int = 1
    k_codo_p_hot: int = 8

    @classmethod
    def gb10(cls) -> EfiMap:
        """Fallback cells from GB10 TP2 GLM-5.3-Flash until the live map lands.

        D: PRE4096 fat pool 2026-10-17. P-cold: P-vs-D 2026-10-16.
        P-hot speedup filled by EFI-MAP JSON (seed+sisters).
        """
        d = {1: 40.7, 4: 80.2, 8: 108.5}
        p_cold = {1: 1420.9, 4: 1511.5, 8: 1492.2}
        return cls(
            parallel=8,
            d=d,
            p_cold=p_cold,
            p_hot={},
            k_codo_d=knee(sorted(d), [d[k] for k in sorted(d)]),
            k_codo_p_cold=1,  # GEMM full at k=1; 1→4 is noise on a flat ~1500
            k_codo_p_hot=8,
        )

    @classmethod
    def load(cls, path: str | Path) -> EfiMap:
        raw = json.loads(Path(path).read_text())
        d = {int(c["k"]): float(c["agg_tok_s"]) for c in raw.get("D") or []}
        p_cold = {int(c["k"]): float(c["agg_prefill_tok_s"]) for c in raw.get("P_cold") or []}
        p_hot = {
            int(c["k"]): float(c["speedup_vs_serial_est"])
            for c in raw.get("P_hot") or []
            if c.get("speedup_vs_serial_est") is not None
        }
        efi = raw.get("efi") or {}
        m = cls(
            parallel=int((raw.get("meta") or {}).get("streams_max") or 8),
            d=d,
            p_cold=p_cold,
            p_hot=p_hot,
            k_codo_d=int(efi.get("k_codo_d") or (knee(sorted(d), [d[k] for k in sorted(d)]) if d else 8)),
            k_codo_p_cold=int(efi.get("k_codo_p_cold") or 1),
            k_codo_p_hot=int(efi.get("k_codo_p_hot") or 8),
        )
        return m

    def curve(self, phase: Phase) -> dict[int, float]:
        return {"D": self.d, "P_cold": self.p_cold, "P_hot": self.p_hot}[phase]

    def k_codo(self, phase: Phase) -> int:
        return {"D": self.k_codo_d, "P_cold": self.k_codo_p_cold, "P_hot": self.k_codo_p_hot}[phase]

    def power(self, phase: Phase, k: int) -> float:
        """Die aggregate (tok/s or P-hot speedup) at k lanes of one job."""
        return _lerp(self.curve(phase), k)

    def allocate(self, n_users: int, phase: Phase) -> list[int]:
        """k strips/sisters per user. Fills paying lanes. Splits across users."""
        return split_lanes(n_users, self.parallel, self.k_codo(phase))

    def users_tok_s(self, n_users: int, phase: Phase) -> list[float]:
        """Predicted tok/s (or speedup) per user after allocate()."""
        ks = self.allocate(n_users, phase)
        k_sum = sum(ks)
        if k_sum <= 0:
            return [0.0] * n_users
        die = self.power(phase, k_sum)
        if phase == "P_hot":
            # speedup is per job; each user with k_i sisters gets that cell
            return [self.power(phase, k) if k else 0.0 for k in ks]
        # D / P-cold: die throughput shared by lane count
        return [die * (k / k_sum) if k else 0.0 for k in ks]

    def plan(self, n_users: int, phase: Phase) -> dict:
        ks = self.allocate(n_users, phase)
        pred = self.users_tok_s(n_users, phase)
        return {
            "n_users": n_users,
            "phase": phase,
            "k": ks,
            "tok_s": [round(x, 1) for x in pred],
            "die_tok_s": round(sum(pred), 1),
            "k_codo": self.k_codo(phase),
            "parallel": self.parallel,
            "idle": self.parallel - sum(ks),
        }


def pays_cell(prev: float, new: float, eps: float = 0.05) -> bool:
    return pays(prev, new, eps)
