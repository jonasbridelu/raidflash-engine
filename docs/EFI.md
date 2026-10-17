# EFI map — injector for RAIDFLASH k

A car EFI table maps RPM × load → fuel and spark. This map maps

```
k (strips / sisters)  ×  phase (P-cold | P-hot | D)  →  tok/s
```

on **this** engine (GB10 TP2, GLM-5.3-Flash EXL3, `--parallel` 8). The
computer then picks k **per user** so the die stays at the paying knee.

## Computer

```
1 user  → k = min(PARALLEL, k_codo_phase)
3 users → split filled lanes, e.g. D on 8 slots → [3, 3, 2]
P-cold  → k_codo = 1 (GEMM already full); one lane per user
P-hot   → k_codo = PARALLEL (sisters inherit that user's KV)
D       → k_codo = last k whose agg tok/s still pays ε=5%
```

Idle slots while any admitted user is below their codo are a miss. Extra
lanes past the codo are a miss. Parking two of three users at zero to give
one user k=8 is a miss **when those users are present**.

```python
from raidflash import EfiMap
m = EfiMap.gb10()                          # fallback cells
# m = EfiMap.load("versiones/EFI-MAP-GB10-2026-10-17.json")
print(m.plan(1, "D"))   # {'k': [8], ...}
print(m.plan(3, "D"))   # {'k': [3, 3, 2], idle: 0}
print(m.plan(3, "P_cold"))  # {'k': [1, 1, 1]}
```

## Fallback cells (until the live sweep JSON lands)

| k | D agg tok/s | P-cold agg prefill tok/s |
|---|---|---|
| 1 | 40.7 | 1421 |
| 4 | 80.2 | 1512 |
| 8 | 108.5 | 1492 |

D still pays at 8 on this die (~2.7× C1). P-cold is flat: extra k does not
raise prefill. P-hot (seed + sisters, 99% cache) is the prefill injector —
cells come from the live EFI sweep (`medir_efi_map.py`).

B300 / 120-wide engines: measure a new map. Copy the computer, not the 8.
