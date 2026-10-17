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

## Live cells — 2026-10-16 22:08 (`delta==sum`)

| k | D agg tok/s | P-cold prefill tok/s | P-hot vs serial | job packer tok/s |
|---|---|---|---|---|
| 1 | 49.5 | 1586 | 1.53× | 15.5 |
| 2 | 71.0 | 1604 | 2.30× | 26.1 |
| 4 | 81.1 | 1651 | 3.57× | 39.9 |
| 8 | **121.1** | 1629 | **5.94×** | **66.8** |

k_codo: D=8, P-cold=1, P-hot=8, job=8. P-hot cache ≥99%. D 8 is **2.45×** C1.

JSON: `/home/admin/raidflash/versiones/EFI-MAP-GB10-2026-10-17.json`.
B300 / 120-wide engines: measure a new map. Copy the computer, not the 8.
