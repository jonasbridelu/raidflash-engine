# RAIDFLASH

**The whole die, to whoever needs compute most.**

Jonathan Briceño De Luise · Peru, b. 1982 · October 2026 · v1.0 PD

Full paper: [PAPER-PD.md](PAPER-PD.md) · [PDF](PAPER-PD.pdf).

## Abstract

Serving stacks already open N concurrent streams. Published ceilings are
tokens at high concurrency (StorageReview 2×GB300 Flash: 353 tok/s at 1
stream, 2721 at 32). RAIDFLASH assigns those streams to **one job** — or,
when several people are present, splits the filled die toward whoever
needs compute now. Prefill and decode ride the same lane pool as stripes.
k is measured per engine (an EFI map), separately for cold prefill, hot
prefix, and decode. On 2×GB10 TensorFold GLM-5.3-Flash EXL3, live
`usage` tokens: decode 49.5 → 121.1 tok/s at k=8 (2.45×); hot prefill
sisters 99% cache, 5.94× vs serial seeds; cold prefill flat ~1600 tok/s
(compute-bound at k=1). One user gets k=8. Three users on decode get
[3, 3, 2]. The idea came from outside the serving lab: a person looking
at an idle GPU while a human waited.

## 1. Origin

RAIDFLASH is not a new kernel and not a TensorFold fork. It is a policy
for *who* the engine's concurrent paths belong to. Commercial serving
keeps a GPU busy with N tenants so each cursor stays “interactive”
(~50 tok/s) while the die does tens of thousands. A clinic, a lab, a
home box bought the silicon for the person in front of it. The unused
principle: those N paths can be stripes of one document. When more
people show up, the same map splits maximum compute instead of rationing
C1.

## 2. Law

```
k = min(PARALLEL, k_codo_phase)
```

`PARALLEL` is the seat count. `k_codo` is the last k that still raises
aggregate tok/s. Prefill-cold, prefill-hot, and decode have different
knees. Idle seats while a user is below their knee are a miss. Seats
past the knee are a miss.

| people | D / P-hot | P-cold |
|---|---|---|
| 1 | [8] on this GB10 | [1] |
| 3 | [3, 3, 2] | [1, 1, 1] |

## 3. Compute vs memory

Cold prefill fills SMs at k=1; extra cold k does not raise aggregate.
Decode is memory-bound; extra k amortizes expert/KV reload until the
knee. Hot prefill is KV: one drafted seed (`max_tokens ≥ 16`) and sisters
inherit (~99% cached). A serial seed (mt=8) drops the keep (0% cache,
measured).

## 4. This engine (EFI, 2026-10-16, `delta==sum`)

`--parallel` 8, PREFILL 4096 / FILL 2048, pool 1.87M.

| k | D tok/s | P-cold prefill | P-hot vs serial | job packer |
|---|---|---|---|---|
| 1 | 49.5 | 1586 | 1.53× | 15.5 |
| 2 | 71.0 | 1604 | 2.30× | 26.1 |
| 4 | 81.1 | 1651 | 3.57× | 39.9 |
| 8 | **121.1** | 1629 | **5.94×** | **66.8** |

Knees: D=8, P-cold=1, P-hot=8, job=8. A B300 map may want 32/64/120.
Copy the law, not the 8.

Public shape: StorageReview Flash 353 C1 vs 2721 @ 32. Those 32 streams
are RAIDFLASH's k, aimed at one job or split by the injector.

## 5. Product

In-engine `POST /v1/raidflash/complete`: seed + k sisters on `App.run`.
v1.1 seed is a numbered outline; sisters share that spine (`rounds=1` default).
Surgical TensorFold overlay (DepthPolicy `most`, no fc7 clobber). Flag
off = stock engine. Headline: **2.45× decode occupancy** (k=8 vs k=1), not a
5× single cursor. CAKE is the company (custom kernels, AI PCs for
enterprise and business, [cakeinfraestructure.com](https://cakeinfraestructure.com));
RAIDFLASH is the occupancy layer so those machines spend the chip the
buyer paid for.

Falsified here: tree T3, request-level P∥D overlap, `MULTI_WINDOW=64`.
MLA/Hydragen is Dynamo, out of this stack. Neighbour: IPPD
(arXiv 2609.05707).

## 6. Close

v0.1's methods are crude on purpose: a seed, k sisters, an EFI table, one
POST. Archaic. Almost silly. The community will write the sharp kernels.
The disruption is the permission: **run the GPU you already own at 100%.**
People who were about to buy another card can enjoy the one they paid for.
Sovereign AI. Local AI. CAKE — https://cakeinfraestructure.com

Map: [EFI.md](EFI.md). Law: [OCCUPANCY.md](OCCUPANCY.md). Tables: [TABLE.md](TABLE.md).
