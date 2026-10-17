# Literature for P (TTFT) and D (tok/s) — 2026-10-17

Work class: 1 user, shared prefix, MoE GLM-5.3-Flash, 2×GB10 ~278 GB/s
LPDDR5x, TensorFold `SHARED_PREFIX`, DFlash2. Occupancy is D/P. k is
dynamic; 8 is this box's measured cap.

## Measured (do not reopen)

| idea | paper / system | result |
|---|---|---|
| overlap P∥D at request level | DistServe / SARATHI mix | M5: decode −59% |
| speculative tree T3 | ASPD / SoT | engine healthy, 0 gain |
| `MULTI_WINDOW=64` | TF docs | worse P and D (all-gather) |
| cold k simultaneous | classic batching | M6 0% cache; P-cold agg **flat ~1500 tok/s** |
| seed + sisters | radix / APC | **99.3%**, 51 200 cached_tokens |

## Neighbours

### IPPD — arXiv 2609.05707

Intra-prompt decode of questions with shared context. Closest published
neighbour to RAIDFLASH. Sidecar `:8010`; keep off the live engine.

### SARATHI chunked prefill — measured 2026-10-17

k=1 uses `TF_GLM_PREFILL_ROWS`; k=4 busy uses `TF_GLM_FILL_ROWS`. Real
tokens, `delta==sum`.

| FILL=PREFILL | pool | k=1 engine tok/s | k=4 engine | k=4 agg wall |
|---|---|---|---|---|
| 2048 | 2.03M | 1541 | 1116 | 1480 |
| 4096 | 1.88M | 1629 | 915 | 1541 |
| 8192 | 1.44M | 1664 | 769 | 1601 |

Lone P-cold +5.7% at 4096, +8% at 8192, diminishing. 8192 eats 29% KV pool.
Equal large FILL tanks per-stream k=4. **Prod: PREFILL_ROWS=4096 FILL_ROWS=2048.**

### Hydragen — arXiv 2402.05099 — Dynamo, out of this stack

After P-hot, decode of k sisters re-reads prefix KV k times. Hydragen
batches Q over the prefix (GEMM). TensorFold GLM uses MLA windows
(`MLA_DECODE_ROWS`). Porting Hydragen to MLA is Dynamo work, outside
RAIDFLASH v0.1.

### MoE-Prefill — arXiv 2605.02960 + DuoServe-MoE 2509.07379

Prefill activates almost every expert; decode is top-k. Expert prefetch
overlapped with GEMM. TensorFold already caches experts.

### Splitwise / DistServe PD disagg

StorageReview 2×GB300: Flash 353 C1 → 2721 @32 with P on one tower and D
on the other. This stack is TP2 occupancy: one job filling streams up to
the dynamic knee.

v0.1 stays on D/P occupancy: seed `mt>=16`, sisters, `k_target`, FILL 2048 /
PREFILL 4096.
