# Occupancy law (v1.1)

v1.1 packer: seed writes a numbered outline, sisters share that spine, optional
`rounds` so wave 2 reads `SO_FAR`. Decode occupancy numbers below are unchanged
(memory-bound). Headline on this box: **2.45×** D at k=8 vs k=1 (121.1 / 49.5).

RAIDFLASH spends unused GPU streams of **one user** on that user's job.
Recipe knobs are caps. CUDA already parallelizes inside kernels.

```
k_target = min(PARALLEL, k_codo) - n_active
```

`PARALLEL` is the engine slot cap. `k_codo` is the highest concurrency that
still pays (the knee). P and D have **separate** knees (`k_codo_p`, `k_codo_d`).

## Dynamic k

Prefill and decode stall until a little concurrency is used. They do not
always want the whole cap.

Raise k while aggregate tok/s still rises past ε. Stop at the knee. The
engine always takes that highest paying k the system allows, separately for
P and D.

8 is the GB10 measured cap/knee here. B300 / GB300 may want 32, 64, or 120.
Measure. Copy the law, not the 8.

P-cold is compute-bound: SMs fill at k=1. Extra P-cold lanes leave agg tok/s
flat. P-hot is KV: seed once, sisters inherit.

D is memory-bound: extra lanes amortize DDR until the knee.

## GB10 — live smoke 2026-10-17

Image `tensorfold-glm53:v0.6.1-raidflash`, `streams.max=8`, pool 1 873 920.
Tree OFF.

| | k | result |
|---|---|---|
| In-engine `POST /v1/raidflash/complete` | 8 | wall 7.366 s, agg 89.1 tok/s, cached_min 99.1%, 51 200 cached_tokens, seed_pt 6413, HTTP 200 |
| D shorts | 1 | 34.2 tok/s, 80 completion tokens, 2.342 s, `delta_gen_ok` |
| D shorts | 8 | 103.5 tok/s, 640 completion tokens, 6.183 s, cached_pct_min 99.5%, `delta_gen_ok` |

Prior knees, same stack (real `usage`, `delta==sum`):

- P-hot 99.3% / 51 200 cached_tokens
- D homogeneous shorts k=4 ~110 tok/s
- D mixed k=8 131.7 tok/s
- P-cold ~1500 tok/s **flat** k=1..8 (SMs full)
- `MULTI_WINDOW=64` worse (all-gather)

Public same shape: StorageReview 2×GB300 Flash 353 tok/s C1 vs 2721 @ 32
streams. Those 32 streams are RAIDFLASH's k, assigned to one job.

## Seed

P seed must be drafted (`max_tokens >= 16`). Serial seed → `drafter_end=-1`
→ `_resume` drops the keep → 0% cached (measured).

## Chunk sizes

Lone P-cold uses `TF_GLM_PREFILL_ROWS`. Busy fill uses `TF_GLM_FILL_ROWS`
(capped at prefill_rows). Prod: **PREFILL 4096 / FILL 2048**.
