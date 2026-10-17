# Behavior table — this TensorFold engine (2×GB10, GLM-5.3-Flash EXL3)

Measured on `tensorfold-glm53:v0.6.2-raidflash`, `--parallel` 8, RAIDFLASH-D ON, tree OFF.
Tokens are engine `usage` + `tensorfold:*_total` with **delta==sum**. Nothing is chars/4.

The method is crude on purpose: open k streams of **one job** until aggregate tok/s stops paying.
That is 100% of the GPU you already own.

## 1. How to use it

```
POST /v1/raidflash/complete
```

Body: `document`, `sections`, optional `n_users`, `stripe_tokens`, `seed_tokens` (≥16).
One user → k fills the die. Three users → `[3, 3, 2]`. P-cold → one lane each.

Chat `/v1/chat/completions` is unchanged. `raidflash.sections` in the body also opens k lanes.

Overlay: drop `patches/tensorfold/raidflash_pack.py` + `raidflash_policy.py`, three hooks
(`decode.py` most-arm, `multi.py` adapt/bind, `http.py` `_raidflash`). Flag `RAIDFLASH=1`.

## 2. EFI map — k × phase (live 2026-10-16, confirmed)

| k | D tok/s | vs C1 | P-cold prefill | P-hot vs serial | job packer tok/s | cache |
|---|---|---|---|---|---|---|
| 1 | 49.5 | 1.00× | 1586 | 1.53× | 15.5 | seed 0% |
| 2 | 71.0 | 1.43× | 1604 | 2.30× | 26.1 | ≥98.8% |
| 4 | 81.1 | 1.64× | 1651 | 3.57× | 39.9 | ≥98.8% |
| 8 | **121.1** | **2.45×** | 1629 | **5.94×** | **66.8** | ≥98.8% |

Knees: **D=8**, **P-cold=1**, **P-hot=8**, **job=8**.

P-cold GEMM is already full at k=1. Extra cold lanes do not raise aggregate.
P-hot needs a drafted seed (`max_tokens ≥ 16`); mt=8 seed → 0% sister cache.
D still pays at k=8 (the engine cap). That 8 is a seat count, not a law.

Injector:

| people | D / P-hot | P-cold |
|---|---|---|
| 1 | `[8]` | `[1]` |
| 3 | `[3, 3, 2]` | `[1, 1, 1]` |

## 3. Same-day occupancy (v0.6.2, recipe that pays)

| | D k=1 | D k=4 | D k=8 | packer k=8 |
|---|---|---|---|---|
| stock concurrent knobs | 42.4 | 81.6 | **124.9** | 68.4 tok/s, 98.8% cache |
| `MULTI_DEPTH=joint` + async + packed | 40.3 | 88.5 | 111.7 | 55.9 |
| async + packed, `depth=policy` | 49.3 | 80.4 | 113.8 | 59.9 |

`joint` splits the 32-row window across 8 streams and fights RAIDFLASH-D `most=7`.
async+packed also lost at the filled die. **Reverted.** Production stays default sampler/depth, `MULTI_LONE=0`.

## 4. Full expression grid (10 → 5000 tokens)

Measured 2026-10-16, 60/60 cells `limpio`. JSON `versiones/EXPRESION-GB10.json`.

**Prose tok/s**

| k | 10 | 16 | 80 | 256 | 1024 | 5000 |
|---|---|---|---|---|---|---|
| 1 | 12.5 | 17.0 | 31.7 | 31.7 | 33.6 | 36.8 |
| 4 | 36.7 | 34.4 | 46.2 | 61.6 | 63.5 | 64.0 |
| 8 | 45.4 | 59.5 | 75.5 | 82.0 | **84.0** | 81.7 |

**Code tok/s**

| k | 10 | 16 | 80 | 256 | 1024 | 5000 |
|---|---|---|---|---|---|---|
| 1 | 16.3 | 22.5 | 44.2 | 49.8 | 57.1 | 57.1 |
| 4 | 50.6 | 44.9 | 66.4 | 73.8 | 87.7 | 82.1 |
| 8 | 74.6 | 82.3 | 88.5 | 95.6 | **109.3** | 106.6 |

**Reasoning tok/s** (ct ≈ rt)

| k | 10 | 16 | 80 | 256 | 1024 | 5000 |
|---|---|---|---|---|---|---|
| 1 | 13.6 | 21.4 | 63.6 | 44.8 | 42.2 | 44.1 |
| 4 | 40.2 | 53.1 | **115.4** | 78.5 | 76.3 | 73.5 |
| 8 | 52.3 | 68.6 | 110.7 | **115.1** | 100.6 | 96.1 |

Same-budget prose: 1024 tokens as 1 chain **33.9 tok/s / 30.2 s** vs 8×128 **80.3 / 12.8 s**.
5000 as 1 chain EOS at 1590 tok; 8×625 delivered **5000 tok at 88.5 tok/s**.

## 5. What paid / what lost (honest)

Paid:

- `--parallel` 8 (was recipe-capped at 4)
- RAIDFLASH-D `most=7`, unfix, never `fc7`
- RAIDFLASH-P drafted seed mt≥16 + sisters + `SHARED_PREFIX`
- Fat KV pool `CACHE_GIB=12.5`, `FILL_ROWS=2048`, `PREFILL_ROWS=4096`
- `MULTI_WINDOW=32`, in-engine packer (one POST)
- EFI split in the packer (`n_users`)

Lost / off:

- tree T3
- `MULTI_WINDOW=64`
- `FILL_ROWS=8192`
- thin pool
- `MULTI_DEPTH=joint`
- `MULTI_ASYNC` + `packed` at k=8
- Hydragen / MLA (Dynamo, other stack)
- Zig TensorFold 1.0.12 (not this EXL3 TP2)

REA was not used. PARALLEL_MOST=8 is still the engine seat cap; D still paid at 8, so a measured k>8 fork is the next occupancy lever on this box.

## 6. GitHub

Apache-2.0 occupancy layer. Not a TensorFold vendor. Overlay is two files and three hooks.
`gh auth login` is still required to push `raidflash-engine`.
