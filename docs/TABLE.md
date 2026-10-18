# Behavior table — this TensorFold engine (2×GB10, GLM-5.3-Flash EXL3)

Measured on TensorFold GLM-5.3-Flash EXL3 2×GB10, `--parallel` 8, RAIDFLASH-D ON, tree OFF.
Engine 1.0.4 packer 8k: **94.6 tok/s**, 2064 tok, cache 98.7% (`raidflash-glm53:1.0.4`).
Tokens are engine `usage` + `tensorfold:*_total` with **delta==sum**. Nothing is chars/4.

The method is crude on purpose: open k streams of **one job** until aggregate tok/s stops paying.
That is 100% of the GPU you already own.

Operator cheat-sheet on this box: fill P/D with k seats. Do not retune cuBLAS/kernels.

## 0. Production validation — 2026-10-17 (single-stream chat path, tokens true)

End-to-end validation of the shipping engine (`tensorfold-glm53:v0.6.0-raidflash`,
hook + reasoning turbo `fnc7:0.3`, tree OFF), **real tokens from streaming `usage`**,
2 identical reps per test, contamination guard (poller on `requests_running`).
Baseline serial = 27.7 tok/s (same box, drafting off, measured 2026-10-16).

| test | wall | TTFT | tok/s | vs serial | seconds saved | drafts→accepted |
|---|---|---|---|---|---|---|
| ping 16 | 0.34s | 0.14s | 47.1 | **1.70×** | 0.2s | 12→11 (92%) |
| tool JSON 48 | 0.84s | 0.18s | 57.1 | **2.06×** | 0.9s | 43→36 (84%) |
| prose 300 | 6.43s | 0.17s | 46.7 | **1.68×** | 4.4s | 192→144 (75%) |
| reasoning 300 | 5.07s | 0.20s | 59.2 | **2.14×** | 5.8s | 231→199 (86%) |
| reasoning 1000 | 14.07s | 0.23s | 71.1 | **2.57×** | 22.0s | 844→751 (89%) |
| reasoning 3000 | 56.12s | 0.19s | 53.5 | **1.93×** | 52.2s | 2301→1807 (79%) |
| code 512 | 8.89s | 0.20s | 57.6 | **2.08×** | 9.6s | 407→340 (84%) |
| prefill 14k-prompt | 3.83s | **0.13s** | 52.2 | 1.89× | 3.4s | 145→121 (83%) |

Read:

- **Gain scales with volume** — 1.70× (ping) → 2.57× (reasoning 1000): drafting amortizes
  with length, exactly the position-decay law in [PAPER-PD](PAPER-PD.md). R3000 dips to
  1.93× because the think-block eats part of the budget.
- **Prefill reuse (RAIDFLASH-P) is live**: first cold 14k-prompt request paid TTFT 1.01s,
  the second **0.13s → 7.8× less prefill** from shared-prefix KV.
- **TTFT flat 0.13–0.23s** across the board: no queues, instant perceived response.
- **~101 s of wait saved** across this 8-test battery vs serial.
- Honest caveats: the 27.7 tok/s serial is the 2026-10-16 measured baseline (no serial
  A/B this session — would have required dropping the hook on a live engine); a third
  client injected 131k-prompt traffic between two reps (guard caught it, clean reps used).

## 1. How to use it

```
POST /v1/raidflash/complete
```

Body: `document`, `sections`, optional `n_users`, `stripe_tokens`, `seed_tokens` (≥16).
One user → k fills the die. Three users → `[3, 3, 2]`. P-cold → one lane each.

Chat `/v1/chat/completions` packs when `raidflash.document` or `raidflash.sections` is set (sections optional). `RAIDFLASH_AUTO=1` packs long non-stream chats (`max_tokens>=256`, prompt ≥512 chars). Packer k is `min(EFI, idle slots on engine.multi.lanes)`.

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

Measured 2026-10-16, 60/60 cells clean. JSON `versiones/EXPRESION-GB10.json`.

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

Reasoning 0.1.2: short think k=4 (keeps `most=7`); long think k=8 (occupancy, `most=min(7,32/n)`). Packer.think = seed + 4 thought-sisters.

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
Public: https://github.com/jonasbridelu/raidflash-engine
