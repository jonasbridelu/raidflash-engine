# Changelog

## 0.1.0 — 2026-10-17

First public cut. Occupancy layer for **one user**. k is dynamic.

- Law: `k_target = min(PARALLEL, k_codo) - n_active`. PARALLEL is a cap.
  P knee and D knee are separate. 8 is the GB10 measured cap here; a B300
  may want 32 / 64 / 120 — `Occupancy.set_knees` / `knee()` measure it.
- RAIDFLASH-D: `adapt_code` / `bind_d` / `depth_next` (most=7, shorts serial,
  never clobber fnc7)
- RAIDFLASH-P: drafted seed (`max_tokens>=16`) + sister prefix cache
- Packer: `POST /v1/raidflash/complete` in-engine (App.run); HTTP fan-out on 404
- TensorFold surgical overlay in `patches/tensorfold/` (no vendored engine)
- Apache-2.0

Live smoke on `tensorfold-glm53:v0.6.1-raidflash` (2×GB10 TP2, real `usage`):

| | result |
|---|---|
| in-engine packer k=8 | wall 7.37 s, **99.1% cache**, 51200 cached_tokens, agg 89.1 tok/s |
| D k=1 | 34.2 tok/s |
| D k=8 | **103.5 tok/s** (3.0× C1), delta==sum |

Prod knobs: `PREFILL_ROWS=4096` `FILL_ROWS=2048` `CACHE_GIB=12.5` `--parallel 8`.
`MULTI_WINDOW=64` and tree T3 stay off.

## Unreleased notes from the GB10 lab

- FILL_ROWS sweep: 8192 +8% k=1 P-cold, −29% KV pool. Keep 4096/2048.
