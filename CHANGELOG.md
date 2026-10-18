# Changelog

## 0.2.2 — 2026-10-17 — CAKE is the company

- CAKE: custom kernels and AI PCs for enterprise and business.
- RAIDFLASH is the occupancy layer on that stack (`docs/CAKE.md`).

## 0.2.1 — 2026-10-17 — English-only public docs

- Papers, README, TABLE, IMPLEMENT, changelog, and packer/sister prompts are English.
- Removed `docs/PAPER.es.md`, `docs/PAPER.es.pdf`, `docs/PAPER-PD.es.md`, `docs/PAPER-PD.es.pdf`.

## 0.2.0 — 2026-10-17 — RAIDFLASH PD paper + Engine 1.0.4 numbers

- Author: **Jonathan Briceño De Luise** (Peru, b. 1982).
- Paper (English): `docs/PAPER-PD.md`, `docs/PAPER-PD.pdf`.
- Live packer 8k stripe 256 on `raidflash-glm53:1.0.4`: **94.6 tok/s**,
  2064 completion (16+8×256), cache 98.7%, `delta_gen=2064`.
- Sister tokens are generated length, not DFlash `accepted`.
- `choose_k` / EFI injector in-tree. Overlay still Apache-2.0, no vendored TensorFold.

## 0.1.2 — occupancy wiring + reasoning EFI

- `choose_k(thinking=True, max_tokens<=80)` → **k=4** (thinking peak 115 tok/s).
  Long (256+) stays k=8.
- `most_for_lanes(n)`: draft depth = min(7, WINDOW/n). k=4 keeps most=7;
  k=8 gets 4 (fits MULTI_WINDOW=32). Drops the busy hammer most 7→3.
- `Packer.think(question)`: seed + 4 thought-sisters (P-hot, same prefix).


- Chat `/v1/chat/completions` packs on `raidflash.document` (sections optional).
- Packer k = `min(EFI, idle)` using `engine.multi.lanes` (GlmEngine has no `.lanes`).
- `RAIDFLASH_AUTO=1` packs long non-stream chats (`max_tokens>=256`, prompt ≥512 chars).
- Image `tensorfold-glm53:v0.6.5-raidflash`.

## 0.1.1 — 2026-10-17 — 8-seat stable

The shareable recipe. `--parallel 8`, fat pool, RAIDFLASH-D/P, in-engine EFI.

- In-engine packer EFI: `n_users` / `user_index` / `phase` (`1 → [8]`, `3 → [3,3,2]`).
- Expression grid 60/60 clean: k × 10…5000 × prose/code/thinking (`docs/TABLE.md`).
- Knobs that lost and stay off: `MULTI_DEPTH=joint`, `MULTI_ASYNC`+`packed`, tree, WINDOW=64.
- k=16 fork measured then rolled back: D k=16 110 vs 8-seat D shorts 121; packer 16 sisters 117. **0.1 ships 8 seats.**
- TensorFold Zig 1.0.x (incl. 1.0.1) is a different engine: GLM CUDA EXL3 occupancy stays on Mia Python 0.6.0 + this overlay. Zig `zig-engine` is WIP (shared rounds, missing Triton variants).

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
