# Changelog

## 0.1.0 — 2026-10-16

First public cut.

- Occupancy: `k_target = min(PARALLEL, k_codo) - n_active` for **one user**
- RAIDFLASH-D: `adapt_code` / `bind_d` / `depth_next` (most=7, shorts serial)
- RAIDFLASH-P: drafted seed (`max_tokens>=16`) + sister prompts + merge
- Packer: one document → seed + k sister HTTP streams (`--parallel` slots)
- TensorFold overlay notes (no vendored engine)
- Apache-2.0
