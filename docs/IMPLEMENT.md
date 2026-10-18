# Implement this — GB10 8-seat, measured 2026-10-17

Numbers are engine `usage` + `delta==sum`. Image `v0.6.2-raidflash`, `--parallel` 8.
JSON: [data/gb10-8seat.json](data/gb10-8seat.json). Code: `raidflash.choose.choose_k`.

## Rules (copy into the product)

| situation | k | measured |
|---|---|---|
| One document, first prefill (seed) | **1** | one prefix is sequential |
| Same document, sisters after seed ≥16 | **8** | P-hot 8k Engine 1.0.4: **94.6 tok/s**, 2064 tok, cache **98.7%**; stress 1.0.1 91.8 / 98.9%; 32k: **61 tok/s**, cache **99.7%** |
| Many independent cold prompts | **8** | prefill 8k: **8806 tok/s** (k=1 was 2317); 32k: **15744 tok/s**, 225k prompt tok |
| Decode, short prefix | **8** | fill-80 **86.5**; 1024 **88.2**; 5000 **73.4** (21k tok, 5 min) |
| Short thinking mt≤80 | **4** | peak **115.4** tok/s; k=8 cuts `most` (WINDOW/n) |
| Long thinking 256+ | **8** | occupancy; 115 tok/s @256 |
| Decode after 8k KV | **8** | **59.3 tok/s** (still > C1) |
| 3 users D / P-hot / many | **[3,3,2]** | fill the die |
| 3 users one-doc seed | **[1,1,1]** | extra cold k on one prefix does not split the sequential prefill |

```python
from raidflash.choose import choose_k, seed_then_sisters
from raidflash.packer import Packer

choose_k(kind="decode")            # 8
choose_k(kind="decode", thinking=True, max_tokens=80)    # 4
choose_k(kind="decode", thinking=True, max_tokens=256)   # 8
choose_k(kind="prefill_many")      # 8
choose_k(kind="prefill_one")       # 1
seed_then_sisters(n_sections=8)    # seed_k=1, sister_k=8, seed_tokens=16
Packer(base_url).think(question)   # seed + 4 thought-sisters
```

## Do not

- Default k=16 (pool tax: D peak 125 shorts → 110).
- `MULTI_DEPTH=joint`, `MULTI_ASYNC`+`packed`, tree, `MULTI_WINDOW=64`.
- Seed `max_tokens < 16` (0% sister cache).
- Zig TensorFold 1.0.x as drop-in (no occupancy overlay yet).

## What the world gets

One user: `POST /v1/raidflash/complete` → seed + 8 sisters. Chat with `raidflash.document` (no sections) packs the same. `RAIDFLASH_AUTO=1` packs long non-stream chats. Packer k = min(EFI, idle `engine.multi.lanes`). Prefill of N independent jobs: 8 lanes. Decode: 8 lanes until the die is full.
