# RAIDFLASH 0.1 — D/P

**One user. Every free GPU stream. Until it stops scaling.**

Commercial serving sells you 50 tok/s and keeps the rest of the die for N
tenants. RAIDFLASH gives that aggregate to **you**. Researchers and people
running a local GPU get the chip they paid for. [CAKE](docs/CAKE.md) ships
this so medical and everyday products stop crawling at C1.

v0.1 is two tools on one occupancy target:

| | job | limit |
|---|---|---|
| **D** (decode) | dense drafts on your lanes | acceptance bounds **depth of one chain**, not lane count |
| **P** (prefill) | seed the document, then sister streams | no acceptance; KV is there or the GEMM runs |

TensorFold already had `SHARED_PREFIX`, `MULTI_PREFILL`, `DepthPolicy`,
`--parallel`. It never combined them for **one user filling C_max**. That
combination is RAIDFLASH (Johann, 2026). Closest published neighbour: IPPD.

Measured on dual GB10 (GLM-5.3-Flash EXL3, TensorFold, clean `delta==sum`):

- D turbo unfixed: kind 16 kept, `DepthPolicy.next` 1/4/7, shorts serial
- P seed + 8 sisters: **98.7% cached**, second wave still inherits
- `--parallel 4` is the slot cap: k=8 queued, GPU stayed 4-wide
- Public same shape: StorageReview 2×GB300 Flash **353 tok/s C1 vs 2721 @ 32**

## Install

```bash
pip install -e ".[dev]"
```

Talk to any OpenAI-compatible engine (TensorFold, vLLM, SGLang):

```python
from raidflash import Packer

p = Packer(base_url="http://127.0.0.1:8000")
out = p.complete(document, ["section A", "section B", "section C", "section D"])
print(out["text"], out["cached_pct_min"], out["agg_tok_s"])
```

P seed **must** use `max_tokens >= 16` (drafted). A serial seed drops the
DFlash keep and sisters miss the cache. That is v0.1 law, measured.

## Engine overlay (TensorFold)

Patches in `patches/tensorfold/`:

1. `DepthPolicy.next` high arm uses `self.most` (was hardcoded 3)
2. `multi.py` admit: `adapt_code` / `bind_d` — never clobber `fnc` with `fc7`

Flag: `/tmp/raidflash.on` or `RAIDFLASH=1`.

## What v0.1 is not

Not a TensorFold fork. Not a 10× single left-to-right cursor. 5000 tok/s on
two B300s is **k concurrent pieces of one job**. The packer opens those pieces.

## License

Apache-2.0. TensorFold is Apache-2.0; this repo is the occupancy layer.
Paper (unpublished beta): Obsidian `PAPER-BETA-RAIDFLASH.md`.
