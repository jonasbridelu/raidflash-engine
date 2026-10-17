# RAIDFLASH 0.1 — D/P

**One user. Every free GPU stream. Until it stops scaling.**

Commercial serving sells you ~50 tok/s and keeps the rest of the die for N
tenants. RAIDFLASH gives that aggregate to **you**. Researchers and people
with a local GPU get the chip they paid for. [CAKE](docs/CAKE.md) ships this
so products stop crawling at C1.

v0.1 is two tools, one occupancy target:

| | job | limit |
|---|---|---|
| **P** (prefill) | seed the document, then sister streams | no acceptance; KV is there or the GEMM runs |
| **D** (decode) | dense drafts on those lanes | acceptance bounds **depth of one chain**, not lane count |

TensorFold already had `SHARED_PREFIX`, `MULTI_PREFILL`, `DepthPolicy`,
`--parallel`. It never combined them so **one user fills C_max**. That
combination is RAIDFLASH (Johann, 2026). Closest published neighbour:
[IPPD](https://arxiv.org/abs/2609.05707). Next kernel: [Hydragen](https://arxiv.org/abs/2402.05099).

## Install

```bash
pip install -e ".[dev]"
```

Any OpenAI-compatible engine (TensorFold, vLLM, SGLang):

```python
from raidflash import Packer

p = Packer(base_url="http://127.0.0.1:8000")  # RAIDFLASH_PARALLEL=8
out = p.complete(document, ["section A", "section B", "section C", "section D"])
print(out["text"], out["cached_pct_min"], out["agg_tok_s"])
```

P seed **must** use `max_tokens >= 16` (drafted). A serial seed drops the
DFlash keep; sisters miss the cache. Measured.

## Measured (2×GB10, GLM-5.3-Flash EXL3, TensorFold, real `usage` tokens, `delta==sum`)

| | result |
|---|---|
| P-hot seed + 8 sisters | **51 200 cached_tokens**, 99.3%, prefill 0.48 s |
| P-cold unique ~6435 tok | ~1500 tok/s **flat** k=1..8 (SMs full, not DDR5) |
| D k=1 / 4 / 8 | 47 / 79 / 97 tok/s (amortizes 278 GB/s LPDDR5x) |
| `--parallel 8` | P `guard_max=8` one wave; D knee +27% vs k=4 |
| `MULTI_WINDOW=64` | **worse** — do not use |

Public same shape: StorageReview 2×GB300 Flash **353 tok/s C1 vs 2721 @ 32**.

## Engine overlay (TensorFold)

`patches/tensorfold/`:

1. `DepthPolicy.next` high arm uses `self.most` (was hardcoded 3)
2. `multi.py` admit: `adapt_code` / `bind_d` — never clobber `fnc` with `fc7`

Flag: `RAIDFLASH=1` or `/tmp/raidflash.on`. Slot cap: `--parallel 8` (engine max).

## What v0.1 is not

Not a TensorFold fork. Not 10× on one left-to-right cursor. Thousands of tok/s
on two B300s is **k concurrent pieces of one job**. The packer opens those pieces.

## License

Apache-2.0. See [CONTRIBUTING](CONTRIBUTING.md). FILL_ROWS sweep landed:
`PREFILL_ROWS=4096` `FILL_ROWS=2048` (k=1 +5.7% vs 2048; 8192 eats 29% KV).
Next: Hydragen-MLA ([docs/LITERATURE.md](docs/LITERATURE.md)).
