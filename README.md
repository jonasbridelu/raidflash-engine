# RAIDFLASH 1.0 — PD occupancy

**Total engine concurrency, used dynamically, for maximum tokens/s.**

One user. Every free GPU stream that still pays. Prefill (P) and decode (D)
each have their own knee. Commercial serving parks ~50 tok/s on you and
keeps the rest of the die for N tenants. RAIDFLASH spends the aggregate on
**you**. [CAKE](https://cakeinfraestructure.com) is the company: custom
kernels and AI PCs for enterprise and business. RAIDFLASH is the occupancy
layer on that stack. [CAKE.md](docs/CAKE.md).

```
k = min(PARALLEL, k_codo_p)   # prefill
k = min(PARALLEL, k_codo_d)   # decode
```

`PARALLEL` is a **cap**. **8 seats is the GB10 0.1 recipe** (D peak, fat KV
pool). k=16 was measured: packer pays, D peak does not (pool tax). Copy the
occupancy law, not a bigger N.

Jonathan Briceño De Luise (Peru, b. 1982), 2026.
Paper PD: [PAPER-PD.md](docs/PAPER-PD.md) · [PDF](docs/PAPER-PD.pdf).
Program: [PROGRAM.md](docs/PROGRAM.md). Neighbour: [IPPD](https://arxiv.org/abs/2609.05707).
Law: [OCCUPANCY](docs/OCCUPANCY.md). Injector: [EFI](docs/EFI.md).
Complete measured table (k × tokens × prose/code/reasoning): [TABLE](docs/TABLE.md).

| | job | limit |
|---|---|---|
| **P** | seed the document, then sister streams | no acceptance; KV is there or the GEMM runs |
| **D** | dense drafts on those lanes | acceptance bounds **depth of one chain**, not lane count |

## Live smoke — 8-seat recipe (Engine 1.0.4)

Image `raidflash-glm53:1.0.4`, `streams.max=8`, tree OFF, PREFIX=0,
`PREFILL_ROWS=4096` `FILL_ROWS=2048` `CACHE_GIB=12.5`. Real `usage`, delta==sum.

| run | result |
|---|---|
| EFI D shorts k=8 | **121.1 tok/s** (2.45× C1 49.5) |
| D k=8 mt=256 | **121.1 tok/s**, 2048 tok, 16.9 s |
| 1024 prose 1 chain vs 8×128 | 30.2 s / 33.9 tok/s → **12.8 s / 80.3 tok/s** |
| P-hot packer 8k stripe 256 | **94.6 tok/s**, 2064 tok, cache **98.7%**, 21.8 s |
| P-hot EFI k=8 vs serial | **5.94×**, cache 99% |
| P-cold | ~**1600 tok/s** at k=1; 8 docs 8k → **8806 tok/s** |
| 3 users | `[3, 3, 2]` |

Full grid: [TABLE](docs/TABLE.md). `joint` / `async+packed` / tree / WINDOW=64 stay off.

## Install

```bash
pip install -e ".[dev]"
```

HTTP packer (fallback, any OpenAI-compatible engine):

```python
from raidflash import Packer

p = Packer(base_url="http://127.0.0.1:8000")  # RAIDFLASH_PARALLEL is the cap
out = p.complete(document, ["section A", "section B", "section C", "section D"])
print(out["text"], out["cached_pct_min"], out["agg_tok_s"])
```

In-engine (TensorFold overlay): one POST, seed then k sisters on the CUDA
thread pool.

```
POST /v1/raidflash/complete
```

Body: `document`, `sections`, optional `n_users` / `user_index` / `phase` /
`stripe_tokens` / `seed_tokens`.
P seed **must** use `max_tokens >= 16` (drafted) so DFlash keep is resumable.
One user fills `min(PARALLEL, k_codo)`. Three users on this box: `[3, 3, 2]`.

## Engine overlay (TensorFold)

`patches/tensorfold/` — surgical. No fork. No vendored engine.

1. `DepthPolicy.next` high arm uses `self.most` (was hardcoded 3)
2. `multi.py` admit: `adapt_code` / `bind_d` — never clobber `fnc` with `fc7`
3. `POST /v1/raidflash/complete` via `raidflash_pack.py`

Flag: `RAIDFLASH=1` or `/tmp/raidflash.on`. Slot cap: `--parallel`
(engine `PARALLEL_MOST` here is 8).

## What v0.1 is not

- Not a TensorFold fork
- Not Hydragen / MLA (Dynamo, out of this stack)
- Not tree T3
- Not a fixed 8 streams

## License

Apache-2.0. Jonathan Briceño De Luise. See [CONTRIBUTING](CONTRIBUTING.md).
Prod chunks: `PREFILL_ROWS=4096` `FILL_ROWS=2048`.
