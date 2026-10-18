# RAIDFLASH 1.1

**One job fills TensorFold’s idle seats — 2.45× decode on 2×GB10 GLM-5.3-Flash.**

Occupancy layer: k concurrent streams of **one packable job** until the knee.
Prefill (P) and decode (D) have separate knees. A short chat / tool loop stays
one seat (~50 tok/s). That 2.45× is k=8 vs k=1 on the **same** drafted engine
(121.1 vs 49.5 tok/s), not a 5× single cursor.

[CAKE](https://cakeinfraestructure.com) is the company: custom kernels and AI
PCs for enterprise and business. RAIDFLASH is the occupancy layer on that
stack. [CAKE.md](docs/CAKE.md). Overlay is Apache-2.0 — use it, patch it.

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

## Live smoke — 8-seat recipe (Engine 1.1.2)

Image `raidflash-glm53:1.1.2`, `streams.max=8`, tree OFF, PREFIX=0,
`PREFILL_ROWS=4096` `FILL_ROWS=2048` `CACHE_GIB=12.5`. Real `usage`, delta==sum.
Occupancy tok/s are the 1.0.4 EFI grid (same seats, same weights). 1.1 adds a
shared outline seed; decode of the k stripes stays memory-bound.

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
`stripe_tokens` / `seed_tokens` / `outline` (default true) / `rounds`
(default 1) / `polish` (default false).
P seed **must** use `max_tokens >= 16` (drafted) so DFlash keep is resumable.
Outline seed uses `≥ 64`. One user fills `min(PARALLEL, k_codo)`. Three users
on this box: `[3, 3, 2]`.

v1.1 default: seed writes the section headings, sisters write those sections,
one concatenated completion. `rounds=2` lets wave 2 read wave 1. Chat/agent
with tools stays 1 lane.

KV-continue fork (1.1.2): `raidflash.fork=true`, optional `leader_tokens`.
Leader ids stay on the prompt; k lanes continue that chain. Occupancy of k
cursors, not a 5× single chat.

```python
out = Packer(base_url).fork(document, k=6, leader_tokens=48, stripe_tokens=100)
print(out["kv_continue"], out["share_len"], len(out["lanes"]))
```

## Engine overlay (TensorFold)

`patches/tensorfold/` — surgical. No fork. No vendored engine.

1. `DepthPolicy.next` high arm uses `self.most` (was hardcoded 3)
2. `multi.py` admit: `adapt_code` / `bind_d` — never clobber `fnc` with `fc7`
3. `POST /v1/raidflash/complete` via `raidflash_pack.py`

Flag: `RAIDFLASH=1` or `/tmp/raidflash.on`. Slot cap: `--parallel`
(engine `PARALLEL_MOST` here is 8).

## What v1.1 is not

- Not a TensorFold fork
- Not Hydragen / MLA (Dynamo, out of this stack)
- Not tree T3
- Not a fixed 8 streams
- Not a 5× single chain / agent loop. Say **2.45× decode occupancy** when one
  job fills 8 seats. P-hot vs serial seeds of the same document is **5.94×**.

## License

Apache-2.0. Jonathan Briceño De Luise. See [CONTRIBUTING](CONTRIBUTING.md).
Prod chunks: `PREFILL_ROWS=4096` `FILL_ROWS=2048`.
