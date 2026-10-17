# RAIDFLASH 0.1 — D/P occupancy

**Total engine concurrency, used dynamically, for maximum tokens/s.**

One user. Every free GPU stream that still pays. Prefill (P) and decode (D)
each have their own knee. Commercial serving parks ~50 tok/s on you and
keeps the rest of the die for N tenants. RAIDFLASH spends the aggregate on
**you**. [CAKE](docs/CAKE.md) ships this so a local GPU does the work it
was paid for.

```
k = min(PARALLEL, k_codo_p)   # prefill
k = min(PARALLEL, k_codo_d)   # decode
```

`PARALLEL` is a **cap**. 8 is the GB10 measured cap/knee on this TensorFold
pair. It is not a law. On a B300, 120 can beat 64. Measure; take the highest
k the system allows that still raises aggregate tok/s.

Johann, 2026. Paper: [PAPER.md](docs/PAPER.md) · [español](docs/PAPER.es.md).
Neighbour: [IPPD](https://arxiv.org/abs/2609.05707). Law:
[OCCUPANCY](docs/OCCUPANCY.md). Injector: [EFI](docs/EFI.md) — k per user
from the power curve (`EfiMap.plan(n_users, phase)`).

| | job | limit |
|---|---|---|
| **P** | seed the document, then sister streams | no acceptance; KV is there or the GEMM runs |
| **D** | dense drafts on those lanes | acceptance bounds **depth of one chain**, not lane count |

## Live smoke — 2026-10-17

Image `tensorfold-glm53:v0.6.1-raidflash`, `streams.max=8`, pool 1 873 920.
Real `usage` tokens. Tree OFF. `PREFILL_ROWS=4096` `FILL_ROWS=2048`.

| run | result |
|---|---|
| In-engine `POST /v1/raidflash/complete` k=8 | wall **7.366 s**, agg **89.1 tok/s**, cached_min **99.1%**, cached_tokens **51 200**, seed_pt 6413, HTTP 200 |
| D k=1 | **34.2 tok/s** (80 completion tokens, 2.342 s, `delta_gen_ok`) |
| D k=8 | **103.5 tok/s** (640 completion tokens, 6.183 s, cached_pct_min 99.5%, `delta_gen_ok`) |
| P-hot (prior) | **99.3%** / 51 200 cached_tokens |
| D homogeneous shorts k=4 (prior) | ~**110 tok/s** |
| D mixed k=8 (prior) | **131.7 tok/s** |
| P-cold k=1..8 (prior) | ~**1500 tok/s** flat (SMs full) |
| `MULTI_WINDOW=64` | worse — keep 32 |

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

Body: `document`, `sections`, optional `stripe_tokens` / `seed_tokens`.
P seed **must** use `max_tokens >= 16` (drafted) so DFlash keep is resumable.

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

Apache-2.0. Johann. See [CONTRIBUTING](CONTRIBUTING.md).
Prod chunks: `PREFILL_ROWS=4096` `FILL_ROWS=2048`.
