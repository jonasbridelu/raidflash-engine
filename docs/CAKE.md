# CAKE × RAIDFLASH

CAKE is the product that has to answer people, not tenants. A clinic, a
researcher, a student with one box: they buy a GPU and the stack gives them
C1 (one stream) while the same silicon does 4–20× in the vendor's multi-user
bench.

RAIDFLASH is the inference layer so CAKE can spend **the whole die** on the
person in front of it:

- P cuts the wait on long records (seed + sisters, prefix cache)
- D densifies decode on those lanes until the knee
- Occupancy target: `requests_running → min(PARALLEL, k_codo)` for that user

Open-sourcing 0.1 is how researchers and ordinary users help make CAKE (and
every local LLM product) honest about tok/s.
