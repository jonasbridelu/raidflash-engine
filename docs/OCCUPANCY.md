# Occupancy law (v0.1)

Recipe knobs are caps. RAIDFLASH fills unused streams of **one user**.

`k_target = min(PARALLEL, k_codo) - n_active`

Measured knees (GB10 TensorFold GLM-5.3-Flash):

- D shorts: k=4
- D mixed Hermes: k=8
- P cache: 8 sisters inherit; compute still `--parallel`

P seed must be drafted (`max_tokens >= 16`). Serial seed → `drafter_end=-1`
→ `_resume` drops the keep → 0% cached (measured).

Lone P-cold uses `TF_GLM_PREFILL_ROWS`. Busy fill uses `TF_GLM_FILL_ROWS`
(capped at prefill_rows). Prod: 4096 / 2048. Do not raise them together to 8192.

StorageReview 2×GB300: Flash 353 C1 vs 2721 at 32 streams (PD split).
Those 32 streams are RAIDFLASH's k, assigned to one job.
