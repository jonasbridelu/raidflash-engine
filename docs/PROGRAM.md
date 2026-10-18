# Program — custom parallel engine before market

The methods in v0.1 are crude. That is the point. The market already has
N-stream serving. RAIDFLASH is the occupancy computer on top: **100% of the
die to one person, or the full die split toward whoever needs compute now.**

Ignorance is running a $120k B300 at ~350 tok/s C1 while the same box does
thousands at 32–128 streams. Same bitrate (HBM, SMs). More resolution on
the person's job. Local AI. Sovereign AI. CAKE
(https://cakeinfraestructure.com — custom kernels, AI PCs for enterprise
and business).

## This lab (2×GB10) — already custom

- TensorFold 0.6.0 + Mia + RAIDFLASH overlay (not a Zig jump)
- In-engine packer `POST /v1/raidflash/complete`
- DepthPolicy `most`, bind_d unfix, no fc7 clobber
- EFI map k=1,2,4,8 live: D 49.5→121.1, P-hot 5.94×, P-cold flat
- Injector: 1 user [8], 3 users [3,3,2]

Keep this engine hot. It is the development die.

## 2×B300 (~$240k) — same computer, bigger map

Public (StorageReview 2026-10-05, two GB300 Stations, 400G DACs):

| | 1 stream | RAIDFLASH target |
|---|---|---|
| Flash PD pair | 362 | **2721 @32** (elbow). 3256 @256 is +20% |
| GLM-5.3 full TP2 | 188 one tower | **5018** on the pair @256 |
| DSV4.1 Flash PD | 441 | **5248 @256** |

Injector (`EfiMap.gb300_flash_pair()`, `--parallel` 128):

```
1 user   D  → k=32     ~2700 tok/s  (thousands, one cursor of stripes)
4 users  D  → [32×4]   die ~2950, ~740 each
```

128 lanes are the seat count. The Flash **elbow** is ~32. Four people each
get the elbow. One person gets thousands. Nobody sits at 50 tok/s while
96 lanes idle.

GLM full still climbs at 128–256: `gb300_glm_full_pair()` gives 1 user
k=128, 4 users [32,32,32,32].

Measure a live EFI on that die the day it lands. Copy the computer, not
the 8 from GB10.

## Before market — engine work (this order)

1. **In-engine EFI.** `n_users` inside TensorFold (one ChatJob, k lanes,
   `requests_running=1` to the client). Client packer already splits
   sections; the scheduler should.
2. **`--name-priority`.** RAIDFLASH job is foreground; batch yields.
3. **Keep Mia knobs** (prefix, fill 4096/2048, copy drafts, fnc7, fat pool)
   as ours even if upstream drops them.
4. **GitHub v0.1** when `gh auth` exists.
5. **B300 EFI night.** Same `medir_efi_map.py` grid at k=1,8,32,64,128.

Do not restart the GB10 serving path for this list. Do not jump to Zig
until the full EXL3/NVFP4 checkpoint serves TP2 with parallel+prefix.

Occupancy on CAKE hardware: P and D stripes on the engine, EFI-metered,
for the work in front of the AI PC.
