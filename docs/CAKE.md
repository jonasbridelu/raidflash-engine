# CAKE

CAKE is the company. It builds **AI PCs for enterprise and business**,
and it runs **custom kernels** on those machines.

RAIDFLASH is the occupancy layer on that stack: one job fills every free
GPU stream that still pays, for prefill (P) and decode (D). Commercial
serving parks ~50 tok/s on a cursor and keeps the rest of the die for N
tenants. A CAKE box spends the aggregate on the work in front of it.

```
k = min(PARALLEL, k_codo_phase)
```

- P: seed the document, then sister stripes (prefix cache)
- D: dense decode on those lanes until the knee
- Injector: `k` per user, per phase

Open occupancy (`raidflash-engine`) is how the measured tok/s stay honest
on silicon CAKE already ships.

CAKE.
