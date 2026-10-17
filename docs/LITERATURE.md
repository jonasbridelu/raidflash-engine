# Literatura para tunear P (TTFT) y D (tok/s) — 2026-10-16

Clase de trabajo: 1 usuario, prefijo compartido, MoE GLM-5.3-Flash, 2×GB10
278 GB/s LPDDR5x, TensorFold ya tiene SHARED_PREFIX, FILL_ROWS=2048, DFlash2, parallel 8.

## Qué ya está medido (no repetir)

| idea | paper / sistema | nuestro resultado |
|---|---|---|
| overlap P∥D a nivel request | DistServe / SARATHI mix | **M5 falsificado** (−59% decode) |
| árbol especulativo T3 | ASPD / SoT | engine sano, **0 ganancia** |
| MULTI_WINDOW 64 | TF docs | **peor** P y D (all-gather) |
| k frío simultáneo | batching clásico | M6 **0% cache**, P-cold agg **plano ~1500 tok/s** |
| seed + hermanas | radix / APC | **99.3%, 51200 cached_tokens** |

## Qué sí ataca el hueco que queda

### 1. Hydragen — arXiv 2402.05099 — PRIORIDAD 1
Tras P-hot, el decode de 8 hermanas **releen el KV del prefijo 8 veces** (278 GB/s).
Hydragen: atención prefijo vs sufijo; batch de Q sobre el prefijo → GEMM, no 8× gemv.
Exacto. Hasta 32× con prefijo largo y batch grande.
Repo: github.com/ScalingIntelligence/hydragen
**Bloqueo:** TF usa ventanas MLA (`MLA_DECODE_ROWS`). PLAN-ACCION-3PAPERS (sep-10) ya lo marcó #1; hay que portar a MLA, no a Llama GQA.
Eso es el tok/s de D **después** de que P ya cacheó.

### 2. IPPD — arXiv 2609.05707
Decode intra-prompt de preguntas con contexto común. Hasta 7× vs prompting independiente.
Vecino más cercano a RAIDFLASH. M3 sidecar :8010, no tocar motor vivo.
Complementa Hydragen (máscara/posiciones vs kernel de atención).

### 3. SARATHI chunked prefill — knob que YA tenemos
`TF_GLM_FILL_ROWS=2048`. Papers: chunk grande → mejor TTFT / GEMM; chunk chico → D no se congela.
Para **1 usuario** que quiere menos espera de prefill: probar **4096 y 8192**.
P-cold ya satura SMs a k=1 (~1500 tok/s): un chunk más ancho puede subir ese techo (menos launches, GEMM más grueso). Barato: env + relanzar.

### 4. MoE-Prefill — arXiv 2605.02960 + DuoServe-MoE 2509.07379
Prefill MoE activa casi todos los expertos (denso agregado); decode es top-k (RAM).
Prefetch de expertos **por peso** solapado con el GEMM. Encaja la tesis 278 GB/s: D sí, P no tanto.
TF ya cachea expertos; el hueco es overlap prefetch↔compute en prefill, no más k HTTP.

### 5. Splitwise / DistServe PD disagg
StorageReview 2×GB300: Flash 353 C1 → 2721 @32 con P en una torre y D en la otra.
Nosotros somos TP2, no PD. Es arquitectura de 2 cajas, no un env.

## Orden de ataque en ESTE engine

1. **FILL_ROWS 2048 vs 4096 vs 8192** — P-cold tok/s reales (barato).
2. **Hydragen-MLA** sobre hermanas P-hot — D tok/s (caro, es el 32× teórico).
3. IPPD sidecar solo si Hydragen no entra en MLA.

No reabrir overlap request-level, árbol T3, ni window 64.
