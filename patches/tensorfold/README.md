# TensorFold overlay v0.1 (surgical)

Do not vendor TensorFold. Do not copy `http.py`, `multi.py`, or `decode.py`.
Tree speculation is **OFF**. MLA/Hydragen is Dynamo, not this overlay.

Drop these two files into `tensorfold/families/glm5_next/cuda/`:

- `raidflash_pack.py` — in-engine packer (`App.run`, no HTTP loopback)
- `raidflash_policy.py` — call-time env reads; `BUSY` default **8** (not frozen at import)

Then apply three hooks.

## 1. `decode.py` — `DepthPolicy.next` high arm = `self.most`

Replace the hardcoded `else 3` with `self.most`:

```python
if self.rate < self.low:
    cap = 1
elif self.rate < self.high:
    cap = max(2, (self.most + 1) // 2)
else:
    cap = self.most
return max(1, min(self.most, cap))
```

`most=3` stays 1/2/3. `most=7` becomes 1/4/7 as the EMA heats.

## 2. `multi.py` — admit: `adapt_code` / `bind_d`

After `code = multi_code(...)`:

```python
from . import raidflash_policy as _rf
code, serial = _rf.adapt_code(code, s.count, getattr(s, "constraint", None), self)
if serial:
    code = [0, 0, 0, 0]
    s.draft = False
```

After `policy = decode_policy(code, g.costs)`:

```python
if policy is not None:
    policy = _rf.bind_d(policy, self)
```

**NEVER** assign `TURBO_CODE` / `fc7`. That clobbers noisy `fnc7` and sets `fixed=True`, so `DepthPolicy.next()` never drops.

## 3. `cuda/http.py` — `_raidflash` only

Add `Handler._raidflash`. Dispatch:

- `POST /v1/raidflash/complete`
- `POST /v1/chat/completions` when the body has `raidflash.sections`

Do not copy the rest of TensorFold `http.py`. Handler (live v0.6.1, abbreviated):

```python
def _raidflash(self, body: dict[str, Any], model: str, gone) -> None:
    """One POST: seed the document, then k sister streams on the engine. No HTTP fan-out."""
    from tensorfold.families.glm5_next.cuda.raidflash_pack import SEED_MIN_TOKENS, complete as pack

    rf = body.get("raidflash") if isinstance(body.get("raidflash"), dict) else body
    document = rf.get("document")
    if not document:
        msgs = body.get("messages") or []
        if msgs and isinstance(msgs[-1], dict):
            document = msgs[-1].get("content")
    sections = rf.get("sections") or []
    stripe = int(rf.get("stripe_tokens") or body.get("max_tokens") or 80)
    seed_tokens = int(rf.get("seed_tokens") or SEED_MIN_TOKENS)
    n_users = int(rf.get("n_users") or 0) or None
    user_index = int(rf.get("user_index") or 0)
    phase = str(rf.get("phase") or "P_hot")
    out = pack(app, str(document or ""), list(sections),
               stripe_tokens=stripe, seed_tokens=seed_tokens, model=model,
               cancelled=lambda: gone.cancelled,
               n_users=n_users, user_index=user_index, phase=phase)
    # 200 chat.completion + usage + raidflash.{k, n_users, k_plan, wall_s, seed, sisters}
```

In `do_POST`, after reading JSON:

```python
raidflash = route.endswith("/raidflash/complete")
chat = route.endswith("/chat/completions")
rf = body.get("raidflash") if isinstance(body, dict) else None
if raidflash or (chat and isinstance(rf, dict) and rf.get("sections")):
    return self._raidflash(body, app.reply_model(body), socket_cancellation(self.connection))
```

## Flag

`touch /tmp/raidflash.on` or `RAIDFLASH=1` / `GLM53_TF_RAIDFLASH=1`.
