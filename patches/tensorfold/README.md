# TensorFold overlay v0.1

Do not vendor TensorFold. Apply two surgical hooks.

## 1. `decode.py` — `DepthPolicy.next`

Replace the hardcoded `else 3` high arm with `self.most`:

```python
if self.rate < self.low:
    cap = 1
elif self.rate < self.high:
    cap = max(2, (self.most + 1) // 2)
else:
    cap = self.most
return max(1, min(self.most, cap))
```

`most=3` stays 1/2/3. `most=7` becomes 1/4/7.

## 2. `multi.py` — admit + policy bind

On admit, after `code = multi_code(...)`:

```python
from raidflash.policy import adapt_code, bind_d, enabled
if enabled():
    code, serial = adapt_code(code, s.count, getattr(s, "constraint", None), self)
    if serial:
        code = [0, 0, 0, 0]
        s.draft = False
```

After `policy = decode_policy(code, g.costs)`:

```python
if enabled() and policy is not None:
    policy = bind_d(policy, self)
```

Never assign `TURBO_CODE` (fc7). That clobbers noisy `fnc7` and sets `fixed=True`.

## 3. Flag

`touch /tmp/raidflash.on` or `RAIDFLASH=1`.
