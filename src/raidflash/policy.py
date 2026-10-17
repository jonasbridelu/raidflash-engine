"""RAIDFLASH-D — decode policy that does not replace the engine optimizer.

Short requests stay serial. Tool/grammar constraints stay untouched.
Unconstrained long streams raise DepthPolicy.most and unfix it so next()
can drop depth when acceptance falls. In-round chain_rule (beta / confidence)
stays in the engine.

Never clobber fnc/fcost with a fixed fc7 code: that was turbo v1, and it
disabled DepthPolicy.next().
"""
from __future__ import annotations

import os
import sys

def _env_int(name: str, legacy: str, default: str) -> int:
    return int(os.environ.get(name, os.environ.get(legacy, default)))


def _flag_path() -> str:
    return os.environ.get("RAIDFLASH_FLAG", os.environ.get("GLM53_TF_RAIDFLASH_FLAG", "/tmp/raidflash.on"))


MIN_TOKENS = 16
D_MOST = 7
BUSY_STREAMS = 4

# Compat only. Do not apply. kind 13 = fc7:0.3 without noisy beta.
TURBO_CODE = [13, 7, 300000, 0]


def enabled() -> bool:
    if os.environ.get("RAIDFLASH", os.environ.get("GLM53_TF_RAIDFLASH", "0")) == "1":
        return True
    try:
        return os.path.exists(_flag_path())
    except OSError:
        return False


def turbo_enabled() -> bool:
    return os.environ.get("RAIDFLASH_TURBO", os.environ.get("GLM53_TF_RAIDFLASH_TURBO", "1")) == "1"


def _log(msg: str) -> None:
    try:
        print(f"[raidflash-D] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _n_active(engine) -> int:
    lanes = getattr(engine, "lanes", None)
    if not lanes:
        return 0
    vals = lanes.values() if isinstance(lanes, dict) else lanes
    n = 0
    for lane in vals:
        if getattr(lane, "decoding", False) or getattr(lane, "s", None) is not None:
            n += 1
    return n


def _bump_most(code: list, most: int) -> list:
    if not code or not code[0]:
        return code
    out = list(code)
    if out[1] < most:
        out[1] = most
    return out


def adapt_code(code: list, max_tokens: int, constraint=None, engine=None):
    """multi.py path. Returns (code, forced_serial)."""
    if not enabled():
        return code, False
    if constraint is not None:
        _log("keep engine (constraint/tool)")
        return code, False
    if not code or not code[0]:
        return code, False
    min_tok = _env_int("RAIDFLASH_MIN_TOKENS", "GLM53_TF_RAIDFLASH_MIN_TOKENS", "16")
    busy = _env_int("RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY", "4")
    most = _env_int("RAIDFLASH_D_MOST", "GLM53_TF_RAIDFLASH_D_MOST", "7")
    if max_tokens < min_tok:
        _log(f"degrade denso->serial: max_tokens={max_tokens} < {min_tok}")
        return [0, 0, 0, 0], True
    n = _n_active(engine)
    if n >= busy:
        _log(f"keep engine (busy n={n}): kind={code[0]} most={code[1]}")
        return code, False
    if not turbo_enabled():
        return code, False
    bumped = _bump_most(code, most)
    if bumped != list(code):
        _log(f"raise most {code[1]}->{bumped[1]} kind={code[0]} n={n} mt={max_tokens}")
    else:
        _log(f"keep denso kind={code[0]} most={code[1]} n={n} mt={max_tokens}")
    return bumped, False


def bind_d(policy, engine=None):
    """Unfix DepthPolicy so next() can drop depth. In-round chain_rule stays."""
    if not enabled() or policy is None or isinstance(policy, tuple):
        return policy
    if not hasattr(policy, "fixed") or not hasattr(policy, "most"):
        return policy
    n = _n_active(engine)
    busy = _env_int("RAIDFLASH_BUSY", "GLM53_TF_RAIDFLASH_BUSY", "4")
    most = _env_int("RAIDFLASH_D_MOST", "GLM53_TF_RAIDFLASH_D_MOST", "7")
    if n >= busy:
        if policy.most > 3:
            _log(f"bind busy n={n}: most {policy.most}->3")
            policy.most = 3
        return policy
    if turbo_enabled():
        if policy.most < most:
            _log(f"bind most {policy.most}->{most}")
            policy.most = most
        if policy.fixed:
            policy.fixed = False
            _log(f"bind unfix DepthPolicy most={policy.most}")
    return policy


def adapt(policy, max_tokens: int, code, engine=None):
    """engine.py _run path: (policy, forced_serial)."""
    if not enabled():
        return policy, False
    if policy is None:
        return policy, False
    if max_tokens < MIN_TOKENS:
        _log(f"degrade denso->serial: max_tokens={max_tokens} < {MIN_TOKENS}")
        return None, True
    return bind_d(policy, engine), False


def depth_next(most: int, rate: float, low: float = 0.8, high: float = 0.9, fixed: bool = False) -> int:
    """Engine-native DepthPolicy.next high arm uses ``most`` (was hardcoded 3).

    most=3 stays 1/2/3. most=7 becomes 1/4/7 as the EMA heats.
    """
    if fixed:
        return most
    if rate < low:
        cap = 1
    elif rate < high:
        cap = max(2, (most + 1) // 2)
    else:
        cap = most
    return max(1, min(most, cap))
