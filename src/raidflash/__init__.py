"""RAIDFLASH — one user, every free GPU stream, until the knee.

D = decode occupancy (depth policy + k lanes).
P = prefill occupancy (seed then sisters; no draft acceptance).
The packer turns one OpenAI-compatible request into k engine streams.
v1.1: outline seed, optional sync rounds, optional polish.
v1.1.2: KV-continue fork — leader token ids stay on the prompt.
"""

from .choose import choose_k, seed_then_sisters
from .efi import EfiMap, split_lanes
from .occupancy import Occupancy, k_target, knee, pays
from .policy import adapt_code, bind_d, enabled, most_for_lanes
from .prefill import (
    bind_outline,
    continue_ids,
    outline_seed_prompt,
    seed_prompt,
    sister_prompts,
    thought_sections,
)
from .packer import Packer

__version__ = "0.3.1"
__all__ = [
    "Occupancy",
    "EfiMap",
    "choose_k",
    "seed_then_sisters",
    "split_lanes",
    "k_target",
    "knee",
    "pays",
    "adapt_code",
    "bind_d",
    "enabled",
    "most_for_lanes",
    "sister_prompts",
    "seed_prompt",
    "outline_seed_prompt",
    "bind_outline",
    "continue_ids",
    "thought_sections",
    "Packer",
]
