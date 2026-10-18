"""RAIDFLASH — one user, every free GPU stream, until the knee.

D = decode occupancy (depth policy + k lanes).
P = prefill occupancy (seed then sisters; no draft acceptance).
The packer turns one OpenAI-compatible request into k engine streams.
"""

from .choose import choose_k, seed_then_sisters
from .efi import EfiMap, split_lanes
from .occupancy import Occupancy, k_target, knee, pays
from .policy import adapt_code, bind_d, enabled, most_for_lanes
from .prefill import sister_prompts, seed_prompt, thought_sections
from .packer import Packer

__version__ = "0.2.1"
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
    "thought_sections",
    "Packer",
]
