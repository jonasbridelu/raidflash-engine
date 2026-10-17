"""RAIDFLASH — one user, every free GPU stream, until the knee.

D = decode occupancy (depth policy + k lanes).
P = prefill occupancy (seed then sisters; no draft acceptance).
The packer turns one OpenAI-compatible request into k engine streams.
"""

from .efi import EfiMap, split_lanes
from .occupancy import Occupancy, k_target, knee, pays
from .policy import adapt_code, bind_d, enabled
from .prefill import sister_prompts, seed_prompt
from .packer import Packer

__version__ = "0.1.0"
__all__ = [
    "Occupancy",
    "EfiMap",
    "split_lanes",
    "k_target",
    "knee",
    "pays",
    "adapt_code",
    "bind_d",
    "enabled",
    "sister_prompts",
    "seed_prompt",
    "Packer",
]
