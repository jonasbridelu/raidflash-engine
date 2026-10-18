"""RAIDFLASH-P — hide prefill wait, then hand the kept prefix to decode lanes.

Prefill has no draft acceptance. Either the KV is there or the GEMM runs.
Cold k-at-once loses (sisters re-prefill). Cascade wins: seed the document
(drafted, max_tokens >= 16 so the keep carries a DFlash window), then sisters
whose prompt is a byte prefix of that seed plus a task suffix.

Measured (GB10 TP2, GLM-5.3-Flash EXL3, clean delta==sum):
  seed 4.5s prefill, 0 cache
  8 sisters 98.6–98.7% cached at grid 6400, prefill 0.49–0.87s
  engine --parallel 4: peak running 4, second wave still inherits
"""
from __future__ import annotations

SEED_MIN_TOKENS = 16  # serial seed (mt<16) leaves drafter_end=-1; _resume drops it

# Parallel-thought aspects for Packer.think. k=4 is the measured reasoning peak.
THOUGHT_ASPECTS = (
    "decompose the problem, fix a hypothesis and what is still unknown",
    "look for contradictions, edge cases, and failures of the obvious hypothesis",
    "pick the shortest path to a verifiable answer",
    "close with the answer and a one-line check",
)


def thought_sections(question: str, k: int = 4) -> list[str]:
    """k occupancy lanes of one question. Prefix-identical → P-hot cache."""
    k = max(1, int(k))
    n = len(THOUGHT_ASPECTS)
    out = []
    for i in range(k):
        aspect = THOUGHT_ASPECTS[i % n]
        if k > n:
            aspect = f"{aspect} (branch {i + 1}/{k})"
        out.append(f"{aspect}. Question: {question}")
    return out


def seed_prompt(document: str) -> str:
    """The seed IS the document. Sisters must start with this exact prefix."""
    return document


def sister_prompts(document: str, sections: list[str]) -> list[str]:
    k = len(sections)
    out = []
    for i, sec in enumerate(sections, 1):
        out.append(
            document
            + f"\n\nTask {i}/{k}: write ONLY section '{sec}' in 2 sentences "
            f"about the document above. Do not write the other sections or an introduction. "
            f"End with FIN{i}."
        )
    return out


def merge_sisters(texts: list[str]) -> str:
    """Concatenate sister bodies in spine order. Strips trailing FIN marks."""
    parts = []
    for i, t in enumerate(texts, 1):
        body = (t or "").strip()
        marker = f"FIN{i}"
        if body.endswith(marker):
            body = body[: -len(marker)].rstrip()
        parts.append(body)
    return "\n\n".join(p for p in parts if p)
