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
OUTLINE_SEED_TOKENS = 64  # enough for a k-line spine; 16 truncates headings
OUTLINE_MARK = "\n\nOUTLINE:\n"
SO_FAR_MARK = "\n\nSO_FAR:\n"

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


def continue_ids(prompt_ids: list[int], generated: list[int]) -> list[int]:
    """Append generated token ids onto the live prompt. No chat-template wrap."""
    return list(prompt_ids) + list(generated)


def seed_prompt(document: str) -> str:
    """v1.0 seed IS the document. Sisters must start with this exact prefix."""
    return document


def outline_seed_prompt(document: str, sections: list[str]) -> str:
    """v1.1 seed: same document prefix, plus 'write these headings'."""
    titles = "\n".join(f"{i}. {s}" for i, s in enumerate(sections, 1))
    return (
        document
        + "\n\nWrite ONLY a numbered outline with exactly these headings, "
        "one line each, no body:\n"
        + titles
        + "\nEnd with FIN0."
    )


def strip_fin(text: str, mark: str) -> str:
    body = (text or "").strip()
    if body.endswith(mark):
        body = body[: -len(mark)].rstrip()
    return body


def bind_outline(seed_prompt_text: str, outline_text: str) -> str:
    """Shared prefix for sisters: seed prompt + generated spine."""
    return seed_prompt_text + OUTLINE_MARK + strip_fin(outline_text, "FIN0")


def sister_prompts(prefix: str, sections: list[str]) -> list[str]:
    """Each sister is prefix + task suffix. Prefix must be a byte-prefix of the seed."""
    k = len(sections)
    out = []
    for i, sec in enumerate(sections, 1):
        out.append(
            prefix
            + f"\n\nTask {i}/{k} section '{sec}': continue from the outline and text above. "
            f"Write ONLY this section, up to the token limit. No introduction, no other sections. "
            f"End with FIN{i}."
        )
    return out


def merge_sisters(texts: list[str]) -> str:
    """Concatenate sister bodies in spine order. Strips trailing FIN marks."""
    parts = []
    for i, t in enumerate(texts, 1):
        body = strip_fin(t, f"FIN{i}")
        parts.append(body)
    return "\n\n".join(p for p in parts if p)
