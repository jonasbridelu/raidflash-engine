from raidflash.prefill import (
    bind_outline,
    continue_ids,
    merge_sisters,
    outline_seed_prompt,
    seed_prompt,
    sister_prompts,
)


def test_sisters_are_prefix_of_seed():
    doc = "Meeting minutes.\n"
    seed = seed_prompt(doc)
    sisters = sister_prompts(doc, ["A", "B", "C", "D"])
    assert len(sisters) == 4
    for s in sisters:
        assert s.startswith(seed)
        assert s != seed


def test_outline_sisters_share_seed_bytes():
    doc = "Meeting minutes.\n"
    secs = ["A", "B"]
    seed = outline_seed_prompt(doc, secs)
    assert seed.startswith(doc)
    assert "1. A" in seed and "2. B" in seed
    shared = bind_outline(seed, "1. A\n2. B FIN0")
    assert shared.endswith("1. A\n2. B")
    assert not shared.endswith("FIN0")
    sisters = sister_prompts(shared, secs)
    for s in sisters:
        assert s.startswith(seed)
        assert "OUTLINE:" in s
        assert "1. A" in s


def test_merge_strips_fin():
    texts = ["one FIN1", "two FIN2"]
    assert merge_sisters(texts) == "one\n\ntwo"


def test_continue_ids():
    assert continue_ids([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert continue_ids([], [9]) == [9]
