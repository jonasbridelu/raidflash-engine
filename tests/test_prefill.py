from raidflash.prefill import merge_sisters, seed_prompt, sister_prompts


def test_sisters_are_prefix_of_seed():
    doc = "Meeting minutes.\n"
    seed = seed_prompt(doc)
    sisters = sister_prompts(doc, ["A", "B", "C", "D"])
    assert len(sisters) == 4
    for s in sisters:
        assert s.startswith(seed)
        assert s != seed


def test_merge_strips_fin():
    texts = ["one FIN1", "two FIN2"]
    assert merge_sisters(texts) == "one\n\ntwo"
