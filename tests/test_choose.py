from raidflash.choose import choose_k, seed_then_sisters


def test_short_thinking_uses_four():
    assert choose_k(kind="decode", thinking=True, max_tokens=80) == 4
    assert choose_k(kind="decode", thinking=True, max_tokens=10) == 4


def test_long_thinking_fills_eight():
    assert choose_k(kind="decode", thinking=True, max_tokens=256) == 8
    assert choose_k(kind="decode", thinking=True, max_tokens=1024) == 8


def test_decode_without_thinking_stays_eight():
    assert choose_k(kind="decode", max_tokens=80) == 8


def test_one_user_decode_fills_eight():
    assert choose_k(kind="decode") == 8


def test_one_user_packer_fills_eight():
    assert choose_k(kind="packer", n_work=8) == 8
    assert choose_k(kind="packer", n_work=3) == 3


def test_one_document_seed_is_one_lane():
    assert choose_k(kind="prefill_one") == 1


def test_many_cold_docs_fill_eight():
    assert choose_k(kind="prefill_many") == 8


def test_three_users_decode():
    assert [choose_k(kind="decode", n_users=3, user_index=i) for i in range(3)] == [3, 3, 2]


def test_three_users_one_doc_seed():
    assert [choose_k(kind="prefill_one", n_users=3, user_index=i) for i in range(3)] == [1, 1, 1]


def test_seed_then_sisters_one_user():
    j = seed_then_sisters(n_sections=8)
    assert j["seed_k"] == 1
    assert j["sister_k"] == 8
    assert j["seed_tokens"] == 16
