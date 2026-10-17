from raidflash.efi import EfiMap, split_lanes


def test_one_user_takes_the_die():
    assert split_lanes(1, 8, 8) == [8]
    assert split_lanes(1, 8, 4) == [4]


def test_three_users_decode_split_max_lanes():
    # 8 slots, 3 users, D still pays at 8 → fill the die [3,3,2]
    assert split_lanes(3, 8, 8) == [3, 3, 2]


def test_three_users_p_cold_one_lane_each():
    assert split_lanes(3, 8, 1) == [1, 1, 1]


def test_more_users_than_slots():
    assert split_lanes(10, 8, 8) == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_gb10_plan_three_users_d():
    m = EfiMap.gb10()
    plan = m.plan(3, "D")
    assert plan["k"] == [3, 3, 2]
    assert plan["idle"] == 0
    assert plan["die_tok_s"] > 0
    assert min(x for x in plan["tok_s"] if x) > 0


def test_gb10_one_user_d_beats_c1():
    m = EfiMap.gb10()
    one = m.plan(1, "D")
    three = m.plan(3, "D")
    assert one["k"] == [8]
    # same die fill: 8 lanes. Split does not dump two users to zero.
    assert sum(three["k"]) == 8
    assert one["tok_s"][0] > three["tok_s"][0]


def test_p_cold_codo_is_one():
    m = EfiMap.gb10()
    assert m.k_codo("P_cold") == 1
    assert m.plan(3, "P_cold")["k"] == [1, 1, 1]


def test_b300_flash_one_user_thousands():
    m = EfiMap.gb300_flash_pair()
    one = m.plan(1, "D")
    assert one["k"] == [32]
    assert one["tok_s"][0] >= 2700


def test_b300_flash_four_users_fill_128():
    m = EfiMap.gb300_flash_pair()
    four = m.plan(4, "D")
    assert four["k"] == [32, 32, 32, 32]
    assert four["idle"] == 0
    assert min(four["tok_s"]) >= 700
    assert four["die_tok_s"] >= 2700


def test_b300_glm_full_one_user_128():
    m = EfiMap.gb300_glm_full_pair()
    one = m.plan(1, "D")
    four = m.plan(4, "D")
    assert one["k"] == [128]
    assert four["k"] == [32, 32, 32, 32]
    assert one["tok_s"][0] > four["tok_s"][0]
