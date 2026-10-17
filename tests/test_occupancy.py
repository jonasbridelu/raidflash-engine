from raidflash.occupancy import Occupancy, k_target, knee, pays


def test_empty_machine_opens_parallel():
    assert k_target(4, 0, 4) == 4
    assert k_target(4, 0, 8) == 4  # cap is recipe parallel


def test_partial_fill():
    assert k_target(4, 1, 4) == 3
    assert k_target(4, 4, 4) == 0
    assert k_target(4, 4, 8) == 0


def test_occupancy_p_and_d():
    o = Occupancy(parallel=4, k_codo_d=4, k_codo_p=8, n_active=0)
    assert o.d_open() == 4
    assert o.p_open() == 4
    assert o.idle() == 4
    o.n_active = 4
    assert o.filling()
    assert o.p_open() == 0


def test_p_knee_wider_than_d():
    o = Occupancy(parallel=8, k_codo_d=4, k_codo_p=8)
    assert o.d_open() == 4
    assert o.p_open() == 8


def test_set_knees_b300_style():
    o = Occupancy(parallel=128, k_codo_d=4, k_codo_p=8)
    o.set_knees(k_codo_d=64, k_codo_p=120)
    assert o.d_open() == 64
    assert o.p_open() == 120


def test_knee_stops_when_agg_flat():
    # P-cold GB10: GEMM already full at k=1 (~1500 tok/s)
    assert knee([1, 4, 8], [1500, 1510, 1492], eps=0.05) == 1
    # D GB10 live 2026-10-17: 34.2 → 103.5 still pays at 8
    assert pays(34.2, 103.5)
    assert knee([1, 8], [34.2, 103.5]) == 8
    # B300-class: 64 pays, 120 does not
    assert knee([1, 32, 64, 120], [350, 1800, 2700, 2750], eps=0.05) == 64
