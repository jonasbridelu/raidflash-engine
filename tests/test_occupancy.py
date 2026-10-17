from raidflash.occupancy import Occupancy, k_target


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
