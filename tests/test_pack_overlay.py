"""EFI injector inside the in-engine overlay (no TensorFold import)."""
import importlib.util
from pathlib import Path


def _pack():
    p = Path(__file__).resolve().parents[1] / "patches" / "tensorfold" / "raidflash_pack.py"
    spec = importlib.util.spec_from_file_location("raidflash_pack_overlay", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_user_fills_the_die():
    p = _pack()
    assert p.k_for(8, n_users=1, parallel=8) == 8
    assert p.k_for(3, n_users=1, parallel=8) == 3


def test_three_users_split_eight():
    p = _pack()
    assert p.split_lanes(3, 8, 8) == [3, 3, 2]
    assert p.k_for(8, n_users=3, user_index=0, parallel=8) == 3
    assert p.k_for(8, n_users=3, user_index=2, parallel=8) == 2


def test_p_cold_one_lane():
    p = _pack()
    assert p.k_for(8, n_users=3, user_index=0, phase="P_cold", parallel=8) == 1


def test_more_sections_than_codo_still_caps():
    p = _pack()
    assert p.k_for(64, n_users=1, phase="D", parallel=8) == 8
