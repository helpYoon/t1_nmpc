from t1_nmpc.robot.config import make_track_config
from sim.pickup import run_pickup


def test_closed_loop_tracks_first_reach():
    # ~12 s wall (= plan ~2.4 s): the first deep floor-reach + left grasp. Must track the reference
    # lean (reference-RELATIVE tilt small), not fall, keep feet loaded. (Full motion is the Step-5 run.)
    cfg = make_track_config(time_scale=5.0)
    res = run_pickup(cfg, duration=12.0)
    assert res["fell"] is False
    assert res["max_reltilt_deg"] < 10.0               # tracks the reference lean (NOT absolute tilt)
    assert 0.80 < res["fz_ratio_p50"] < 1.20           # Σfz/mg ~ 1 (balanced)
    assert res["solve_p90_ms"] > 0.0
