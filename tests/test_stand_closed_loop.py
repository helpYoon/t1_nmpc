import numpy as np
from t1_nmpc.robot.config import make_config
from sim.stand import run_stand

def test_closed_loop_stand_holds():
    cfg = make_config()
    m = run_stand(cfg, duration=4.0)
    assert not m["fell"], "robot fell during stand"
    assert 0.9 <= m["fz_ratio_p50"] <= 1.1, m["fz_ratio_p50"]
    assert 0.9 <= m["grf_ratio_p50"] <= 1.1, m["grf_ratio_p50"]   # MuJoCo-measured plant GRF
    assert m["max_tilt_deg"] < 10.0, m["max_tilt_deg"]
    assert m["solve_p90_ms"] < 60.0, m["solve_p90_ms"]
