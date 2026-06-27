import numpy as np
from t1_nmpc.robot.config import make_config
from sim.stand import run_stand

def test_closed_loop_stand_holds():
    cfg = make_config()
    m = run_stand(cfg, duration=2.0, view=False)
    assert not m["fell"], "robot fell during stand"
    assert 0.9 <= m["fz_ratio_p50"] <= 1.1, m["fz_ratio_p50"]
    assert m["max_tilt_deg"] < 10.0, m["max_tilt_deg"]
    assert m["solve_p90_ms"] < 60.0, m["solve_p90_ms"]
