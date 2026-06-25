from sim.wb_stand_croco import run_wb_stand_croco

def test_m0_stand_holds():
    m = run_wb_stand_croco(duration_s=3.0, control_hz=60.0)
    assert m["n_fail"] == 0
    assert m["peak_tilt_rad"] is not None and m["peak_tilt_rad"] < 0.05
    assert m["final_base_z"] > 0.85 * 0.6734
    assert m["held"] is True
