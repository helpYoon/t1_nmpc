# tests/test_gait.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.wb.gait import WalkGait, StandGait, v_z_ref


def test_stand_always_double_support():
    g = StandGait(make_config())
    for t in np.linspace(0, 3, 13):
        assert g.mode_at(t) == (True, True)
        assert g.swing_phase(t, 0) is None


def test_walk_mode_sequence():
    cfg = make_config(); g = WalkGait(cfg)
    assert g.mode_at(0.0) == (False, True)     # LF swing
    assert g.mode_at(0.65) == (True, True)     # double
    assert g.mode_at(1.0) == (True, False)     # RF swing
    assert g.mode_at(1.35) == (True, True)     # double
    assert g.mode_at(1.4) == g.mode_at(0.0)    # periodic
    assert g.mode_at(0.6) == (True, True)    # LF-swing ends, double begins
    assert g.mode_at(0.7) == (True, False)   # double ends, RF-swing begins
    assert g.mode_at(1.3) == (True, True)    # RF-swing ends, double begins


def test_walk_swing_phase_progresses():
    cfg = make_config(); g = WalkGait(cfg)
    p0 = g.swing_phase(0.0, 0); p1 = g.swing_phase(0.3, 0)
    assert p0 is not None and 0.0 <= p0 < 0.1
    assert p1 is not None and 0.4 < p1 < 0.6
    assert g.swing_phase(0.0, 1) is None       # RF not swinging at t=0


def test_horizon_modes_length():
    cfg = make_config(); g = WalkGait(cfg)
    modes = g.horizon_modes(0.0)
    assert len(modes) == cfg.nodes


def test_v_z_ref_shape():
    cfg = make_config()
    assert v_z_ref(0.0, cfg) > 0          # liftoff rising
    assert abs(v_z_ref(0.5, cfg)) < 1e-6  # zero at apex
    assert v_z_ref(1.0, cfg) < 0          # touchdown descending
