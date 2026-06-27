import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.wb.gait import WalkGait, StandGait


def test_stand_schedules_all_contact():
    c = make_config(); ct, sw = StandGait(c).schedules(0.0)
    assert ct.shape == (2, c.nodes) and sw.shape == (2, c.nodes)
    assert np.all(ct == 1.0) and np.all(sw == 0.0)


def test_walk_mode_sequence():
    c = make_config(); g = WalkGait(c)
    ct, sw = g.schedules(0.0)
    # node 0 at t=0 -> LF swing (foot0 out), RF stance (foot1 in)
    assert ct[0, 0] == 0.0 and ct[1, 0] == 1.0
    assert 0.0 <= sw[0, 0] < 0.1 and sw[1, 0] == 0.0
    # a node near t=1.0 (RF swing): find node index for ~1.0s
    i = int(round((1.0 - 0.0) / c.dt_min))
    if i < c.nodes:
        assert ct[1, i] == 0.0 and ct[0, i] == 1.0


def test_walk_boundaries_and_periodicity():
    c = make_config(); g = WalkGait(c)
    # mode_at helper boundaries (half-open)
    assert g.mode_at(0.6) == (True, True)
    assert g.mode_at(0.7) == (True, False)
    assert g.mode_at(1.3) == (True, True)
    assert g.mode_at(1.4) == g.mode_at(0.0)
