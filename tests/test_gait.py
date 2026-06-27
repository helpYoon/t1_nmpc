import numpy as np
from t1_nmpc.wb.gait import StandGait

def test_stand_schedule_all_contact():
    g = StandGait(n_corners=8)
    dts = [0.02]*14
    cs = g.contact_schedule(0.0, dts, 14)
    sw = g.swing_schedule(0.0, dts, 14)
    assert cs.shape == (8, 14) and sw.shape == (8, 14)
    assert np.all(cs == 1.0) and np.all(sw == 0.0)
