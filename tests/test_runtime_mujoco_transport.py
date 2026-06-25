import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.runtime.mujoco_transport import MujocoTransport

def test_read_state_shape_and_settle():
    t = MujocoTransport(make_wb_config())
    x = t.read_state()
    assert x.shape == (68,)
    assert abs(x[2] - make_wb_config().nominal_base_height) < 0.05   # spawned near nominal height
