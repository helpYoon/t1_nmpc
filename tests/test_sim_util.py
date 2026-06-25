import numpy as np
from sim._sim_util import tilt_from_quat_wxyz, upright_ok

def test_tilt_identity_quat_is_zero():
    assert tilt_from_quat_wxyz(np.array([1.0, 0, 0, 0])) < 1e-9

def test_upright_ok_thresholds():
    assert upright_ok(0.95, 0.05, 1.0) is True
    assert upright_ok(0.80, 0.05, 1.0) is False
    assert upright_ok(0.95, 0.30, 1.0) is False
