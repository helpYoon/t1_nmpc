# tests/test_wb_config_walk.py
import numpy as np
from t1_nmpc.wb.config import make_wb_config


def test_walking_config_fields():
    cfg = make_wb_config()
    assert cfg.foot_linacc_err_gain_z == 1.0
    np.testing.assert_array_equal(cfg.swingfoot_cost_weights, [1e4, 1e4, 5.0, 5.0, 2.0, 2.0, 2.0])
    assert cfg.arm_swing_amplitude == 0.15 and cfg.arm_swing_phase_offset == 0.15
    assert (cfg.max_vel_x, cfg.max_vel_y, cfg.max_yaw_rate) == (1.0, 0.6, 1.0)


def test_pin_rho_default():
    from t1_nmpc.wb.config import make_wb_config
    assert make_wb_config().pin_rho == 1.0


def test_vdot_s_input_weight_regularized():
    from t1_nmpc.wb.config import make_wb_config
    assert make_wb_config().R[39] > 0.0      # was 0 -> singular GN Hessian on range(P) under lm=0
