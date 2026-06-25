# tests/test_wb_config_walk.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb import cost_wb


def test_walking_config_fields():
    cfg = make_wb_config()
    assert cfg.foot_linacc_err_gain_z == 1.0
    np.testing.assert_array_equal(cfg.swingfoot_cost_weights, [1e4, 1e4, 5.0, 5.0, 2.0, 2.0, 2.0])
    assert cfg.arm_swing_amplitude == 0.15 and cfg.arm_swing_phase_offset == 0.15
    assert (cfg.max_vel_x, cfg.max_vel_y, cfg.max_yaw_rate) == (1.0, 0.6, 1.0)


def test_param_layout_grown_and_contiguous():
    assert cost_wb.N_PARAM_WB == 119                  # D4 appended P_DT -> 118 + 1
    assert cost_wb.P_XREF == slice(0, 68) and cost_wb.P_UREF == slice(68, 108)
    assert cost_wb.P_CONTACT == slice(108, 110)
    assert cost_wb.P_SWINGZ == slice(110, 116)
    assert cost_wb.P_IMPACT == slice(116, 118)
    assert cost_wb.P_DT == 118                        # D4: per-stage dt_k appended AFTER all existing slots
