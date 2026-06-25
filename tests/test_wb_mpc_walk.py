# tests/test_wb_mpc_walk.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.mpc_wb import build_node_params
from t1_nmpc.wb import cost_wb

cfg = make_wb_config(); model = WBModel(cfg)


def test_node_params_shape_and_layout():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, 0.0, comm, SLOW_WALK, cfg, model)
    assert P.shape == (cfg.N + 1, cost_wb.N_PARAM_WB)


def test_node_params_contact_flags_follow_schedule():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, 0.0, comm, SLOW_WALK, cfg, model)
    k = int(round(0.315 / cfg.dt))               # node in LF window -> [left_stance, right_swing]
    np.testing.assert_array_equal(P[k, cost_wb.P_CONTACT], [1.0, 0.0])
    assert P[k, cost_wb.P_IMPACT][1] < 1.0        # right foot swinging -> impact < 1
    assert P[k, cost_wb.P_SWINGZ][3] > 0.0        # right swing z above ground
    np.testing.assert_allclose(P[k, cost_wb.P_XREF][33], 0.3, atol=1e-9)  # base-vel x = command


def test_stance_node_params_are_double_support():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, 0.0, comm, SLOW_WALK, cfg, model)
    k = int(round(0.75 / cfg.dt))                 # STANCE window
    np.testing.assert_array_equal(P[k, cost_wb.P_CONTACT], [1.0, 1.0])
