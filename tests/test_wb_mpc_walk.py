# tests/test_wb_mpc_walk.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.gait_wb import SLOW_WALK
from t1_nmpc.wb.mpc_wb import build_node_params
from t1_nmpc.wb import cost_wb

cfg = make_wb_config(); model = WBModel(cfg)

# Uniform node times starting at t=0 (matches original scalar t=0.0 behaviour)
_uniform_nt_0 = 0.0 + np.arange(cfg.N + 1) * cfg.dt


def test_node_params_shape_and_layout():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, _uniform_nt_0, comm, SLOW_WALK, cfg, model)
    assert P.shape == (cfg.N + 1, cost_wb.N_PARAM_WB)


def test_node_params_contact_flags_follow_schedule():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, _uniform_nt_0, comm, SLOW_WALK, cfg, model)
    k = int(round(0.315 / cfg.dt))               # node in LF window -> [left_stance, right_swing]
    np.testing.assert_array_equal(P[k, cost_wb.P_CONTACT], [1.0, 0.0])
    assert P[k, cost_wb.P_IMPACT][1] < 1.0        # right foot swinging -> impact < 1
    assert P[k, cost_wb.P_SWINGZ][3] > 0.0        # right swing z above ground
    np.testing.assert_allclose(P[k, cost_wb.P_XREF][33], 0.3, atol=1e-9)  # base-vel x = command


def test_stance_node_params_are_double_support():
    x0 = model.nominal_state()
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(x0, _uniform_nt_0, comm, SLOW_WALK, cfg, model)
    k = int(round(0.75 / cfg.dt))                 # STANCE window
    np.testing.assert_array_equal(P[k, cost_wb.P_CONTACT], [1.0, 1.0])


def test_build_node_params_fills_pdt_and_aligns_switch():
    from t1_nmpc.wb.grid_wb import event_aligned_grid
    from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_DT
    cfg = make_wb_config(); m = WBModel(cfg)
    t0 = 0.2
    nt = event_aligned_grid(t0, SLOW_WALK, cfg)
    comm = np.array([0.3, 0.0, cfg.nominal_base_height, 0.0])
    P = build_node_params(m.nominal_state(), nt, comm, SLOW_WALK, cfg, m)
    assert P.shape == (cfg.N + 1, N_PARAM_WB)
    # P_DT column == interval lengths
    np.testing.assert_allclose(P[:cfg.N, P_DT], np.diff(nt), atol=1e-12)
    # a switch lands on a node (grid invariant carried through)
    for s in SLOW_WALK.switch_times_in(t0, t0 + cfg.N * cfg.dt):
        assert np.min(np.abs(nt - s)) < 1e-9
