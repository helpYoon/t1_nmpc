"""Walk-gated constraint tests for the new 36-row build_con_h.

Per-foot block order: [ZeroAccel(6), ZeroWrench(6), SwingZ(1), Friction(1), CoP(4)]
foot L rows 0-17, foot R rows 18-35.
"""
import numpy as np
import casadi as cs
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.constraints_wb import build_con_h
from t1_nmpc.wb import cost_wb

cfg = make_wb_config(); model = WBModel(cfg)
x = cs.SX.sym("x", 68); u = cs.SX.sym("u", 40); p = cs.SX.sym("p", cost_wb.N_PARAM_WB)
con_h, lh, uh = build_con_h(x, u, p, cfg, model)
F = cs.Function("h", [x, u, p], [con_h])


def _p(left_stance, right_stance, uvec=None):
    pv = np.zeros(cost_wb.N_PARAM_WB)
    pv[cost_wb.P_CONTACT] = [left_stance, right_stance]
    pv[cost_wb.P_IMPACT] = [1.0, 1.0]
    return pv


def test_row_count_is_36():
    assert con_h.shape[0] == 36 and len(lh) == 36 and len(uh) == 36


def test_double_stance_swing_rows_are_zero():
    xn = model.nominal_state(); uv = np.zeros(40); uv[2] = uv[8] = model.total_mass() * 9.81 / 2
    h = np.array(F(xn, uv, _p(1, 1))).ravel()
    # foot L swing block = rows 6..12 (ZW 6-11, SwingZ 12); foot R = rows 24..30
    assert np.allclose(h[6:13], 0.0) and np.allclose(h[24:31], 0.0)


def test_single_stance_gates_correctly():
    xn = model.nominal_state()
    uv = np.zeros(40); uv[2] = model.total_mass() * 9.81   # full weight on left (stance)
    uv[6:12] = [1, 2, 3, 4, 5, 6]                           # nonzero right (swing) wrench
    h = np.array(F(xn, uv, _p(1, 0))).ravel()               # left stance, right swing
    # right stance rows (ZA 18-23, Fric 31, CoP 32-35) gated OFF -> 0
    assert np.allclose(h[18:24], 0.0) and h[31] == 0.0 and np.allclose(h[32:36], 0.0)
    # right ZeroWrench rows (24-29) == the right wrench (gate (1-0)=1)
    np.testing.assert_allclose(h[24:30], [1, 2, 3, 4, 5, 6])
    # left stance ZeroAccel rows (0-5) are ACTIVE (left foot block present); left swing rows zero
    assert np.allclose(h[6:13], 0.0)
