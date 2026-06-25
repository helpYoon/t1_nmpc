# tests/test_wb_cost_walk.py
import numpy as np
import casadi as cs
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import build_residual, N_PARAM_WB, P_CONTACT, P_IMPACT

cfg = make_wb_config(); model = WBModel(cfg)
x = cs.SX.sym("x", 68); u = cs.SX.sym("u", 40); p = cs.SX.sym("p", N_PARAM_WB)
y, yref, W = build_residual(x, u, p, cfg, model)
Fy = cs.Function("y", [x, u, p], [y])


def _p(left_stance, right_stance):
    pv = np.zeros(N_PARAM_WB)
    pv[P_CONTACT] = [left_stance, right_stance]; pv[P_IMPACT] = [1.0, 1.0]
    return pv


def test_residual_size_walking():
    # walking residual = 68 (x) + 40 (u) + n_finite_torque + 14 (swing-foot 7/foot) -- matches OCS2 structure
    n_fin = int(np.isfinite(np.asarray(cfg.torque_limit)).sum())
    assert y.shape[0] == 68 + 40 + n_fin + 14
    assert len(W) == y.shape[0]


def test_swing_foot_cost_active_on_moving_stance_foot():
    # Gate is impact-proximity ONLY (not (1-contact)); OCS2 keeps the foot task cost on the STANCE foot
    # (EndEffectorDynamicsFootCost.cpp:123). With both feet in stance but the base translating, the feet
    # move, so the swing-foot linvel rows must be NONZERO -- the old (1-contact) gate forced them to 0.
    xn = model.nominal_state(); xn[33] = 0.3                  # base world-vx -> stance feet move with base
    yv = np.array(Fy(xn, np.zeros(40), _p(1, 1))).ravel()
    sf = yv[135:149]                                          # the 14 swing-foot rows (7 per foot)
    assert np.any(np.abs(sf) > 1e-3)


def test_swing_cost_zero_in_double_stance():
    xn = model.nominal_state(); uv = np.zeros(40)
    yv = np.array(Fy(xn, uv, _p(1, 1))).ravel()
    assert np.allclose(yv[-14:], 0.0)            # at nominal the feet are flat & still -> swing residual = 0


def test_swing_orientation_error_zero_at_flat_foot():
    xn = model.nominal_state(); uv = np.zeros(40)
    yv = np.array(Fy(xn, uv, _p(1, 0))).ravel()  # right foot swinging
    # right-foot swing block = last 7 rows; ori_x, ori_y (rows -7,-6) ~0 at flat foot
    assert abs(yv[-7]) < 1e-6 and abs(yv[-6]) < 1e-6
