import casadi as cs
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import (
    build_residual,
    build_residual_terminal,
    stage_cost_value,
    N_PARAM_WB,
    P_XREF,
    P_UREF,
)


def _yfun(cfg, model):
    x = cs.SX.sym("x", 68); u = cs.SX.sym("u", 40); p = cs.SX.sym("p", N_PARAM_WB)
    y, yref, W = build_residual(x, u, p, cfg, model)
    return cs.Function("y", [x, u, p], [y]), yref, W


def _stand_params(model):
    p = np.zeros(N_PARAM_WB)
    p[P_XREF] = model.nominal_state()
    u_ref = np.zeros(40)
    fz = model.total_mass() * 9.81 / 2.0
    u_ref[2] = fz; u_ref[8] = fz
    p[P_UREF] = u_ref
    return p, model.nominal_state(), u_ref


def test_residual_shapes_and_weights():
    cfg = make_wb_config(); m = WBModel(cfg)
    yfun, yref, W = _yfun(cfg, m)
    # 68 (x) + 40 (u) + 27 (joint-torque soft-cap) + 14 (swing-foot: 7 per foot) = 149 -- matches OCS2 structure
    assert yref.shape == (149,) and W.shape == (149,)
    assert np.all(W >= 0.0)                      # diagonal GN weight is PSD by construction
    assert np.allclose(W[:68], cfg.Q) and np.allclose(W[68:108], cfg.R)
    # joint-torque GN weight = scaling*weight (NOT 2x — OCS2's LS weight has no factor of 2). (audit 2026-06-25)
    assert np.allclose(W[108:135], cfg.jointtorque_scale * cfg.jointtorque_weight)


def test_cost_zero_at_reference_stand():
    cfg = make_wb_config(); m = WBModel(cfg)
    yfun, yref, W = _yfun(cfg, m)
    p, x_ref, u_ref = _stand_params(m)
    y = yfun(x_ref, u_ref, p)
    assert stage_cost_value(y, W) < 1e-6         # rx=ru=0 and torque below caps -> zero cost


def test_jointtorque_residual_activates_over_limit():
    cfg = make_wb_config(); m = WBModel(cfg)
    yfun, yref, W = _yfun(cfg, m)
    p, x_ref, u_ref = _stand_params(m)
    u_big = u_ref.copy()
    u_big[12:39] = 500.0                          # huge joint accel -> torque past URDF caps
    y = np.asarray(yfun(x_ref, u_big, p)).ravel()
    assert np.any(y[108:] > 0.0)                  # joint-torque residual block fires
    assert stage_cost_value(y, W) > 1.0


def test_terminal_residual():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", 68); p = cs.SX.sym("p", N_PARAM_WB)
    y_e, yref_e, W_e = build_residual_terminal(x, p, cfg)
    assert y_e.shape[0] == 68 and W_e.shape == (68,)
    assert np.allclose(W_e, cfg.terminal_scale * cfg.Q_final)
