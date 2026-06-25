"""M0 double-stance constraint tests, re-targeted to the new 36-row per-foot layout.

Per-foot block (18 rows): [ZA(6), ZW(6), SwingZ(1), Fric(1), CoP(4)]
foot L rows 0-17, foot R rows 18-35.

At double-stance (contact=[1,1]) all swing rows (ZW 6-11, SwingZ 12 per foot) are zero,
so the substantive M0 assertions reduce to:
  ZeroAccel   : rows 0-5 (L) and 18-23 (R)  — equality at zero
  Friction    : rows 13 (L) and 31 (R)       — > 0
  CoP         : rows 14-17 (L) and 32-35 (R) — > 0
"""
import casadi as cs
import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.constraints_wb import build_con_h
from t1_nmpc.wb import cost_wb


def _confun(cfg, model):
    x = cs.SX.sym("x", 68); u = cs.SX.sym("u", 40)
    p = cs.SX.sym("p", cost_wb.N_PARAM_WB)
    con_h, lh, uh = build_con_h(x, u, p, cfg, model)
    return cs.Function("h", [x, u, p], [con_h]), lh, uh


def _p_double_stance(cfg):
    """Parameter vector: double stance, no swing-Z ref, no impact proximity."""
    pv = np.zeros(cost_wb.N_PARAM_WB)
    pv[cost_wb.P_CONTACT] = [1.0, 1.0]   # both stance
    return pv


def _weightcomp(model):
    u = np.zeros(40)
    fz = model.total_mass() * 9.81 / 2.0
    u[2] = fz; u[8] = fz
    return u


def test_con_h_rowcounts_and_bounds():
    cfg = make_wb_config(); m = WBModel(cfg)
    hfun, lh, uh = _confun(cfg, m)
    pv = _p_double_stance(cfg)
    con = np.asarray(hfun(m.nominal_state(), _weightcomp(m), pv)).ravel()
    assert con.shape == (36,) and lh.shape == (36,) and uh.shape == (36,)
    assert np.all(lh == 0.0)
    # per-foot block: ZA(6)+ZW(6)+SwingZ(1) equality (uh=0), Fric(1)+CoP(4) one-sided (uh=big)
    # rows with uh==0: ZA 0-5, ZW 6-11, SwingZ 12 for each foot (repeat at +18)
    eq_rows = list(range(0, 13)) + list(range(18, 31))
    ineq_rows = [13, *range(14, 18), 31, *range(32, 36)]
    assert np.all(uh[eq_rows] == 0.0) and np.all(uh[ineq_rows] > 1e8)


def test_friction_and_cop_satisfied_at_weightcomp_stand():
    cfg = make_wb_config(); m = WBModel(cfg)
    hfun, _, _ = _confun(cfg, m)
    pv = _p_double_stance(cfg)
    con = np.asarray(hfun(m.nominal_state(), _weightcomp(m), pv)).ravel()
    assert np.all(np.isfinite(con))
    assert con[13] > 0.0    # friction margin > 0 (left)
    assert con[31] > 0.0    # friction margin > 0 (right)
    assert np.all(con[14:18] > 0.0)    # CoP inside the foot rectangle (left)
    assert np.all(con[32:36] > 0.0)    # CoP inside the foot rectangle (right)


def test_zeroaccel_small_at_weightcomp_stand():
    cfg = make_wb_config(); m = WBModel(cfg)
    hfun, _, _ = _confun(cfg, m)
    pv = _p_double_stance(cfg)
    con = np.asarray(hfun(m.nominal_state(), _weightcomp(m), pv)).ravel()
    # foot planted + nearly balanced -> ZeroAccel residual small (not a perfect equilibrium wrench)
    za_L = con[0:6]
    za_R = con[18:24]
    assert np.max(np.abs(za_L)) < 1.0
    assert np.max(np.abs(za_R)) < 1.0
