import copy
import numpy as np
import pinocchio as pin
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb import constraint as C


def _setup():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn


def test_rnea_residual_deepcopy_shares_funcs():
    cfg, rm, dyn = _setup()
    r = C.RneaBaseResidual(cfg.ndx, cfg.nu, dyn.rnea_funcs(base_only=True))
    r2 = copy.deepcopy(r)
    assert r2._funcs is r._funcs          # shared, not recompiled
    assert r2.nr == 6


def test_rnea_residual_evaluate_matches_factory():
    cfg, rm, dyn = _setup()
    funcs = dyn.rnea_funcs(base_only=True)
    r = C.RneaBaseResidual(cfg.ndx, cfg.nu, funcs)
    data = r.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45)
    r.evaluate(x, u, data)
    expect = np.asarray(funcs[0](x, u, np.zeros(66))).flatten()
    np.testing.assert_allclose(data.value, expect, atol=1e-9)
    r.computeJacobians(x, u, data)
    assert data.Jx.shape == (6, 66) and data.Ju.shape == (6, 45)


def test_wrench_cone_unilateral_and_friction():
    cfg, rm, dyn = _setup()
    wc = C.WrenchConeResidual(cfg.ndx, cfg.nu, foot_index=0, mu=cfg.friction_mu,
                              X=cfg.half_len, Y=cfg.half_width)
    data = wc.createData()
    x = nominal_x(cfg, rm.model)
    # vertical-only force inside cone -> all rows <= 0 (feasible)
    u = np.zeros(45); u[33 + 2] = 100.0
    wc.evaluate(x, u, data)
    assert np.all(data.value <= 1e-9)
    # large lateral force -> friction row > 0 (violated)
    u2 = np.zeros(45); u2[33 + 0] = 100.0; u2[33 + 2] = 10.0
    wc.evaluate(x, u2, data)
    assert np.any(data.value > 0.0)
    # finite-diff Ju check
    wc.computeJacobians(x, u2, data)
    eps = 1e-6; r0 = np.asarray(data.value).copy()
    Jfd = np.zeros_like(data.Ju)
    for i in range(45):
        du = u2.copy(); du[i] += eps
        d2 = wc.createData(); wc.evaluate(x, du, d2)
        Jfd[:, i] = (np.asarray(d2.value) - r0) / eps
    assert np.max(np.abs(np.asarray(data.Ju) - Jfd)) < 1e-3


def test_swing_wrench_selects_foot():
    cfg, rm, dyn = _setup()
    sw = C.SwingWrenchResidual(cfg.ndx, cfg.nu, foot_index=1)
    data = sw.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45); u[39:45] = [1, 2, 3, 4, 5, 6]
    sw.evaluate(x, u, data)
    np.testing.assert_allclose(data.value, [1, 2, 3, 4, 5, 6])


def test_contact_velocity_residual_zero_at_rest():
    cfg, rm, dyn = _setup()
    res = C.contact_velocity_residual(rm, cfg.ndx, cfg.nu, foot_index=0)
    data = res.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45)
    res.evaluate(x, u, data)
    assert np.max(np.abs(data.value)) < 1e-9     # v=0 -> foot velocity 0


def test_swing_z_setreference():
    cfg, rm, dyn = _setup()
    sliced, base = C.swing_z_residual(rm, cfg.ndx, cfg.nu, foot_index=0)
    assert sliced.nr == 1
    # vref is the non-deprecated reference setter (setReference is deprecated in 0.19.0)
    base.vref = pin.Motion(np.array([0, 0, 0.05, 0, 0, 0.0]))
    assert abs(base.getReference().linear[2] - 0.05) < 1e-12
