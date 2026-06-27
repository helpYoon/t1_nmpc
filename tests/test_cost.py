# tests/test_cost.py
import numpy as np
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics
from t1_nmpc.wb import cost as K


def _setup():
    cfg = make_config(); rm = load_model(cfg); dyn = WBDynamics(rm, cfg)
    return cfg, rm, dyn


def test_state_tracking_zero_at_nominal():
    cfg, rm, dyn = _setup()
    x_des = nominal_x(cfg, rm.model)
    c = K.state_tracking(dyn.space, cfg.nu, x_des, cfg.Q_diag)
    data = c.createData()
    c.evaluate(x_des, np.zeros(45), data)
    assert abs(data.value) < 1e-12


def test_input_reg_penalizes_off_target():
    cfg, rm, dyn = _setup()
    u_des = K.gravity_comp_u_des(rm, n_support=2)
    c = K.input_reg(dyn.space, cfg.nu, u_des, cfg.R_diag)
    data = c.createData()
    x = nominal_x(cfg, rm.model)
    c.evaluate(x, u_des, data); assert abs(data.value) < 1e-12
    u_off = u_des.copy(); u_off[0] += 10.0
    c.evaluate(x, u_off, data); assert data.value > 0.0


def test_gravity_comp_supports_split():
    cfg, rm, dyn = _setup()
    u = K.gravity_comp_u_des(rm, n_support=2)
    assert u.shape == (45,)
    fz = rm.mass * 9.81 / 2.0
    np.testing.assert_allclose(u[33 + 2], fz)     # left foot f_z
    np.testing.assert_allclose(u[39 + 2], fz)     # right foot f_z
    assert np.allclose(u[:33], 0.0)


def test_arm_to_nominal_weights_arms_more():
    cfg, rm, dyn = _setup()
    x_des = nominal_x(cfg, rm.model)
    c = K.arm_to_nominal(dyn.space, cfg.nu, x_des, cfg)
    data = c.createData()
    # perturb an arm joint -> nonzero cost
    x = x_des.copy(); x[7] += 0.2     # first arm joint (q index 7)
    c.evaluate(x, np.zeros(45), data)
    assert data.value > 0.0
