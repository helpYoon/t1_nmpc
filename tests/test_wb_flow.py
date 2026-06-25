import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel


def test_flow_passthrough_and_inputs():
    cfg = make_wb_config()
    m = WBModel(cfg)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(68)
    u = rng.standard_normal(40)
    xd = m.flow(x, u)
    assert xd.shape == (68,)
    assert np.allclose(xd[0:33], x[33:66])      # dq/dt = v exactly (euler base + revolute)
    assert np.allclose(xd[39:66], u[12:39])     # d v_joints = qdd_joints
    assert np.isclose(xd[66], x[67])            # ds = v_s
    assert np.isclose(xd[67], u[39])            # d v_s = vddot_s


def test_flow_vdot_base_matches_block_inversion_numpy():
    cfg = make_wb_config()
    m = WBModel(cfg)
    rng = np.random.default_rng(2)
    x = m.nominal_state()
    x[33:66] = 0.1 * rng.standard_normal(33)
    u = np.zeros(40)
    u[0:12] = rng.standard_normal(12)
    u[12:39] = 0.05 * rng.standard_normal(27)
    xd = m.flow(x, u)
    # reference block-diagonal RBD inversion, recomputed in numpy
    q = x[0:33]; v = x[33:66]
    M = m.M_numeric_pin(q); nle = m.nle_numeric_pin(q, v)
    Jl = m.Jl(q); Jr = m.Jr(q)
    tau_ext = Jl.T @ u[0:6] + Jr.T @ u[6:12]
    inter = tau_ext[0:6] - nle[0:6] - M[0:6, 6:] @ u[12:39]
    vdot_lin = np.linalg.solve(M[0:3, 0:3], inter[0:3])
    vdot_ang = np.linalg.solve(M[3:6, 3:6], inter[3:6])
    assert np.allclose(xd[33:39], np.concatenate([vdot_lin, vdot_ang]), atol=1e-6)


def test_flow_equilibrium_vertical_balance():
    cfg = make_wb_config()
    m = WBModel(cfg)
    x = m.nominal_state()
    u = np.zeros(40)
    fz = m.total_mass() * 9.81 / 2.0
    u[2] = fz  # left foot fz
    u[8] = fz  # right foot fz
    xd = m.flow(x, u)
    assert abs(xd[35]) < 1e-4  # vertical base accel ~ 0 under weight-comp
