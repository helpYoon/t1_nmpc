import numpy as np

from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel


def test_joint_torque_matches_numpy_inverse_dynamics():
    cfg = make_wb_config()
    m = WBModel(cfg)
    rng = np.random.default_rng(3)
    x = m.nominal_state()
    x[33:66] = 0.1 * rng.standard_normal(33)
    u = np.zeros(40)
    u[0:12] = rng.standard_normal(12)
    u[12:39] = 0.05 * rng.standard_normal(27)
    tau = m.joint_torque(x, u)
    assert tau.shape == (27,)
    # gold-standard numpy reimplementation
    q = x[0:33]; v = x[33:66]
    vdot_base = m.flow(x, u)[33:39]
    accel = np.concatenate([vdot_base, u[12:39]])
    M = m.M_numeric_pin(q); nle = m.nle_numeric_pin(q, v)
    tau_full = M @ accel + nle - (m.Jl(q).T @ u[0:6] + m.Jr(q).T @ u[6:12])
    tau_ref = tau_full[6:] + np.asarray(cfg.viscous_damping) * v[6:]
    assert np.allclose(tau, tau_ref, atol=1e-6)


def test_joint_torque_stand_within_limits():
    cfg = make_wb_config()
    m = WBModel(cfg)
    x = m.nominal_state()
    u = np.zeros(40)
    fz = m.total_mass() * 9.81 / 2.0
    u[2] = fz
    u[8] = fz
    tau = m.joint_torque(x, u)
    assert np.all(np.isfinite(tau))
    assert np.all(np.abs(tau) < cfg.torque_limit)   # gravity-holding stand within URDF caps
    assert np.abs(tau).max() < 5.0                  # near-nominal stand needs only small holding torque
