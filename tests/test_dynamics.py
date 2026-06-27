import numpy as np
import pinocchio as pin
import aligator
from t1_nmpc.robot.config import make_config
from t1_nmpc.robot.model import load_model, nominal_x
from t1_nmpc.wb.dynamics import WBDynamics


def _setup():
    cfg = make_config(); rm = load_model(cfg)
    return cfg, rm, WBDynamics(rm, cfg)


def test_rnea_jacobian_matches_finite_diff():
    cfg, rm, dyn = _setup()
    val, Jx, Ju = dyn.rnea_funcs(base_only=True)
    rng = np.random.default_rng(0)
    q = pin.integrate(rm.model, pin.neutral(rm.model), rng.standard_normal(33) * 0.1)
    v = rng.standard_normal(33) * 0.1
    x = np.concatenate([q, v]); u = rng.standard_normal(45) * 0.1
    z = np.zeros(66)
    r0 = np.asarray(val(x, u, z)).flatten()
    J = np.asarray(Jx(x, u, z))
    # central finite diff on the manifold tangent. (Central, not forward: the base-wrench
    # residual has O(100) second derivatives in the orientation tangent, so forward diff at
    # eps=1e-6 carries ~1.7e-4 truncation error and cannot resolve the analytic Jacobian to
    # 1e-4. Central diff validates the exact same Jx to ~3.8e-8. Tolerance unchanged.)
    eps = 1e-6; Jfd = np.zeros((6, 66))
    for i in range(66):
        dp = np.zeros(66); dp[i] = eps
        dm = np.zeros(66); dm[i] = -eps
        rp = np.asarray(val(x, u, dp)).flatten()
        rm_ = np.asarray(val(x, u, dm)).flatten()
        Jfd[:, i] = (rp - rm_) / (2 * eps)
    assert np.max(np.abs(J - Jfd)) < 1e-4
    # Ju vs finite diff
    Jfu = np.zeros((6, 45))
    for i in range(45):
        du = u.copy(); du[i] += eps
        Jfu[:, i] = (np.asarray(val(x, du, z)).flatten() - r0) / eps
    assert np.max(np.abs(np.asarray(Ju(x, u, z)) - Jfu)) < 1e-4


def test_rnea_base_zero_at_gravity_comp_stand():
    cfg, rm, dyn = _setup()
    val, _, _ = dyn.rnea_funcs(base_only=True)
    x = nominal_x(cfg, rm.model)
    fz = rm.mass * 9.81 / 2.0
    W = np.array([0, 0, fz, 0, 0, 0])               # per foot, sole-frame vertical
    u = np.concatenate([np.zeros(33), W, W])        # a=0, both feet support
    r = np.asarray(val(x, u, np.zeros(66))).flatten()
    assert np.max(np.abs(r)) < 5.0                  # base wrench residual small at gravity comp


def test_double_integrator_ode_forward():
    cfg, rm, dyn = _setup()
    ode = dyn.DoubleIntegratorODE(dyn.space, dyn.nu)
    data = ode.createData()
    x = nominal_x(cfg, rm.model); u = np.zeros(45); u[:33] = 1.0   # a = 1
    ode.forward(x, u, data)
    np.testing.assert_allclose(data.xdot[:33], x[34:])             # qdot = v (=0)
    np.testing.assert_allclose(data.xdot[33:], np.ones(33))        # vdot = a
    ode.dForward(x, u, data)
    # data.Ju is (ndx=66, nu=45); d(vdot)/d(a) is the bottom-left 33x33 block (cols 0:33),
    # the wrench columns 33:45 are zero. Index the acceleration block explicitly.
    assert np.allclose(data.Ju[33:, :33], np.eye(33))
    assert np.allclose(data.Ju[33:, 33:], 0.0)


def test_ode_deepcopy_survives_integrator():
    cfg, rm, dyn = _setup()
    ode = dyn.DoubleIntegratorODE(dyn.space, dyn.nu)
    disc = aligator.dynamics.IntegratorEuler(ode, cfg.dt)   # deep-copies ode internally
    assert disc.timestep == cfg.dt
