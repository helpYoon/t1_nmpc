# tests/test_wb_projection.py
import numpy as np
from t1_nmpc.wb.config_wb import make_wb_config
from t1_nmpc.wb.model_wb import WBModel
from t1_nmpc.wb.cost_wb import N_PARAM_WB, P_CONTACT, P_DT
from t1_nmpc.wb import projection_wb


def _p_vec(cfg, lf, rf):
    p = np.zeros(N_PARAM_WB)
    p[P_CONTACT] = [float(lf), float(rf)]
    p[P_DT] = cfg.dt
    return p


def test_folded_residual_has_zerowrench_rows():
    cfg = make_wb_config(); m = WBModel(cfg)
    r_fun, D_fun, C_fun = projection_wb.build_projector_funcs(cfg, m)
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu)
    # single support: left stance, right swing -> right foot wrench u[6:12] must appear in r
    p = _p_vec(cfg, lf=1, rf=0)
    r = np.asarray(r_fun(x0, u0, p)).ravel()
    D = np.asarray(D_fun(x0, u0, p))
    assert r.shape == (26,) and D.shape == (26, 40)
    # rows 14..19 = left ZeroWrench (left is STANCE -> gated to 0); rows 20..25 = right ZeroWrench (SWING -> active = u[6:12])
    assert np.allclose(D[14:20, :], 0.0)                     # left stance -> no ZeroWrench
    assert np.allclose(D[20:26, 6:12], np.eye(6))            # right swing -> identity on its wrench
    assert np.allclose(np.delete(D[20:26, :], np.s_[6:12], axis=1), 0.0)


def test_projector_identities_single_support():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    x0 = m.nominal_state(); u0 = np.zeros(cfg.nu)
    u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0               # gravity-ish stance wrench
    p = _p_vec(cfg, lf=1, rf=0)
    P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
    assert P.shape == (40, 40) and Q.shape == (40, 68) and u_p.shape == (40,)
    assert np.allclose(P, P.T, atol=1e-10)                    # orthogonal projector: symmetric
    assert np.allclose(P @ P, P, atol=1e-10)                  # idempotent
    D = np.asarray(funcs[1](x0, u0, p))
    assert np.linalg.norm(D @ P) < 1e-9                       # range(P) = ker(D)
    # for an arbitrary raw u, u_phys satisfies the (active) linearized equality and zeros the swing wrench
    rng = np.random.default_rng(0); u = rng.standard_normal(40)
    u_phys = P @ u + Q @ x0 + u_p
    r_fun = funcs[0]
    # linearized residual at (x0,u_phys): r0 + D(u_phys-u0) (C term drops since x=x0)
    r0 = np.asarray(r_fun(x0, u0, p)).ravel()
    lin = r0 + D @ (u_phys - u0)
    assert np.linalg.norm(lin) < 1e-9
    assert np.allclose(u_phys[6:12], 0.0, atol=1e-10)         # right (swing) wrench == 0
