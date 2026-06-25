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


def _null_space(D):
    """Orthonormal basis of ker(D) via SVD (numpy-only; matches scipy.linalg.null_space)."""
    _u, s, vh = np.linalg.svd(D)
    tol = (np.amax(s) if s.size else 0.0) * np.finfo(s.dtype).eps * max(D.shape)
    return vh[np.sum(s > tol):].conj().T


def _solve_reduced_qp(D, r0, R, u0, u_ref):
    """OCS2-style reduced QP at dx=0: u_phys = u0 - D⁺ r0 + Z v, minimize 0.5 (u_phys-u_ref)' R (u_phys-u_ref)."""
    Dp = np.linalg.pinv(D)
    Z = _null_space(D)                                           # nu × (nu - rank(D))
    base = u0 - Dp @ r0                                          # particular solution
    a = base - u_ref
    Rz = Z.T @ R @ Z
    v = -np.linalg.solve(Rz, Z.T @ R @ a)
    return base + Z @ v


def _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho):
    """Full-nu QP at dx=0: min over raw u of 0.5(u_phys-u_ref)'R(u_phys-u_ref) + 0.5 rho ||(I-P)(u-u_ref)||^2."""
    c = Q @ x0 + u_p                                             # u_phys = P u + c
    ImP = np.eye(P.shape[0]) - P
    H = P.T @ R @ P + rho * ImP                                  # (I-P) sym-idempotent -> (I-P)'(I-P)=(I-P)
    b = -P.T @ R @ (c - u_ref) + rho * ImP @ u_ref
    u = np.linalg.solve(H, b)
    return P @ u + c


def test_full_nu_projector_matches_ocs2_reduced_qp():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    R = np.diag(np.asarray(cfg.R, dtype=np.float64))
    rng = np.random.default_rng(1); u_ref = rng.standard_normal(40)
    for lf, rf in [(1, 1), (1, 0), (0, 1)]:                      # double + each single support
        x0 = m.nominal_state(); u0 = np.zeros(40)
        u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
        p = _p_vec(cfg, lf, rf)
        D = np.asarray(funcs[1](x0, u0, p)); r0 = np.asarray(funcs[0](x0, u0, p)).ravel()
        P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
        u_red = _solve_reduced_qp(D, r0, R, u0, u_ref)
        u_full = _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho=1.0)
        # tol 5e-8: u_phys is ρ-independent analytically; the residual is float64 round-off in the
        # cond(H)=ρ/R_min ≈ 1e6 pin direction (a naive-solve artifact, not a u_phys property).
        assert np.linalg.norm(u_full - u_red) <= 5e-8, f"mode {lf,rf}: {np.linalg.norm(u_full-u_red)}"


def test_pin_rho_does_not_bias_u_phys():
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    R = np.diag(np.asarray(cfg.R, dtype=np.float64))
    rng = np.random.default_rng(2); u_ref = rng.standard_normal(40)
    x0 = m.nominal_state(); u0 = np.zeros(40); u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
    p = _p_vec(cfg, lf=1, rf=0)
    P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
    base = _solve_full_qp(P, Q, u_p, x0, R, u_ref, rho=1.0)
    # sweep capped at ρ<=1.0: larger ρ only worsens the naive np.linalg.solve conditioning (cond=ρ/R_min);
    # u_phys is analytically ρ-independent, so invariance across these decades is the real check.
    for rho in (1e-3, 1e-2, 1e-1, 1.0):
        assert np.linalg.norm(_solve_full_qp(P, Q, u_p, x0, R, u_ref, rho) - base) <= 5e-8


def test_full_nu_matches_ocs2_reduced_with_dx():
    """Equivalence at x != x0 (dx != 0) -- exercises Q = -D⁺C (untested when dx=0)."""
    cfg = make_wb_config(); m = WBModel(cfg)
    funcs = projection_wb.build_projector_funcs(cfg, m)
    R = np.diag(np.asarray(cfg.R, dtype=np.float64))
    rng = np.random.default_rng(3); u_ref = rng.standard_normal(40)
    x0 = m.nominal_state(); u0 = np.zeros(40); u0[2] = u0[8] = m.total_mass() * 9.81 / 2.0
    dx = np.zeros(cfg.nx); dx[3] = 0.05; dx[20] = 0.1            # perturb base-yaw + a joint
    x = x0 + dx
    for lf, rf in [(1, 1), (1, 0)]:
        p = _p_vec(cfg, lf, rf)
        D = np.asarray(funcs[1](x0, u0, p)); r0 = np.asarray(funcs[0](x0, u0, p)).ravel()
        C = np.asarray(funcs[2](x0, u0, p))
        P, Q, u_p = projection_wb.compute_projector(x0, u0, p, funcs, cfg)
        Dp = np.linalg.pinv(D); Z = _null_space(D)
        base = u0 - Dp @ (r0 + C @ dx); a = base - u_ref        # OCS2 reduced offset at dx
        Rz = Z.T @ R @ Z; v = -np.linalg.solve(Rz, Z.T @ R @ a)
        u_red = base + Z @ v
        c = Q @ x + u_p; ImP = np.eye(40) - P                   # full-nu at x = x0 + dx
        H = P.T @ R @ P + 1.0 * ImP; b = -P.T @ R @ (c - u_ref) + 1.0 * ImP @ u_ref
        u = np.linalg.solve(H, b); u_full = P @ u + c
        assert np.linalg.norm(u_full - u_red) <= 5e-8, f"dx mode {lf,rf}: {np.linalg.norm(u_full-u_red)}"


# ---------------------------------------------------------------------------
# Task 5: ker(P)-confined pin block in build_cost_conl
# ---------------------------------------------------------------------------
import casadi as cs
from t1_nmpc.wb import cost_wb


def test_cost_appends_kerp_pin_block():
    cfg = make_wb_config(); m = WBModel(cfg)
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu); p = cs.SX.sym("p", cost_wb.N_PARAM_WB)
    Pm = cs.reshape(p[cost_wb.P_PROJ_P], (cfg.nu, cfg.nu))
    y0, _, _, _ = cost_wb.build_cost_conl(x, u, p, cfg, m)                       # no pin
    y1, _, psi1, _ = cost_wb.build_cost_conl(x, u, p, cfg, m, u_raw=u, P_mat=Pm)  # with pin
    assert y1.shape[0] == y0.shape[0] + cfg.nu                                   # +40 pin rows


# ---------------------------------------------------------------------------
# Task 6: OCP wiring — u_phys into dynamics+cost, contact con_h removed
# ---------------------------------------------------------------------------
from t1_nmpc.wb.ocp_wb import make_ocp


def test_ocp_has_projector_params_and_no_contact_con_h():
    cfg = make_wb_config()
    ocp, _ = make_ocp(cfg)
    assert ocp.parameter_values.shape[0] == cost_wb.N_PARAM_WB == 4479
    assert ocp.solver_options.levenberg_marquardt == 0.0
    assert ocp.solver_options.regularize_method == "NO_REGULARIZE"
    # con_h contact-equality rows removed (no nonlinear constraint expr left)
    _con_h = getattr(ocp.model, "con_h_expr", None)
    assert _con_h is None or (hasattr(_con_h, "shape") and _con_h.shape[0] == 0) or _con_h == []
    # ZeroWrench input box removed; joint-position box (idxbx) kept
    assert getattr(ocp.constraints, "idxbu", np.array([])).size == 0
    assert ocp.constraints.idxbx.size == 27
