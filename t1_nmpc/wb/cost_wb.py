"""WB stage cost as a NONLINEAR_LS residual (acados forms the GN Hessian at runtime).

This is the OCS2/CppAD-equivalent: instead of handing acados a symbolic 108x108
Gauss-Newton Hessian (EXTERNAL + custom_hess, which CasADi codegens as one giant C
function that -O2 chokes on), we give the RESIDUAL y(x,u) and weight W. acados assembles
H = J^T W J numerically per node from the (small) residual Jacobian — no symbolic Hessian,
no compile blowup. Valid here because the WB cost is PURE least-squares (the friction/CoP/
joint-limit barriers live in con_h, not the cost).

cost = 0.5 (y - yref)^T diag(W) (y - yref), yref = 0:
  y = [ x - x_ref(p),  u - u_ref(p),  g_jt(x,u) ]
  W = [ Q(68),         R(40),         2*scale*w (per finite-limit joint) ]
  g_jt_i = ReLU(tau_i^2 - lim_i^2) / lim_i^2   (the quartic soft-cap as an LS residual)
"""
from __future__ import annotations

import casadi as cs
import numpy as np

# per-node param layout. M0: reference state + input (folded into the residual).
P_XREF = slice(0, 68)
P_UREF = slice(68, 108)
# M1 walking: per-node gait params.
P_CONTACT = slice(108, 110)   # [left_stance, right_stance] in {0,1}
P_SWINGZ = slice(110, 116)    # [zL, zdotL, zddotL, zR, zdotR, zddotR]  (swing-foot z spline)
P_IMPACT = slice(116, 118)    # [impact_L, impact_R]  (impact-proximity scaler per foot)
# The contact EQUALITIES (ZeroAccel/SwingZ) are NOT params: they are con_h rows on the RAW input, toggled
# active/inactive per node via the constraint bounds (constraints_wb). The reduced-basis projector
# (u_phys = P@u_raw + Q@x + u_p, P/Q/u_p as per-node params) is now used, with a ker(P)-confined pin
# 0.5·ρ·||(I−P)(u_raw−u_ref)||² appended to the cost to regularise the nullspace. (The 2026-06-24
# rejection was the ε-regularised square-projector variant that densified the RK4 Jacobian.)
N_PARAM_WB = 118
# D4 event-aligned grid: per-stage interval length dt_k (scalar); appended AFTER all existing slots.
P_DT = N_PARAM_WB            # 118
N_PARAM_WB = N_PARAM_WB + 1  # 119
# Reduced-basis projection (D1): per-node affine projector u_phys = P@u + Q@x + u_p, passed as params.
# Matrices stored COLUMN-MAJOR (order='F'); ocp_wb reconstructs via cs.reshape (also column-major). nu=40, nx=68.
P_PROJ_P = slice(N_PARAM_WB, N_PARAM_WB + 40 * 40)              # 119:1719  (P, 40x40)
N_PARAM_WB = N_PARAM_WB + 40 * 40                               # 1719
P_PROJ_Q = slice(N_PARAM_WB, N_PARAM_WB + 40 * 68)             # 1719:4439 (Q, 40x68)
N_PARAM_WB = N_PARAM_WB + 40 * 68                              # 4439
P_PROJ_UP = slice(N_PARAM_WB, N_PARAM_WB + 40)                 # 4439:4479 (u_p, 40)
N_PARAM_WB = N_PARAM_WB + 40                                    # 4479


def _finite_idx(cfg):
    return np.where(np.isfinite(np.asarray(cfg.torque_limit, dtype=np.float64)))[0]


def _swing_foot_residual(x, p, cfg, model):
    """Per-foot swing-foot task-space residual (7 nonzero rows), gated by impact-proximity ONLY.
    errors = [ori_x, ori_y, linvel_x, linvel_y, angvel_x, angvel_y, angvel_z], refs all 0.
    quaternionDistanceToPlane(R[:,2],[0,0,1]) = -qc.vec(); for b=[0,0,1]: vec=[a1,-a0,0],
    w=1+a2, norm=sqrt(a0^2+a1^2+(1+a2)^2) -> ori_x=-a1/norm, ori_y=a0/norm.
    (EndEffectorDynamicsFootCost.cpp:114-152)"""
    q = x[0:33]; v = x[33:66]
    _v, qdd_j, _M, _nle, _te, vdot_base = model._dyn_terms(x, cs.SX.zeros(cfg.nu))
    a_full = cs.vertcat(vdot_base, qdd_j)
    rows = []
    for i in (0, 1):
        # OCS2 applies the foot task cost at EVERY node for BOTH feet, scaled ONLY by impact-proximity
        # (=1.0 on a stance foot, dipping to ~0.005 mid-swing); there is NO (1-contact) factor
        # (EndEffectorDynamicsFootCost.cpp:123; SwingTrajectoryPlanner.cpp:182-186). (audit 2026-06-25)
        gate = p[P_IMPACT][i]
        twist, _accel, _pose = model.foot_kin_fun[i](q, v, a_full)
        R = model.foot_R_fun[i](q)
        a0, a1, a2 = R[0, 2], R[1, 2], R[2, 2]            # foot z-axis in world
        norm = cs.sqrt(a0 ** 2 + a1 ** 2 + (1.0 + a2) ** 2)
        ori_x = -a1 / norm; ori_y = a0 / norm
        linvel_x, linvel_y = twist[0], twist[1]
        angvel_x, angvel_y, angvel_z = twist[3], twist[4], twist[5]
        errors = cs.vertcat(ori_x, ori_y, linvel_x, linvel_y, angvel_x, angvel_y, angvel_z)
        rows.append(gate * errors)
    return cs.vertcat(*rows)


def _relaxed_barrier(r, mu, delta):
    """OCS2 RelaxedBarrierPenalty: -mu*ln(r) for r>delta; a C^1 quadratic extension for r<=delta
    (so it stays finite when an SQP step drives the margin negative). Convex in r -> the acados CONL
    GGN outer Hessian is PSD. (RelaxedBarrierPenalty.h; matches OCS2 frictionForceConeSoftConstraint /
    contactMomentXYSoftConstraint.)
    NaN-guard: CasADi's AD of if_else evaluates BOTH branches, so a bare ln(r) makes the Hessian NaN
    once r<=0 (0*NaN=NaN). Feed the log a clamped fmax(r,delta) (>=delta>0); it equals r exactly on
    r>delta where the log branch is actually selected, so the value/derivative are unchanged."""
    r_pos = cs.fmax(r, delta)                                # >= delta > 0 -> ln never sees r<=0
    z = (r - 2.0 * delta) / delta
    quad = mu * (-float(np.log(delta)) + 0.5 * z ** 2 - 0.5)
    return cs.if_else(r > delta, -mu * cs.log(r_pos), quad)


def _piecewise_poly_barrier(h, mu, delta):
    """OCS2 PieceWisePolynomialBarrierPenalty (the FootCollisionConstraint penalty): 0 for h>=delta, a C^2
    polynomial that grows as the margin h drops below delta and goes negative -- finite everywhere,
    repulsive only near/under the bound (gentler than the log RelaxedBarrier). (PieceWisePolynomialBarrierPenalty.cpp)
      h<=0:       mu*(0.5 h^2 - delta h/2 + delta^2/6)
      0<h<delta:  mu*(-h^3/(6 delta) + 0.5 h^2 - delta h/2 + delta^2/6)
      h>=delta:   0
    No log/division-by-h -> AD-safe (both if_else branches finite)."""
    d = float(delta); m = float(mu)
    base = 0.5 * h ** 2 - d * h / 2.0 + d * d / 6.0
    return cs.if_else(h >= d, 0.0, m * (cs.if_else(h > 0.0, -h ** 3 / (6.0 * d), 0.0) + base))


def _foot_collision_residual(x, p, cfg, model):
    """16 leg-pair collision margins h = ||p_i - p_j|| - minDist (>=0 desired), faithful to t1_controller's
    FootCollisionConstraint (row order + minDists from FootCollisionConstraint.cpp:120-141). Keeps the two
    legs' collision spheres apart so the emergent swing foot can't collapse to the midline and step on the
    stance foot. Gated to SINGLE-SUPPORT: in double-stance (both contact) a big constant is added so the
    barrier contributes 0 (OCS2 isActive = !(lf && rf) -- avoids fighting the two stance-foot constraints)."""
    q = x[0:33]
    Pc = model.collision_pts_fun(q)                          # 3x10, columns per model_wb collision order
    f_l, l_p1, l_p2, f_r, r_p1, r_p2, ank_l, ank_r, k_l, k_r = [Pc[:, i] for i in range(10)]
    minF = 2.0 * cfg.foot_collision_radius
    minK = 2.0 * cfg.knee_collision_radius

    def dist(a, b):
        return cs.sqrt(cs.sumsqr(a - b) + 1e-9)             # eps: ||.|| has a NaN AD-gradient at coincident points

    rows = cs.vertcat(
        dist(l_p1, r_p1) - minF, dist(l_p1, r_p2) - minF, dist(l_p2, r_p1) - minF, dist(l_p2, r_p2) - minF,
        dist(f_l, r_p1) - minF, dist(f_l, r_p2) - minF, dist(f_r, l_p1) - minF, dist(f_r, l_p2) - minF,
        dist(f_l, f_r) - minF, dist(k_l, k_r) - minK,
        dist(f_l, ank_r) - minF, dist(l_p1, ank_r) - minF, dist(l_p2, ank_r) - minF,
        dist(f_r, ank_l) - minF, dist(r_p1, ank_l) - minF, dist(r_p2, ank_l) - minF,
    )
    both = p[P_CONTACT][0] * p[P_CONTACT][1]                 # 1 in double-stance -> add 10 -> barrier 0 (inactive)
    return rows + both * 10.0


def _contact_barrier_args(x, u, p, cfg, model):
    """Per-foot friction(1) + CoP(4) margins h>=0 for the interior-repulsive barrier (10 rows total).
    Moved here from con_h: OCS2 enforces friction/CoP as RelaxedBarrier SOFT inequalities, not hard.
    Gated by the stance flag: when swinging (flag=0) the arg -> 1.0 so the barrier contributes a
    constant 0 with zero gradient (OCS2 only activates contact inequalities for the stance foot;
    leaving the raw margin in would blow the barrier up at the swing foot's ~zero wrench)."""
    q = x[0:33]
    x_min, x_max = cfg.foot_rect_x
    y_min, y_max = cfg.foot_rect_y
    rows = []
    for i in (0, 1):
        flag = p[P_CONTACT][i]
        b = 6 * i
        f = u[b:b + 3]; m = u[b + 3:b + 6]
        fric = cfg.friction_mu * f[2] - cs.sqrt(f[0] ** 2 + f[1] ** 2 + cfg.friction_cone_reg)
        Rf = model.foot_R_fun[i](q)
        f_loc = Rf.T @ f; M_loc = Rf.T @ m
        fzr, Mxl, Myl = f_loc[2], M_loc[0], M_loc[1]
        cop = cs.vertcat(Mxl - y_min * fzr, -Mxl + y_max * fzr,
                         -Myl - x_min * fzr, Myl + x_max * fzr)
        h = cs.vertcat(fric, cop)                            # 5 margins (>=0 desired)
        rows.append(flag * h + (1.0 - flag) * 1.0)          # swing -> 1.0 (barrier inactive)
    return cs.vertcat(*rows)                                 # 10


def build_cost_conl(x, u, p, cfg, model, u_raw=None, P_mat=None):
    """Stage CONVEX_OVER_NONLINEAR cost; `u` is u_phys (the projected input). When (u_raw, P_mat) are
    given, append the ker(P)-confined pin √ρ·(I−P)(u_raw−u_ref) (pin rows are pure-LS; ρ does not bias
    u_phys because (I−P) is orthogonal to range(P))."""
    y_ls, yref_ls, W_ls = build_residual(x, u, p, cfg, model)
    h_bar = _contact_barrier_args(x, u, p, cfg, model)      # 10 friction/CoP margins (RelaxedBarrier)
    h_coll = _foot_collision_residual(x, p, cfg, model)     # 16 foot/knee collision margins
    blocks = [y_ls, h_bar, h_coll]
    n_ls = y_ls.shape[0]; n_bar = h_bar.shape[0]; n_coll = h_coll.shape[0]
    if u_raw is not None and P_mat is not None:
        u_ref = p[P_UREF]
        y_pin = (u_raw - u_ref) - P_mat @ (u_raw - u_ref)   # (I−P)(u_raw−u_ref)
        blocks.append(y_pin)
    y = cs.vertcat(*blocks)
    r = cs.SX.sym("r_psi", y.shape[0])
    psi = 0.5 * cs.dot(cs.DM(np.asarray(W_ls, dtype=np.float64)), r[0:n_ls] ** 2)
    bar = ([(cfg.friction_barrier_mu, cfg.friction_barrier_delta)]
           + [(cfg.cop_barrier_mu, cfg.cop_barrier_delta)] * 4) * 2
    for j, (mu, dl) in enumerate(bar):
        psi = psi + _relaxed_barrier(r[n_ls + j], float(mu), float(dl))
    for j in range(n_coll):
        psi = psi + _piecewise_poly_barrier(r[n_ls + n_bar + j], cfg.collision_barrier_mu, cfg.collision_barrier_delta)
    if u_raw is not None and P_mat is not None:
        off = n_ls + n_bar + n_coll
        psi = psi + 0.5 * float(cfg.pin_rho) * cs.sumsqr(r[off:off + cfg.nu])
    psi = (p[P_DT] / cfg.dt) * psi
    yref = np.zeros(y.shape[0])
    return y, yref, psi, r


def build_residual(x, u, p, cfg, model):
    """Stage NONLINEAR_LS residual. Returns (y_expr, yref(zeros), W_diag)."""
    x_ref = p[P_XREF]
    u_ref = p[P_UREF]
    lim = np.asarray(cfg.torque_limit, dtype=np.float64)
    fin = _finite_idx(cfg)
    tau = model.joint_torque_expr(x, u)
    g = [cs.fmax(tau[i] ** 2 - float(lim[i]) ** 2, 0.0) / float(lim[i]) ** 2 for i in fin]
    y_sf = _swing_foot_residual(x, p, cfg, model)
    y = cs.vertcat(x - x_ref, u - u_ref, *g, y_sf)
    # OCS2's diagonal GN LS weight is scaling*weight (= sqrtWeight^2), cost = 0.5*||sqrtW*g||^2; the
    # port's 0.5*sum(W*g^2) matches with W = scaling*weight (NO factor of 2 — the old 2.0 doubled it).
    w_jt = cfg.jointtorque_scale * cfg.jointtorque_weight * np.ones(len(fin))
    w_sf = np.tile(np.asarray(cfg.swingfoot_cost_weights), 2)   # 7 per foot, L then R
    W = np.concatenate([np.asarray(cfg.Q), np.asarray(cfg.R), w_jt, w_sf])
    yref = np.zeros(y.shape[0])
    return y, yref, W


def build_residual_terminal(x, p, cfg):
    """Terminal NONLINEAR_LS residual: y_e = x - x_ref, W_e = terminal_scale * Q_final."""
    y_e = x - p[P_XREF]
    W_e = cfg.terminal_scale * np.asarray(cfg.Q_final)
    yref_e = np.zeros(68)
    return y_e, yref_e, W_e


def stage_cost_value(y_val, W) -> float:
    """0.5 * sum(W * y^2) — the scalar cost (yref=0), for tests/verification."""
    yv = np.asarray(y_val, dtype=np.float64).ravel()
    return 0.5 * float(np.sum(np.asarray(W) * yv ** 2))
