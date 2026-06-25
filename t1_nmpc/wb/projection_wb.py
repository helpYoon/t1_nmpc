"""Reduced-basis state-input equality projection (faithful to OCS2 projectStateInputEqualityConstraints).

u_phys = P@u + Q@x + u_p makes the LINEARIZED contact equalities (ZeroAccel + SwingZ + ZeroWrench)
satisfied by construction. P,Q,u_p are frozen at the warm-start each tick and passed as acados params,
so acados never auto-differentiates the matrix pseudoinverse. ZeroWrench (swing-foot wrench == 0) is
FOLDED INTO the residual r (identity rows on the swing wrench), so a single exact pseudoinverse projects
onto ker([D_accel; D_swingZ; S]) jointly.
"""
from __future__ import annotations

import casadi as cs
import numpy as np

from .constraints_wb import contact_residual_gated
from .cost_wb import N_PARAM_WB, P_CONTACT


def folded_contact_residual(x, u, p, cfg, model):
    """26-row residual r(x,u,p)=0: [contact_residual_gated(14); ZeroWrench(12)].
    ZeroWrench row block for foot i = swing_i * u[6i:6i+6] (== 0 desired); gated to 0 on a stance foot."""
    r_contact = contact_residual_gated(x, u, p, cfg, model)          # 14
    zw_rows = []
    for i in (0, 1):
        swing = 1.0 - p[P_CONTACT][i]
        zw_rows.append(swing * u[6 * i:6 * i + 6])                   # 6 per foot
    return cs.vertcat(r_contact, *zw_rows)                           # 26


def build_projector_funcs(cfg, model):
    """CasADi evaluators of the folded contact residual r and its Jacobians D=dr/du, C=dr/dx."""
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu); p = cs.SX.sym("p", N_PARAM_WB)
    r = folded_contact_residual(x, u, p, cfg, model)
    return (cs.Function("proj_r", [x, u, p], [r]),
            cs.Function("proj_D", [x, u, p], [cs.jacobian(r, u)]),
            cs.Function("proj_C", [x, u, p], [cs.jacobian(r, x)]))


def compute_projector(x_node, u_node, p_node, funcs, cfg):
    """(P [nu×nu], Q [nu×nx], u_p [nu]) for u_phys = P@u + Q@x + u_p — the exact (rank-detecting)
    linearized projection of the folded contact equalities at the warm-start (x_node, u_node).
    P = I − D⁺D (orthogonal projector onto ker(D)); u_p places u_node on the feasible manifold."""
    r_fun, D_fun, C_fun = funcs
    x0 = np.asarray(x_node, dtype=np.float64); u0 = np.asarray(u_node, dtype=np.float64)
    r0 = np.asarray(r_fun(x0, u0, p_node)).ravel()
    D = np.asarray(D_fun(x0, u0, p_node), dtype=np.float64)        # nr × nu
    C = np.asarray(C_fun(x0, u0, p_node), dtype=np.float64)        # nr × nx
    Dp = np.linalg.pinv(D)                                         # nu × nr, rank-detecting (no Tikhonov)
    DpD = Dp @ D                                                   # nu × nu == I − P
    P = np.eye(cfg.nu) - DpD
    Q = -Dp @ C
    u_p = DpD @ u0 - Dp @ r0 + Dp @ C @ x0
    return P, Q, u_p
