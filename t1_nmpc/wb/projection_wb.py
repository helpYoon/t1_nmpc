"""Numerical contact-equality projection, computed per node and passed as acados params.

u_phys = P @ u + Q @ x + u_p makes the LINEARIZED contact equalities (ZeroAccel/SwingZ/ZeroWrench)
satisfied by construction, so acados never auto-differentiates the projector's matrix inverse (which
blew the codegen up to ~195 MB). P, Q, u_p are frozen at the warm-start point each tick -- exactly
OCS2's projectStateInputEqualityConstraints (it projects the linearized constraint per SQP iteration).

The state-coupling Q = -D^T (DD^T+eps I)^-1 C is REQUIRED: without it the projection leaves a
first-order C*dx contact violation (verified). G zeroes the swing-foot wrench (ZeroWrench), baked into P.
"""
from __future__ import annotations

import casadi as cs
import numpy as np

from .constraints_wb import contact_residual_gated
from .cost_wb import N_PARAM_WB, P_CONTACT


def build_projector_funcs(cfg, model):
    """CasADi evaluators of the gated contact residual r and its Jacobians dr/du, dr/dx."""
    x = cs.SX.sym("x", cfg.nx); u = cs.SX.sym("u", cfg.nu); p = cs.SX.sym("p", N_PARAM_WB)
    r = contact_residual_gated(x, u, p, cfg, model)
    return (cs.Function("proj_g", [x, u, p], [r]),
            cs.Function("proj_D", [x, u, p], [cs.jacobian(r, u)]),
            cs.Function("proj_C", [x, u, p], [cs.jacobian(r, x)]))


def compute_projector(x_node, u_node, p_node, funcs, cfg):
    """(P [nu x nu], Q [nu x nx], u_p [nu]) for u_phys = P@u + Q@x + u_p, the regularized linearized
    projection of the contact equalities at (x_node, u_node). Swing-foot wrench zeroing baked in via G."""
    g_fun, D_fun, C_fun = funcs
    nu = cfg.nu
    x_node = np.asarray(x_node, dtype=np.float64); u_node = np.asarray(u_node, dtype=np.float64)
    fl = float(p_node[P_CONTACT.start]); fr = float(p_node[P_CONTACT.start + 1])
    G = np.eye(nu); G[0:6, 0:6] *= fl; G[6:12, 6:12] *= fr        # zero the SWING-foot wrench (ZeroWrench)
    u_g = G @ u_node
    g = np.asarray(g_fun(x_node, u_g, p_node)).ravel()
    D = np.asarray(D_fun(x_node, u_g, p_node))                    # nr x nu
    C = np.asarray(C_fun(x_node, u_g, p_node))                    # nr x nx
    nr = D.shape[0]
    DtMinv = D.T @ np.linalg.inv(D @ D.T + cfg.contact_proj_eps * np.eye(nr))   # nu x nr
    P = (np.eye(nu) - DtMinv @ D) @ G
    Q = -DtMinv @ C
    u_p = -DtMinv @ (g - C @ x_node - D @ u_g)
    return P, Q, u_p
