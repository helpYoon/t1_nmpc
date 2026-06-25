"""WB path EQUALITY constraints, RAW (ungated) + per-stage activation via bounds (M1 walking).

14 rows, per-foot block [ZeroAccel(6), SwingZ(1)]; foot L rows 0-6, foot R rows 7-13:
  ZeroAccel(6) : accel + Av*twist + Ax*pose                          (active for a STANCE foot)
  SwingZ(1)    : 100*(z-zref)+10*(zdot-zdotref)+1*(zddot-zddotref)   (active for a SWING foot)

zeroWrench (swing-foot wrench == 0) is NOT here: it is an identity block on the 6 wrench INPUTS, so it
lives in the input box bounds (lbu/ubu) where HPIPM handles it natively (see stage_wrench_bounds + ocp_wb).

Constraints are emitted RAW (no flag gating) so the Jacobian A stays FULL RANK. Activation is via PER-STAGE
BOUNDS (stage_constraint_bounds, set each tick in mpc_wb): active equality -> lh=uh=0; inactive ->
lh=-ACADOS_INFTY, uh=+ACADOS_INFTY (ignored). The OLD approach gated the EXPRESSION to zero, which made the
inactive rows ZERO Jacobian rows -> rank-deficient A -> singular KKT -> HPIPM MINSTEP/ill-conditioning
(13 of 26 rows were zero at single support). OCS2 avoids this by conditionally INCLUDING only active
constraints; acados needs a fixed con_h dim, so we keep all rows and toggle the bounds.

Faithful to t1_controller:
  ZeroAccel  -- EndEffectorDynamicsAccelerationsConstraint (Aa=I; Ax/Av gains task.info foot_constraint)
  SwingZ     -- EndEffectorDynamicsLinearAccConstraint (gains 100/10/1)
"""
from __future__ import annotations

import casadi as cs
import numpy as np
from acados_template import ACADOS_INFTY

NH = 14            # con_h rows: per-foot [ZeroAccel(6), SwingZ(1)]
NBU = 12           # input box bounds: per-foot wrench W (6); idxbu = u[0:12] = [W_l, W_r]


def build_con_h(x, u, p, cfg, model):
    """Raw (ungated) 14-row con_h + default ALL-INACTIVE bounds (per-stage bounds activate at runtime)."""
    from .cost_wb import P_SWINGZ
    q = x[0:33]; v = x[33:66]
    _v, qdd_j, _M, _nle, _te, vdot_base = model._dyn_terms(x, u)
    a_full = cs.vertcat(vdot_base, qdd_j)
    gz = cfg.foot_pos_err_gain_z
    go = cfg.foot_ori_err_gain
    lvz = cfg.foot_linvel_err_gain_z
    lvxy = cfg.foot_linvel_err_gain_xy
    av = cfg.foot_angvel_err_gain
    gaz = cfg.foot_linacc_err_gain_z
    Ax = cs.DM(np.diag([0.0, 0.0, gz, go, go, go]))
    Av = cs.DM(np.diag([lvxy, lvxy, lvz, av, av, av]))
    rows = []
    for i in (0, 1):
        twist, accel, pose = model.foot_kin_fun[i](q, v, a_full)
        za = accel + Av @ twist + Ax @ pose                                  # ZeroAccel (6), RAW
        zref = p[P_SWINGZ][3 * i]; zdref = p[P_SWINGZ][3 * i + 1]; zddref = p[P_SWINGZ][3 * i + 2]
        sv = gz * (pose[2] - zref) + lvz * (twist[2] - zdref) + gaz * (accel[2] - zddref)   # SwingZ (1), RAW
        rows += [za, cs.vertcat(sv)]
    con_h = cs.vertcat(*rows)                                                 # 14 = 2*(6+1)
    lh = -ACADOS_INFTY * np.ones(NH)                                          # default: all inactive
    uh = ACADOS_INFTY * np.ones(NH)
    return con_h, lh, uh


def contact_residual_gated(x, u, p, cfg, model):
    """The contact EQUALITIES as a single gated residual r(x,u,p) that should be 0, for the input
    PROJECTION (ocp_wb._project_inputs) -- NOT a con_h constraint. 14 rows, per-foot block
    [ZeroAccel(6) x stance, SwingZ(1) x swing] (ZeroWrench handled by gating the swing wrench in
    _project_inputs, to keep the projector small enough to codegen). Gating-to-zero is FINE here
    (unlike con_h) because the regularized projector (DDt + eps*I)^-1 makes zero rows harmless.
    r is AFFINE in u (verified), so D = dr/du is constant in u and the projection is an affine map."""
    from .cost_wb import P_CONTACT, P_SWINGZ
    q = x[0:33]; v = x[33:66]
    _v, qdd_j, _M, _nle, _te, vdot_base = model._dyn_terms(x, u)
    a_full = cs.vertcat(vdot_base, qdd_j)
    gz = cfg.foot_pos_err_gain_z; go = cfg.foot_ori_err_gain
    lvz = cfg.foot_linvel_err_gain_z; lvxy = cfg.foot_linvel_err_gain_xy
    av = cfg.foot_angvel_err_gain; gaz = cfg.foot_linacc_err_gain_z
    Ax = cs.DM(np.diag([0.0, 0.0, gz, go, go, go]))
    Av = cs.DM(np.diag([lvxy, lvxy, lvz, av, av, av]))
    flags = (p[P_CONTACT][0], p[P_CONTACT][1])
    rows = []
    for i in (0, 1):
        flag = flags[i]; swing = 1.0 - flag
        twist, accel, pose = model.foot_kin_fun[i](q, v, a_full)
        za = flag * (accel + Av @ twist + Ax @ pose)                          # ZeroAccel (6) x stance
        zref = p[P_SWINGZ][3 * i]; zdref = p[P_SWINGZ][3 * i + 1]; zddref = p[P_SWINGZ][3 * i + 2]
        sv = swing * (gz * (pose[2] - zref) + lvz * (twist[2] - zdref) + gaz * (accel[2] - zddref))  # SwingZ (1) x swing
        rows += [za, cs.vertcat(sv)]
    return cs.vertcat(*rows)                                                  # 14 = 2*(6+1); ZeroWrench handled by gating the swing wrench in _project_inputs


def stage_constraint_bounds(lf, rf):
    """(lh, uh) for con_h at a node with stance flags (lf, rf). ZeroAccel active for a STANCE foot;
    SwingZ active for a SWING foot. Active -> [0,0]; inactive -> [-ACADOS_INFTY, +ACADOS_INFTY]."""
    lh = -ACADOS_INFTY * np.ones(NH); uh = ACADOS_INFTY * np.ones(NH)
    for i, stance in enumerate((bool(lf), bool(rf))):
        za = slice(7 * i, 7 * i + 6); sz = 7 * i + 6
        if stance:
            lh[za] = 0.0; uh[za] = 0.0          # stance: zero foot acceleration
        else:
            lh[sz] = 0.0; uh[sz] = 0.0          # swing: track the swing-Z spline
    return lh, uh


def stage_wrench_bounds(lf, rf):
    """(lbu, ubu) over idxbu = u[0:12] (W_l, W_r). Swing-foot wrench forced to 0; stance-foot free."""
    lbu = -ACADOS_INFTY * np.ones(NBU); ubu = ACADOS_INFTY * np.ones(NBU)
    for i, stance in enumerate((bool(lf), bool(rf))):
        if not stance:
            lbu[6 * i:6 * i + 6] = 0.0; ubu[6 * i:6 * i + 6] = 0.0   # swing -> zero wrench
    return lbu, ubu
