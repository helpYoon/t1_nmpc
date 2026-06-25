# t1_nmpc/wb/croco_costs.py
"""Crocoddyl cost/constraint residual builders for the T1 whole-body MPC (M0).

All terms map t1_controller costs/constraints to NATIVE crocoddyl residuals reading
config_wb. t1_controller's inequalities are soft relaxed-barriers, so penalty costs are
faithful; M0 approximates the barrier SHAPE with QuadraticBarrier (exact relaxed barrier
is M1). Swing/arm-swing/foot-collision are M1 (skipped here)."""
from __future__ import annotations

import numpy as np
import crocoddyl

_BIG = 1e3  # effective +inf for one-sided state bounds


def _control_weights(nv: int, nc: int, R: np.ndarray) -> np.ndarray:
    """Map config_wb.R [W_l(6),W_r(6),qdd(27),vdot_s(1)] to crocoddyl control
    [a(nv); forces(nc)] weights. a[0:6]=base accel (constrained) -> tiny; a[6:33]=qdd
    -> R[12:39]; forces ordered [left, right] -> R[0:6], R[6:12]."""
    assert nv == 33, f"nv={nv}"  # qdd/force slices assume nv==33
    w = np.empty(nv + nc)
    w[0:6] = 1e-6
    w[6:nv] = R[12:39]
    if nc >= 6:
        w[nv:nv + 6] = R[0:6]            # left foot wrench
    if nc >= 12:
        w[nv + 6:nv + 12] = R[6:12]      # right foot wrench
    return w


def build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg):
    nv = state.pinocchio.nv
    nc = 6 * len(stance_fids)
    costs = crocoddyl.CostModelSum(state, nu)

    # 1. state tracking/regularization (weights = config_wb.Q diagonal, 67->66)
    xreg = crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)
    xact = crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q[:66], float))
    costs.addCost("xreg", crocoddyl.CostModelResidual(state, xact, xreg), 1.0)

    # 2. CoM tracking (M0: com_ref = com0, low weight; forward drive is M1)
    creg = crocoddyl.ResidualModelCoMPosition(state, np.asarray(com_ref, float), nu)
    costs.addCost("com", crocoddyl.CostModelResidual(state, creg), 1.0)

    # 3. input regularization (weights from config_wb.R)
    ureg = crocoddyl.ResidualModelControl(state, nu)
    uact = crocoddyl.ActivationModelWeightedQuad(_control_weights(nv, nc, np.asarray(cfg.R, float)))
    costs.addCost("ureg", crocoddyl.CostModelResidual(state, uact, ureg), 1.0)

    # 4. torque-limit soft barrier on recovered tau (JointEffort)
    tau_lim = np.asarray(cfg.torque_limit, float)
    teff = crocoddyl.ResidualModelJointEffort(state, actuation, np.zeros(actuation.nu), nu, False)
    tbar = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-tau_lim, tau_lim))
    costs.addCost("tau_lim", crocoddyl.CostModelResidual(state, tbar, teff),
                  float(cfg.jointtorque_weight))

    # 5. joint-position-limit soft barrier (bounds relative to neutral on the joint block)
    lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
    lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float)
    ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
    jres = crocoddyl.ResidualModelState(state, np.zeros(state.nx), nu)
    jbar = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
    costs.addCost("joint_lim", crocoddyl.CostModelResidual(state, jbar, jres),
                  float(cfg.joint_limit_barrier_mu))

    # 6. friction-cone + CoP-in-rectangle + unilateral, per stance foot (WrenchCone)
    box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)   # half-extents
    R_foot = np.eye(3)                                                # feet flat on flat ground (M0)
    for fid in stance_fids:
        cone = crocoddyl.WrenchCone(R_foot, float(cfg.friction_mu), box)
        wres = crocoddyl.ResidualModelContactWrenchCone(state, fid, cone, nu, False)
        wbar = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub))
        costs.addCost(f"wrenchcone_{fid}",
                      crocoddyl.CostModelResidual(state, wbar, wres),
                      float(cfg.friction_cone_reg))
    return costs
