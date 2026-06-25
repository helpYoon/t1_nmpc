# t1_nmpc/wb/croco_costs.py
"""Crocoddyl cost/constraint residual builders for the T1 whole-body MPC (M0/M1).

All terms map t1_controller costs/constraints to NATIVE crocoddyl residuals reading
config_wb. t1_controller's inequalities are soft relaxed-barriers, so penalty costs are
faithful; M0 approximates the barrier SHAPE with QuadraticBarrier (exact relaxed barrier
is M1). Swing/arm-swing/foot-collision are M1 (gated by walk=True).

M1 extensions (walk=True):
  - Faithful relaxed-barrier FrictionCone + CoP (vs M0's QuadraticBarrier WrenchCone)
  - Yawed R_foot rotation from planted dict
  - Stance z-ground + foot-flat stabilization costs
  - Swing-foot block (foot-flat, lin/ang-vel -> 0, z-tracking)
  - Terminal Q_final*terminal_scale cost (state-only)
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import crocoddyl

from .croco_activations import RelaxedBarrier

_BIG = 1e3  # effective +inf for one-sided state bounds
_LWA = pin.LOCAL_WORLD_ALIGNED


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


def _flat_se3(planted, fid):
    """Return a copy of the planted SE3 with rotation zeroed (foot-flat orientation reference)."""
    p = planted[fid].copy()
    p.rotation = np.eye(3)
    return p


def _frame_vel_zero(state, fid, nu):
    """ResidualModelFrameVelocity targeting zero motion (LOCAL_WORLD_ALIGNED)."""
    return crocoddyl.ResidualModelFrameVelocity(state, fid, pin.Motion.Zero(), _LWA, nu)


def build_costs(state, actuation, nu, x_ref, com_ref, stance_fids, cfg,
                swing=None, planted=None, terminal=False, walk=False):
    nv = state.pinocchio.nv
    nc = 6 * len(stance_fids)
    costs = crocoddyl.CostModelSum(state, nu)

    # --- terminal: state-only Q_final*terminal_scale ---
    if terminal:
        xres = crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)
        wq = crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q_final[:66], float) * float(cfg.terminal_scale))
        costs.addCost("xreg", crocoddyl.CostModelResidual(state, wq, xres), 1.0)
        return costs

    # --- 1. state tracking/regularization (weights = config_wb.Q diagonal, 67->66) ---
    xreg = crocoddyl.ResidualModelState(state, np.asarray(x_ref, float), nu)
    xact = crocoddyl.ActivationModelWeightedQuad(np.asarray(cfg.Q[:66], float))
    costs.addCost("xreg", crocoddyl.CostModelResidual(state, xact, xreg), 1.0)

    # --- 2. CoM tracking (M0: com_ref = com0, low weight; forward drive is M1) ---
    creg = crocoddyl.ResidualModelCoMPosition(state, np.asarray(com_ref, float), nu)
    costs.addCost("com", crocoddyl.CostModelResidual(state, creg), 1.0)

    # --- 3. input regularization (weights from config_wb.R) ---
    ureg = crocoddyl.ResidualModelControl(state, nu)
    uact = crocoddyl.ActivationModelWeightedQuad(_control_weights(nv, nc, np.asarray(cfg.R, float)))
    costs.addCost("ureg", crocoddyl.CostModelResidual(state, uact, ureg), 1.0)

    # --- 4. torque-limit soft barrier on recovered tau (JointEffort) ---
    tau_lim = np.asarray(cfg.torque_limit, float)
    teff = crocoddyl.ResidualModelJointEffort(state, actuation, np.zeros(actuation.nu), nu, False)
    tbar = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-tau_lim, tau_lim))
    costs.addCost("tau_lim", crocoddyl.CostModelResidual(state, tbar, teff),
                  float(cfg.jointtorque_weight))

    # --- 5. joint-position-limit soft barrier (bounds relative to neutral on the joint block) ---
    lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
    lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float)
    ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
    jres = crocoddyl.ResidualModelState(state, np.zeros(state.nx), nu)
    jbar = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
    costs.addCost("joint_lim", crocoddyl.CostModelResidual(state, jbar, jres),
                  float(cfg.joint_limit_barrier_mu))

    if not walk:
        # --- M0 stance: combined WrenchCone + QuadraticBarrier (unchanged) ---
        box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)
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

    # --- M1 WALK: faithful relaxed-barrier friction + CoP, yawed R_foot, stance stabilization ---
    box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)
    for fid in stance_fids:
        R_foot = np.asarray(planted[fid].rotation, float)
        fcone = crocoddyl.FrictionCone(R_foot, float(cfg.friction_mu), 4, False)
        fres = crocoddyl.ResidualModelContactFrictionCone(state, fid, fcone, nu, False)
        costs.addCost(f"friction_{fid}", crocoddyl.CostModelResidual(
            state, RelaxedBarrier(fres.nr, cfg.friction_barrier_mu, cfg.friction_barrier_delta), fres),
            float(cfg.friction_cone_reg))
        cop = crocoddyl.CoPSupport(R_foot, box)
        cres = crocoddyl.ResidualModelContactCoPPosition(state, fid, cop, nu, False)
        costs.addCost(f"cop_{fid}", crocoddyl.CostModelResidual(
            state, RelaxedBarrier(cres.nr, cfg.cop_barrier_mu, cfg.cop_barrier_delta), cres), 1.0)
        # stance z->ground + foot-flat stabilization (weights seeded from gains)
        zres = crocoddyl.ResidualModelFrameTranslation(state, fid, planted[fid].translation, nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 1.], float))   # z only
        costs.addCost(f"stance_z_{fid}", crocoddyl.CostModelResidual(state, zact, zres),
                      float(cfg.foot_pos_err_gain_z))
        flat = crocoddyl.ResidualModelFramePlacement(state, fid, planted[fid], nu)
        flatact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 0., 1., 1., 1.], float))  # rot only
        costs.addCost(f"stance_flat_{fid}", crocoddyl.CostModelResidual(state, flatact, flat),
                      float(cfg.foot_ori_err_gain))

    # --- M1 swing-foot block (foot-flat, vel, z-track), w_z folded in by caller ---
    if swing is not None:
        sfid = int(swing["fid"]); wz = float(swing["w_z"])
        sw = float(cfg.swingfoot_cost_weights[0])  # ori_xy weight 1e4 (impact scale applied by caller)
        # foot-flat orientation (FramePlacement, rotation rows only)
        fp = crocoddyl.ResidualModelFramePlacement(state, sfid, _flat_se3(planted, sfid), nu)
        fpact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 0., 1., 1., 1.], float))
        costs.addCost("swing_flat", crocoddyl.CostModelResidual(state, fpact, fp), sw)
        # lin-vel xy -> 0, ang-vel -> 0
        velact = crocoddyl.ActivationModelWeightedQuad(np.array(
            [cfg.swingfoot_cost_weights[2], cfg.swingfoot_cost_weights[3], 0.,
             cfg.swingfoot_cost_weights[4], cfg.swingfoot_cost_weights[5], cfg.swingfoot_cost_weights[6]], float))
        costs.addCost("swing_vel", crocoddyl.CostModelResidual(state, velact, _frame_vel_zero(state, sfid, nu)), 1.0)
        # swing-z tracking (strong cost; replaces the hard SwingLegVerticalConstraint)
        ztarget = np.array([0., 0., float(swing["z"])], float)
        zres = crocoddyl.ResidualModelFrameTranslation(state, sfid, ztarget, nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0., 0., 1.], float))
        costs.addCost("swing_z", crocoddyl.CostModelResidual(state, zact, zres), wz)
    return costs
