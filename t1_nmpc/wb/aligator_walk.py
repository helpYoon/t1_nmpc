"""Per-contact-mode StageModel factory + fixed-N TrajOptProblem builder for the kinodynamic walk.
Stance feet: zero-vel equality + Centroidal friction/wrench cones (hard NegativeOrthant or soft
RelaxedLogBarrierCost). Swing feet: FrameTranslation z-track + heavy force-slot regularization."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
import aligator
from aligator import dynamics, constraints
from .aligator_model import make_ode


def _foot_half_extents(wb_cfg):
    return ((wb_cfg.foot_rect_x[1] - wb_cfg.foot_rect_x[0]) / 2.0,
            (wb_cfg.foot_rect_y[1] - wb_cfg.foot_rect_y[0]) / 2.0)


def _weights(am, al_cfg, contact_flags, FS=6):
    nv = am.nv
    ndx = am.ndx
    wx = np.r_[np.full(6, al_cfg.w_base_pose), np.full(nv - 6, al_cfg.w_joint_pos), np.full(nv, al_cfg.w_vel)]
    nu = 2 * FS + (nv - 6)
    wu = np.empty(nu)
    wu[:2 * FS] = al_cfg.w_force_reg
    wu[2 * FS:] = al_cfg.w_accel_reg
    for k, on in enumerate(contact_flags):           # pin a SWING foot's force slots to zero
        if not on:
            wu[k * FS:(k + 1) * FS] = al_cfg.w_swing_force
    return wx, wu


def make_stage(am, wb_cfg, al_cfg, contact_flags, x_ref, swing_refs, ode, FS=6):
    nu = ode.nu
    ndx = am.ndx
    mu = float(wb_cfg.friction_mu)
    L, W = _foot_half_extents(wb_cfg)
    mg = am.mass * 9.81
    nst = max(1, sum(contact_flags))
    wx, wu = _weights(am, al_cfg, contact_flags, FS)
    u_ref = np.zeros(nu)
    for k, on in enumerate(contact_flags):
        if on:
            u_ref[k * FS + 2] = mg / nst             # weight-supporting reference
    cost = aligator.CostStack(am.space, nu)
    cost.addCost("xreg", aligator.QuadraticStateCost(am.space, nu, x_ref, np.diag(wx)))
    cost.addCost("ureg", aligator.QuadraticControlCost(am.space, u_ref, np.diag(wu)))
    for foot_idx, p_ref in swing_refs:
        ft = aligator.FrameTranslationResidual(ndx, nu, am.model, np.asarray(p_ref, float), int(am.foot_ids[foot_idx]))
        cost.addCost(f"swz{foot_idx}", aligator.QuadraticResidualCost(am.space, ft, np.diag([0., 0., al_cfg.w_swing_z])))
    st = aligator.StageModel(cost, dynamics.IntegratorSemiImplEuler(ode, float(wb_cfg.dt)))
    for k, on in enumerate(contact_flags):
        if not on:
            continue
        zv = aligator.FrameVelocityResidual(ndx, nu, am.model, pin.Motion.Zero(), int(am.foot_ids[k]), pin.LOCAL_WORLD_ALIGNED)
        st.addConstraint(zv, constraints.EqualityConstraintSet())
        fr = aligator.CentroidalFrictionConeResidual(ndx, nu, k, mu, al_cfg.cone_eps)
        wc = aligator.CentroidalWrenchConeResidual(ndx, nu, k, mu, L, W)
        if al_cfg.hard_cones:
            st.addConstraint(fr, constraints.NegativeOrthant())
            st.addConstraint(wc, constraints.NegativeOrthant())
        else:
            cost.addCost(f"fric{k}", aligator.RelaxedLogBarrierCost(am.space, fr, np.ones(2), al_cfg.barrier_thr))
            cost.addCost(f"wcon{k}", aligator.RelaxedLogBarrierCost(am.space, wc, np.ones(17), al_cfg.barrier_thr))
    return st


def build_problem(am, wb_cfg, al_cfg, x0, x_ref, schedule, swing_schedule, FS=6):
    odes = {}

    def ode_for(flags):
        key = tuple(flags)
        if key not in odes:
            odes[key] = make_ode(am, flags, FS)
        return odes[key]

    stages = [make_stage(am, wb_cfg, al_cfg, schedule[t], x_ref, swing_schedule[t], ode_for(schedule[t]), FS)
              for t in range(al_cfg.N)]
    wx, _ = _weights(am, al_cfg, [True, True], FS)
    term = aligator.CostStack(am.space, stages[0].nu)
    term.addCost("xt", aligator.QuadraticStateCost(am.space, stages[0].nu, x_ref, np.diag(wx * al_cfg.w_term_scale)))
    return aligator.TrajOptProblem(x0, stages, term)


def build_gait_cycle(am, wb_cfg, al_cfg, gait, x_ref, node_times, FS=6):
    odes = {}
    def ode_for(flags):
        k = tuple(flags)
        if k not in odes: odes[k] = make_ode(am, flags, FS)
        return odes[k]
    models, schedule = [], []
    for t in np.asarray(node_times, float):
        flags = [bool(b) for b in gait.contact_flags(float(t))]
        swing_refs = []
        for i, on in enumerate(flags):
            if not on:
                z, _, _ = gait.swing_z(float(t), i)        # gait swing-z height (xy target = current foot xy)
                rdata = am.model.createData(); pin.framesForwardKinematics(am.model, rdata, x_ref[:am.nq])
                p = rdata.oMf[int(am.foot_ids[i])].translation.copy(); p[2] = z
                swing_refs.append((i, p))
        models.append(make_stage(am, wb_cfg, al_cfg, flags, x_ref, swing_refs, ode_for(flags), FS))
        schedule.append(flags)
    return models, schedule
