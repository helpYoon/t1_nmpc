# t1_nmpc/wb/croco_walk.py
"""Persistent walk OCP for the single-RTI CrocoMPC, built once and mutated in place.

Why this exists (vs rebuilding the ShootingProblem every cycle):
  t1_controller/OCS2 runs multiple-shooting SQP with `sqpIteration 1` and `coldStart false`
  (task.info): one SQP iteration per MPC update, warm-started from the PERSISTENT previous
  solution (SqpSolver.cpp re-uses primalSolution_, spreads it across mode-schedule changes, and
  feeds the initial-state gap delta_x0 = initState - x[0] into the QP). To be faithful we must
  reuse ONE crocoddyl solver and warm-start it from the previous solution.

  crocoddyl's solver pre-allocates per-node data sized to each node's control dimension and its
  `problem` is not reassignable, so the persistent solver requires a FIXED structure. We get that
  by putting BOTH foot contacts on EVERY node (nu = nv + 12 everywhere) and toggling each contact's
  active status per cycle with `changeContactStatus` -- which keeps nu fixed and genuinely frees a
  swing foot (an inactive ContactInvDynamics contact behaves exactly like an absent one). Contact
  placements, cost references, and stance/swing cost activity are mutated in place each cycle; the
  per-node residual/contact handles are captured at build time because `costs.todict()` excludes
  inactive costs.

  Friction/CoP use ActivationModelQuadraticBarrier (M0's bounded-Hessian barrier-shape
  approximation), NOT the exact OCS2 log-relaxed barrier: the log barrier's ~1/h Hessian is too
  stiff for a single DDP iteration and makes the warm-started solve diverge. OCS2 tolerates it only
  because each SQP iteration solves a full QP.
"""
from __future__ import annotations

import os
import numpy as np
import pinocchio as pin
import crocoddyl

# --- LANDED stable crocoddyl/DDP baseline (defaults below). Several faithful-in-isolation OCS2 fixes
# DESTABILIZE warm-started DDP and are OFF by default (bisected 2026-06-26) -- deferred to the aligator
# (ProxDDP, hard-constraint) port where they belong. Kept env-togglable for that comparison spike.
#   _U67 "quad"  : friction/CoP barrier. "quad" = M0's bounded-Hessian QuadraticBarrier (STABLE).
#                  "relaxed" = OCS2's interior-point log barrier -- FAITHFUL but its ~1/h Hessian
#                  COLLAPSES the walk @ t~0.09s at every iteration count (maxit 80 WORSE than 40 ->
#                  defines a bad optimum, not under-convergence). OCS2 survives it only via full-QP SQP.
#   _U5 off      : torque-limit weight x jointtorque_scale (1e2). ON -> falls 1.18s (starves push-off).
#   _U6 off      : friction-cone normal-force floor. ON -> falls 1.14s (fights DS->SS foot unloading).
#   _U1 on       : self-collision barrier -- HELPS (1.45->1.84s; feet really collide at the topple).
_U5 = os.environ.get("T1_U5", "0") == "1"          # torque-limit weight x jointtorque_scale (OFF: hurts)
_U67 = os.environ.get("T1_U67", "quad")             # friction/cop barrier: "quad" (stable) | "relaxed"
_U6 = os.environ.get("T1_U6", "0") == "1"           # friction-cone normal-force floor (OFF: hurts)
_U1 = os.environ.get("T1_U1", "1") == "1"           # self-collision barrier (ON: helps)

from . import reference_wb
from .croco_costs import _control_weights
from .croco_swingz import build_swingz_fn, SwingZResidual
from .croco_contact import PerComponentContact6D, stance_gains
from .croco_collision import build_collision_fn, collision_lb, CollisionResidual
from .croco_activations import ActivationModelRelaxedBarrier

_LWA = pin.LOCAL_WORLD_ALIGNED
_BIG = 1e3  # effective +inf for one-sided state bounds


def _qbar(lb, ub):
    return crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(np.asarray(lb, float), np.asarray(ub, float)))


def _wq(w):
    return crocoddyl.ActivationModelWeightedQuad(np.asarray(w, float))


def _relbar(lb, ub, mu, delta):
    """OCS2 relaxed/interior-point barrier (friction/CoP) -- the barrier mu IS the strength (cost
    weight 1.0), unlike the QuadraticBarrier which needed a misused scalar weight (U7/U8)."""
    return ActivationModelRelaxedBarrier(
        crocoddyl.ActivationBounds(np.asarray(lb, float), np.asarray(ub, float)), float(mu), float(delta))


class WalkOCP:
    """The persistent walk shooting problem + in-place updater.

    `self.problem` is built once (both contacts on every node). Call `update(...)` each MPC cycle to
    retarget references and toggle stance/swing per node, then solve the SAME solver warm-started
    from the previous solution.
    """

    def __init__(self, cfg, wb):
        self.cfg = cfg
        self.wb = wb
        self.model = wb.model
        self.nv = self.model.nv
        self.nq = self.model.nq
        self.N = int(cfg.N)
        self.dt = float(cfg.dt)
        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.fids = list(wb.contact_fids)               # [L, R]
        self.nu = self.nv + 12                          # both contacts on every node
        # crocoddyl CoPSupport takes the FULL support-rectangle dimensions and bounds the CoP to
        # +/-box/2 (verified). The half-extents were being passed as full dims -> support polygon
        # silently halved. Pass the full foot length/width = OCS2's contact_rectangle.
        self._box = np.array([cfg.foot_rect_x[1] - cfg.foot_rect_x[0],
                              cfg.foot_rect_y[1] - cfg.foot_rect_y[0]], float)
        # OCS2 getStanceFootConstraint (MpcInterface.cpp:333-356): the stance foot is a HARD 6D
        # Baumgarte equality with position/orientation feedback (positionErrorGain_z=100,
        # orientationErrorGain=80) + velocity. crocoddyl's ContactModel6D applies a single [kp, kd]
        # uniformly; kp=foot_pos_err_gain_z hard-holds the stance foot's height+orientation (xy gets
        # corrected too, harmless under no-slip). kp=0 (emergent landing) let the stance foot
        # drift/yaw -> base divergence -> fall. Matches the working stand path (croco_problem.py).
        # faithful OCS2 getStanceFootConstraint per-component Baumgarte diagonals (xy-pos free,
        # pos_z=100, ori=80, vel 20/10/20) -- selective, so it conditions where uniform kp=100 diverges.
        self._stance_Ax, self._stance_Av = stance_gains(cfg)
        self._mg = float(wb.total_mass() * 9.81)         # body weight, for the weight-comp control ref
        # U1: foot/knee self-collision avoidance (t1_controller FootCollisionConstraint, was dropped).
        self._coll_fn = build_collision_fn(wb)
        self._coll_lb = collision_lb(0.07, 0.075, 0.03)  # 2*radius + 3cm repel margin per sphere pair
        # per-node handles (index 0..N; N is terminal)
        self._con = []     # ContactModelMultiple per node
        self._cos = []     # CostModelSum per node
        self._ch = []      # {foot_idx: ContactModel6D}
        self._rh = []      # {cost_name: ResidualModel}
        self._cns = []     # ConstraintModelManager per node (hard swing-z equality)
        self._csz = []     # {foot_idx: SwingZResidual} per node
        # faithful swing-z = t1_controller foot_constraint Baumgarte gains (task.info)
        self._swingz_fns = [build_swingz_fn(wb, i, cfg.foot_pos_err_gain_z,
                                            cfg.foot_linvel_err_gain_z,
                                            cfg.foot_linacc_err_gain_z) for i in range(len(self.fids))]
        self.problem = self._build()

    # ---- construction -------------------------------------------------------
    def _nominal(self):
        q0 = pin.neutral(self.model); q0[2] = self.cfg.nominal_base_height
        q0[6:6 + self.cfg.n_joints] = self.cfg.nominal_joint_pos
        return np.concatenate([q0, np.zeros(self.nv)])

    def _planted(self, x66):
        q = np.asarray(x66[:self.nq], float)
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        return {fid: data.oMf[fid].copy() for fid in self.fids}

    def _build_node(self, x0, planted, terminal=False):
        nu = self.nu
        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        ch = {}
        for i, fid in enumerate(self.fids):
            c = PerComponentContact6D(self.state, fid, planted[fid], nu, self._stance_Ax, self._stance_Av)
            contacts.addContact("c%d" % i, c); ch[i] = c
        costs = crocoddyl.CostModelSum(self.state, nu)
        rh = {}

        def add(name, activation, residual, weight):
            costs.addCost(name, crocoddyl.CostModelResidual(self.state, activation, residual), float(weight))
            rh[name] = residual

        cfg = self.cfg
        if terminal:
            add("xreg", _wq(np.asarray(cfg.Q_final[:66], float) * float(cfg.terminal_scale)),
                crocoddyl.ResidualModelState(self.state, x0, nu), 1.0)
            dam = crocoddyl.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contacts, costs)
            return crocoddyl.IntegratedActionModelEuler(dam, 0.0), contacts, costs, ch, rh, None, {}

        # common costs
        add("xreg", _wq(cfg.Q[:66]), crocoddyl.ResidualModelState(self.state, x0, nu), 1.0)
        # NB: OCS2 has NO CoM-position cost (base pose is tracked via Q in xreg, CoM emergent). A CoM
        # cost tracking a forward-ramping reference reintroduces forward coupling OCS2 omits -> removed.
        add("ureg", _wq(_control_weights(self.nv, 12, np.asarray(cfg.R, float))),
            crocoddyl.ResidualModelControl(self.state, nu), 1.0)
        tau = np.asarray(cfg.torque_limit, float)
        _tauw = float(cfg.jointtorque_weight * cfg.jointtorque_scale) if _U5 else float(cfg.jointtorque_weight)
        add("tau_lim", _qbar(-tau, tau),
            crocoddyl.ResidualModelJointEffort(self.state, self.actuation, np.zeros(self.actuation.nu), nu, False),
            _tauw)   # U5: OCS2 effective weight = 1.0*1e2
        lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
        lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float)
        ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
        add("joint_lim", _qbar(lb, ub),
            crocoddyl.ResidualModelState(self.state, np.zeros(self.state.nx), nu), float(cfg.joint_limit_barrier_mu))
        # U1: foot/knee self-collision barrier (always active). Penalizes pairwise sphere distances
        # below 2*radius + margin -> repels the swing foot before it crosses into the stance foot.
        add("collision", _qbar(self._coll_lb, np.full(len(self._coll_lb), _BIG)),
            CollisionResidual(self.state, nu, self._coll_fn, self.nq, self.nv), 500.0 if _U1 else 0.0)

        # per-foot stance set (active) + swing set (inactive); references retargeted each cycle
        velw = np.array([cfg.swingfoot_cost_weights[2], cfg.swingfoot_cost_weights[3], 0.,
                         cfg.swingfoot_cost_weights[4], cfg.swingfoot_cost_weights[5],
                         cfg.swingfoot_cost_weights[6]], float)
        for i, fid in enumerate(self.fids):
            # U6: min normal force (keeps the stance foot loaded; OCS2 cone has a normal-force floor).
            # U7: friction cone uses the RELAXED interior barrier (mu/delta), not QuadraticBarrier+weight.
            _nf = float(cfg.friction_min_nforce) if _U6 else 0.0
            fc = crocoddyl.FrictionCone(np.eye(3), float(cfg.friction_mu), 4, False, _nf)
            cop = crocoddyl.CoPSupport(np.eye(3), self._box * float(cfg.cop_margin_scale))
            if _U67 == "relaxed":
                _fric_act = _relbar(fc.lb, fc.ub, cfg.friction_barrier_mu, cfg.friction_barrier_delta)
                _cop_act = _relbar(cop.lb, cop.ub, cfg.cop_barrier_mu, cfg.cop_barrier_delta)
                _fric_w, _cop_w = 1.0, 1.0
            else:  # "quad": M0's bounded-Hessian QuadraticBarrier + the old misused scalar weights
                _fric_act, _fric_w = _qbar(fc.lb, fc.ub), float(cfg.friction_cone_reg)
                _cop_act, _cop_w = _qbar(cop.lb, cop.ub), float(cfg.cop_weight)
            add("friction%d" % i, _fric_act,
                crocoddyl.ResidualModelContactFrictionCone(self.state, fid, fc, nu, False), _fric_w)
            add("cop%d" % i, _cop_act,
                crocoddyl.ResidualModelContactCoPPosition(self.state, fid, cop, nu, False), _cop_w)
            # U3: stance foot held by the HARD PerComponentContact6D ONLY (OCS2). Soft sz/sflat removed
            # -- they re-anchored the foot to its planted pose each cycle and fought the contact's
            # Baumgarte at touchdown (a source of the trunk dip).
            # U4: swing-foot orientation penalizes SOLE TILT only (roll x, pitch y), NOT yaw (z): OCS2
            # frees foot yaw (Q[trunk_yaw]=0); pinning yaw to world-identity fought turn commands.
            add("swflat%d" % i, _wq([0., 0., 0., 1., 1., 0.]),
                crocoddyl.ResidualModelFramePlacement(self.state, fid, pin.SE3(np.eye(3), planted[fid].translation), nu),
                float(cfg.swingfoot_cost_weights[0]))
            add("swvel%d" % i, _wq(velw),
                crocoddyl.ResidualModelFrameVelocity(self.state, fid, pin.Motion.Zero(), _LWA, nu), 1.0)
            for s in ("swflat%d" % i, "swvel%d" % i):
                costs.changeCostStatus(s, False)
        # Hard, force-free swing-z EQUALITY = t1_controller SwingLegVerticalConstraint (accel-level
        # Baumgarte, MpcPreComputation.cpp:91-104). Built inactive; update() activates it on the swing
        # foot and sets its (z, zdot, zddot) reference. The accel term (ka>0) keeps the input Jacobian
        # full-rank so SolverIntro projects it hard -- a position/velocity equality would be state-only
        # (Hu=0) and segfault. The swing foot's contact is deactivated, so it carries zero force.
        cons = crocoddyl.ConstraintModelManager(self.state, nu)
        csz = {}
        for i in range(len(self.fids)):
            res = SwingZResidual(self.state, nu, self._swingz_fns[i], self.nq, self.nv)
            cons.addConstraint("swz%d" % i, crocoddyl.ConstraintModelResidual(self.state, res))
            cons.changeConstraintStatus("swz%d" % i, False)
            csz[i] = res
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contacts, costs, cons)
        # OCS2 uses RK4; tested IntegratedActionModelRK(four) here -- it did NOT fix the lunge
        # (1.45s->1.25s) and is ~4x slower, so kept Euler. RK4 remains a known faithfulness gap.
        return crocoddyl.IntegratedActionModelEuler(dam, self.dt), contacts, costs, ch, rh, cons, csz

    def _build(self):
        x0 = self._nominal()
        planted = self._planted(x0)
        running = []
        for _ in range(self.N):
            iam, con, cos, ch, rh, cns, csz = self._build_node(x0, planted)
            running.append(iam); self._con.append(con); self._cos.append(cos); self._ch.append(ch)
            self._rh.append(rh); self._cns.append(cns); self._csz.append(csz)
        iam, con, cos, ch, rh, cns, csz = self._build_node(x0, planted, terminal=True)
        self._con.append(con); self._cos.append(cos); self._ch.append(ch); self._rh.append(rh)
        self._cns.append(cns); self._csz.append(csz)
        return crocoddyl.ShootingProblem(x0, running, iam)

    # ---- per-cycle in-place update -----------------------------------------
    def update(self, x0_66, t_gait, comm_filt, gait, x_meas_68):
        """Retarget references and toggle stance/swing per node for the current state/gait phase.

        OCS2-equivalent of re-linearising primalSolution_ around the new initial state + mode
        schedule (no rebuild). Contact placements hold the current foot positions (kp=0 means no
        position pull; the active contact still pins zero foot acceleration = emergent landing)."""
        x66 = np.asarray(x0_66, float)
        self.problem.x0 = x66
        node_times = t_gait + np.arange(self.N + 1) * self.dt
        x_ref = reference_wb.build_reference_66(np.asarray(x_meas_68, float), comm_filt, gait,
                                                t_gait, node_times, self.cfg, self.wb)
        planted = self._planted(x66)
        for k in range(self.N + 1):
            rh = self._rh[k]
            rh["xreg"].reference = np.asarray(x_ref[k], float)
            if k == self.N:                              # terminal: state-only, both contacts stay active
                continue
            flags = gait.contact_flags(float(node_times[k]))
            con, cos, cns = self._con[k], self._cos[k], self._cns[k]
            # OCS2 weightCompensatingInput: regularize the control toward the WEIGHT-SUPPORTING input
            # (stance feet sharing m*g vertically, swing feet + accels zero), NOT toward zero. Zero-ref
            # made the MPC under-load the stance foot at the DS->single-support transition (fz dipped
            # below m*g) -> the trunk sank. The contact force layout is u=[a(nv), f_c0(6), f_c1(6)];
            # fz of contact i is u[nv + 6i + 2].
            nst = int(flags[0]) + int(flags[1])
            u_ref = np.zeros(self.nu)
            if nst:
                for j in range(len(self.fids)):
                    if flags[j]:
                        u_ref[self.nv + 6 * j + 2] = self._mg / nst
            rh["ureg"].reference = u_ref
            for i, fid in enumerate(self.fids):
                stance = bool(flags[i])
                self._ch[k][i].reference = planted[fid]
                con.changeContactStatus("c%d" % i, stance)
                for nm in ("friction%d" % i, "cop%d" % i):
                    cos.changeCostStatus(nm, stance)
                for nm in ("swflat%d" % i, "swvel%d" % i):
                    cos.changeCostStatus(nm, not stance)
                cns.changeConstraintStatus("swz%d" % i, not stance)   # hard swing-z only while swinging
                if not stance:
                    z, vz, az = gait.swing_z(float(node_times[k]), i)
                    # OCS2 SwingLegVerticalConstraint reference (p_z, v_z, a_z); XY emergent, z absolute
                    # world target (flat ground -> liftoff foot z ~= 0).
                    self._csz[k][i].ref = np.array([z, vz, az], float)
                    rh["swflat%d" % i].reference = pin.SE3(np.eye(3), planted[fid].translation)
                    # OCS2 EndEffectorDynamicsFootCost scales the ENTIRE swing task-space tracking by a
                    # cubic impactProximityFactor (1 at liftoff/touchdown, ~0.005 mid-swing) so the foot
                    # is free to fly forward and reach the next step. Constant damping -> short steps ->
                    # foot lands behind the CoM -> forward lunge. Apply the same per-node scaling.
                    ip = float(gait.impact_proximity(float(node_times[k]), i))
                    cos.costs["swflat%d" % i].weight = float(self.cfg.swingfoot_cost_weights[0]) * ip
                    cos.costs["swvel%d" % i].weight = ip
