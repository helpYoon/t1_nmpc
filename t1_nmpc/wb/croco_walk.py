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

import numpy as np
import pinocchio as pin
import crocoddyl

from . import reference_wb
from .croco_costs import _control_weights

_LWA = pin.LOCAL_WORLD_ALIGNED
_BIG = 1e3  # effective +inf for one-sided state bounds


def _qbar(lb, ub):
    return crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(np.asarray(lb, float), np.asarray(ub, float)))


def _wq(w):
    return crocoddyl.ActivationModelWeightedQuad(np.asarray(w, float))


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
        self._box = np.array([cfg.foot_rect_x[1], cfg.foot_rect_y[1]], float)
        self._gains = np.array([0.0, cfg.foot_linvel_err_gain_xy], float)  # kp=0 emergent landing
        # per-node handles (index 0..N; N is terminal)
        self._con = []     # ContactModelMultiple per node
        self._cos = []     # CostModelSum per node
        self._ch = []      # {foot_idx: ContactModel6D}
        self._rh = []      # {cost_name: ResidualModel}
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
            c = crocoddyl.ContactModel6D(self.state, fid, planted[fid], _LWA, nu, self._gains)
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
            return crocoddyl.IntegratedActionModelEuler(dam, 0.0), contacts, costs, ch, rh

        # common costs
        add("xreg", _wq(cfg.Q[:66]), crocoddyl.ResidualModelState(self.state, x0, nu), 1.0)
        data = self.model.createData(); pin.centerOfMass(self.model, data, x0[:self.nq])
        add("com", crocoddyl.ActivationModelQuad(3),
            crocoddyl.ResidualModelCoMPosition(self.state, data.com[0].copy(), nu), 1.0)
        add("ureg", _wq(_control_weights(self.nv, 12, np.asarray(cfg.R, float))),
            crocoddyl.ResidualModelControl(self.state, nu), 1.0)
        tau = np.asarray(cfg.torque_limit, float)
        add("tau_lim", _qbar(-tau, tau),
            crocoddyl.ResidualModelJointEffort(self.state, self.actuation, np.zeros(self.actuation.nu), nu, False),
            float(cfg.jointtorque_weight))
        lb = np.full(66, -_BIG); ub = np.full(66, _BIG)
        lb[6:6 + cfg.n_joints] = np.asarray(cfg.joint_lower, float)
        ub[6:6 + cfg.n_joints] = np.asarray(cfg.joint_upper, float)
        add("joint_lim", _qbar(lb, ub),
            crocoddyl.ResidualModelState(self.state, np.zeros(self.state.nx), nu), float(cfg.joint_limit_barrier_mu))

        # per-foot stance set (active) + swing set (inactive); references retargeted each cycle
        velw = np.array([cfg.swingfoot_cost_weights[2], cfg.swingfoot_cost_weights[3], 0.,
                         cfg.swingfoot_cost_weights[4], cfg.swingfoot_cost_weights[5],
                         cfg.swingfoot_cost_weights[6]], float)
        for i, fid in enumerate(self.fids):
            fc = crocoddyl.FrictionCone(np.eye(3), float(cfg.friction_mu), 4, False)
            add("friction%d" % i, _qbar(fc.lb, fc.ub),
                crocoddyl.ResidualModelContactFrictionCone(self.state, fid, fc, nu, False), float(cfg.friction_cone_reg))
            cop = crocoddyl.CoPSupport(np.eye(3), self._box)
            add("cop%d" % i, _qbar(cop.lb, cop.ub),
                crocoddyl.ResidualModelContactCoPPosition(self.state, fid, cop, nu, False), 1.0)
            add("sz%d" % i, _wq([0., 0., 1.]),
                crocoddyl.ResidualModelFrameTranslation(self.state, fid, planted[fid].translation, nu),
                float(cfg.foot_pos_err_gain_z))
            add("sflat%d" % i, _wq([0., 0., 0., 1., 1., 1.]),
                crocoddyl.ResidualModelFramePlacement(self.state, fid, planted[fid], nu), float(cfg.foot_ori_err_gain))
            add("swflat%d" % i, _wq([0., 0., 0., 1., 1., 1.]),
                crocoddyl.ResidualModelFramePlacement(self.state, fid, pin.SE3(np.eye(3), planted[fid].translation), nu),
                float(cfg.swingfoot_cost_weights[0]))
            add("swvel%d" % i, _wq(velw),
                crocoddyl.ResidualModelFrameVelocity(self.state, fid, pin.Motion.Zero(), _LWA, nu), 1.0)
            add("swz%d" % i, _wq([0., 0., 1.]),
                crocoddyl.ResidualModelFrameTranslation(self.state, fid, planted[fid].translation, nu),
                float(cfg.swingfoot_z_weight))
            for s in ("swflat%d" % i, "swvel%d" % i, "swz%d" % i):
                costs.changeCostStatus(s, False)
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contacts, costs)
        return crocoddyl.IntegratedActionModelEuler(dam, self.dt), contacts, costs, ch, rh

    def _build(self):
        x0 = self._nominal()
        planted = self._planted(x0)
        running = []
        for _ in range(self.N):
            iam, con, cos, ch, rh = self._build_node(x0, planted)
            running.append(iam); self._con.append(con); self._cos.append(cos); self._ch.append(ch); self._rh.append(rh)
        iam, con, cos, ch, rh = self._build_node(x0, planted, terminal=True)
        self._con.append(con); self._cos.append(cos); self._ch.append(ch); self._rh.append(rh)
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
        data = self.model.createData()
        for k in range(self.N + 1):
            rh = self._rh[k]
            rh["xreg"].reference = np.asarray(x_ref[k], float)
            if "com" in rh:
                pin.centerOfMass(self.model, data, x_ref[k][:self.nq])
                rh["com"].reference = data.com[0].copy()
            if k == self.N:                              # terminal: state-only, both contacts stay active
                continue
            flags = gait.contact_flags(float(node_times[k]))
            con, cos = self._con[k], self._cos[k]
            for i, fid in enumerate(self.fids):
                stance = bool(flags[i])
                self._ch[k][i].reference = planted[fid]
                con.changeContactStatus("c%d" % i, stance)
                for nm in ("friction%d" % i, "cop%d" % i, "sz%d" % i, "sflat%d" % i):
                    cos.changeCostStatus(nm, stance)
                for nm in ("swflat%d" % i, "swvel%d" % i, "swz%d" % i):
                    cos.changeCostStatus(nm, not stance)
                rh["sz%d" % i].reference = planted[fid].translation
                rh["sflat%d" % i].reference = planted[fid]
                if not stance:
                    z = gait.swing_z(float(node_times[k]), i)[0]
                    rh["swz%d" % i].reference = np.array(
                        [planted[fid].translation[0], planted[fid].translation[1], z], float)
                    rh["swflat%d" % i].reference = pin.SE3(np.eye(3), planted[fid].translation)
