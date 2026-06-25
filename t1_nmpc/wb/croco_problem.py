# t1_nmpc/wb/croco_problem.py
"""T1ProblemBuilder: per-node ContactInvDynamics action-model factory + stand problem.
Construction only (no solver, no mutable state). Bodies derive from the validated
spikes/croco_stand_spike.py."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import crocoddyl

from .croco_costs import build_costs
from . import reference_wb

_LWA = pin.LOCAL_WORLD_ALIGNED


class T1ProblemBuilder:
    def __init__(self, cfg, wb):
        self.cfg = cfg
        self.wb = wb
        self.model = wb.model
        self.nv = wb.model.nv
        self.state = crocoddyl.StateMultibody(wb.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.foot_fids = list(wb.contact_fids)            # [L, R]
        self.dt = float(cfg.dt)
        self.N = int(cfg.N)
        # foot-constraint Baumgarte gains = t1_controller foot_constraint feedback
        self._gains = np.array([cfg.foot_pos_err_gain_z, cfg.foot_linvel_err_gain_xy], float)
        # walk contact gains: kp=0 for emergent landing (no position correction on contact switch)
        self._walk_gains = np.array([0.0, cfg.foot_linvel_err_gain_xy], float)

    def _planted(self, x0_66):
        """SE3 placement of each foot at x0 (held by the contact).

        Returns the full SE3 (position AND orientation) of each foot frame; ContactModel6D
        constrains both translation and rotation, so the complete placement is required."""
        q = np.asarray(x0_66[:self.model.nq], float)
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        return {fid: data.oMf[fid].copy() for fid in self.foot_fids}

    def make_node(self, stance_fids, x_ref, com_ref, planted, swing=None, gains=None, terminal=False, walk=False):
        nu = self.nv + 6 * len(stance_fids)
        g = self._gains if gains is None else np.asarray(gains, float)
        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        for i, fid in enumerate(stance_fids):
            c6 = crocoddyl.ContactModel6D(self.state, fid, planted[fid], _LWA, nu, g)
            contacts.addContact("%d_c%d" % (i, fid), c6)
        costs = build_costs(self.state, self.actuation, nu, x_ref, com_ref, stance_fids, self.cfg,
                            swing=swing, planted=planted, terminal=terminal, walk=walk)
        dam = crocoddyl.DifferentialActionModelContactInvDynamics(
            self.state, self.actuation, contacts, costs)
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else self.dt)

    def build_walk_problem(self, x0_66, t_gait, comm_filt, gait, x_meas_68):
        x0 = np.asarray(x0_66, float)
        planted = self._planted(x0)
        node_times = t_gait + np.arange(self.N + 1) * self.dt
        x_ref = reference_wb.build_reference_66(x_meas_68, comm_filt, gait, t_gait, node_times, self.cfg, self.wb)
        data = self.model.createData()
        running = []
        for k in range(self.N):
            tk = float(node_times[k])
            flags = gait.contact_flags(tk)                       # (left, right) bool
            stance = [self.foot_fids[i] for i in (0, 1) if flags[i]]
            swing = None
            for i in (0, 1):
                if not flags[i]:                                 # swinging foot i
                    z = gait.swing_z(tk, i)[0]
                    w_z = float(self.cfg.swingfoot_cost_weights[0]) * 1e-1 * gait.impact_proximity(tk, i)
                    # base swing-z weight tunable; seeded here. (impact scaling folded in.)
                    swing = dict(fid=self.foot_fids[i], z=z, w_z=max(w_z, 1e-6))
            pin.centerOfMass(self.model, data, x_ref[k][:self.model.nq])
            running.append(self.make_node(stance, x_ref[k], data.com[0].copy(), planted,
                                          swing=swing, gains=self._walk_gains, walk=True))
        flagsN = gait.contact_flags(float(node_times[self.N]))
        stanceN = [self.foot_fids[i] for i in (0, 1) if flagsN[i]] or self.foot_fids
        pin.centerOfMass(self.model, data, x_ref[self.N][:self.model.nq])
        term = self.make_node(stanceN, x_ref[self.N], data.com[0].copy(), planted,
                              gains=self._walk_gains, terminal=True, walk=True)
        return crocoddyl.ShootingProblem(x0, running, term)

    def build_stand_problem(self, x0_66):
        x0 = np.asarray(x0_66, float)
        planted = self._planted(x0)
        data = self.model.createData()
        pin.centerOfMass(self.model, data, x0[:self.model.nq])
        com0 = data.com[0].copy()
        stance = self.foot_fids
        running = [self.make_node(stance, x0, com0, planted) for _ in range(self.N)]
        terminal = self.make_node(stance, x0, com0, planted, terminal=True)
        return crocoddyl.ShootingProblem(x0, running, terminal)
