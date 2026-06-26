# t1_nmpc/wb/croco_swingz.py
"""Faithful port of t1_controller's SwingLegVerticalConstraint as a crocoddyl residual.

t1_controller (humanoid_nmpc) enforces the swing foot's VERTICAL trajectory as a HARD
state-input EQUALITY at the ACCELERATION level (MpcPreComputation.cpp:91-104,
EndEffectorDynamicsLinearAccConstraint.cpp):

    h = kp*(p_z - p_z_ref) + kv*(v_z - v_z_ref) + ka*(a_z - a_z_ref) = 0

where p_z / v_z / a_z are the foot's world (LOCAL_WORLD_ALIGNED) z position / linear
velocity / classical linear acceleration, and (p_z_ref, v_z_ref, a_z_ref) come from the
swing-trajectory spline. Gains kp=positionErrorGain_z=100, kv=linVelErrorGain_z=10,
ka=linAccErrorGain_z=1 (t1_mpc task.info foot_constraint).

Why this is the faithful AND solver-compatible form (vs the position/velocity costs we used):
  In crocoddyl's DifferentialActionModelContactInvDynamics the state is x=[q,v] and the
  control is u=[a (generalized accel, nv), f (contact forces)]. A foot POSITION or VELOCITY
  equality is a function of (q,v) only -> its input Jacobian Hu=0 -> SolverIntro's nullspace
  resolution cannot project it AND crocoddyl SEGFAULTS on the rank-deficient Hu. The
  ACCELERATION term a_z = J(q)*a + Jdot(q,v)*v depends on the input a=u[:nv], so the full
  Baumgarte residual has Hu != 0 (the ka*da_z/da row) and is projected hard by SolverIntro --
  exactly as OCS2 projects it via projectStateInputEqualityConstraints. ka>0 is REQUIRED
  (it is what keeps Hu full-rank); OCS2 uses ka=1.
"""
from __future__ import annotations

import casadi as cs
import numpy as np
import crocoddyl


def build_swingz_fn(wb, idx: int, kp: float, kv: float, ka: float) -> cs.Function:
    """CasADi f(q, v, a, ref3) -> (r, dr/dq, dr/dv, dr/da) for foot `idx` (0=L, 1=R).

    Reuses wb.foot_kin_fun[idx] (LWA twist/accel/pose). ref3 = [p_z_ref, v_z_ref, a_z_ref]."""
    q = cs.SX.sym("q", wb.nq)
    v = cs.SX.sym("v", wb.nv)
    a = cs.SX.sym("a", wb.nv)
    ref = cs.SX.sym("ref", 3)
    twist, accel, pose = wb.foot_kin_fun[idx](q, v, a)   # twist/accel = [lin(3), ang(3)], pose=[pos(3), orient(3)]
    kin = cs.vertcat(pose[2], twist[2], accel[2])        # [p_z, v_z, a_z] world (LWA)
    gains = cs.DM([kp, kv, ka])
    r = cs.dot(gains, kin - ref)
    return cs.Function("swingz%d" % idx, [q, v, a, ref],
                       [r, cs.jacobian(r, q), cs.jacobian(r, v), cs.jacobian(r, a)])


class SwingZResidual(crocoddyl.ResidualModelAbstract):
    """1-D foot vertical Baumgarte residual (the t1_controller SwingLegVerticalConstraint).

    Added to a node's ConstraintModelManager as an EQUALITY (nh) constraint and projected by
    SolverIntro. `ref` (= [p_z_ref, v_z_ref, a_z_ref] from gait.swing_z) is mutated per cycle."""

    def __init__(self, state, nu: int, fn: cs.Function, nq: int, nv: int):
        super().__init__(state, 1, nu, True, True, True)   # nr=1; q-, v-, u-dependent
        self._fn = fn
        self._nq = nq
        self._nv = nv
        self.ref = np.zeros(3)

    def _qva(self, x, u):
        q = np.asarray(x[:self._nq], float)
        v = np.asarray(x[self._nq:self._nq + self._nv], float)
        a = np.zeros(self._nv) if u is None else np.asarray(u[:self._nv], float)
        return q, v, a

    def calc(self, data, x, u=None):
        q, v, a = self._qva(x, u)
        data.r[0] = float(self._fn(q, v, a, self.ref)[0])

    def calcDiff(self, data, x, u=None):
        q, v, a = self._qva(x, u)
        _, Rq, Rv, Ra = self._fn(q, v, a, self.ref)
        Rx = data.Rx                       # (1, ndx) or squeezed (ndx,) when nr==1
        Rx2 = Rx if Rx.ndim == 1 else Rx[0]
        Rx2[:self._nv] = np.asarray(Rq).ravel()
        Rx2[self._nv:] = np.asarray(Rv).ravel()
        if u is not None:
            Ru = data.Ru
            Ru2 = Ru if Ru.ndim == 1 else Ru[0]
            Ru2[:self._nv] = np.asarray(Ra).ravel()
