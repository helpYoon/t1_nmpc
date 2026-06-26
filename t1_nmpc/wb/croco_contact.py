# t1_nmpc/wb/croco_contact.py
"""Per-component 6D stance contact = faithful port of t1_controller's getStanceFootConstraint
(MpcInterface.cpp:333-356): a HARD 6D ZeroAccelerationConstraint with a SELECTIVE per-component
Baumgarte law

    foot_classical_accel = -Ax_diag * placement_err - Av_diag * foot_velocity

    Ax_diag = [0, 0, positionErrorGain_z, ori, ori, ori]   = [0,0,100,80,80,80]
    Av_diag = [linvel_xy, linvel_xy, linvel_z, angvel, angvel, angvel] = [20,20,10,20,20,20]

Why this class exists: crocoddyl's ContactModel6D applies a single UNIFORM [kp, kd] to all 6 DoF,
which over-stiffens the xy-position (OCS2 uses 0) and orientation, ill-conditioning the OCP so the
DDP SolverIntro diverges at the faithful pos_z=100 (uniform kp=100 -> plan blows up). The selective
per-component gains condition fine (like OCS2). crocoddyl has no per-component contact, so we
recombine three base ContactModel6D evaluated at unit gains -- gains [0,0] gives the drift, [1,0]
adds the 6D placement error, [0,1] adds the 6D velocity -- then scale each row by Ax/Av. This reuses
crocoddyl's exact contact Jacobian and analytical derivatives (validated: with Ax=kp*1, Av=kd*1 it
reproduces ContactModel6D[kp,kd] to machine precision).
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import crocoddyl

_LWA = pin.LOCAL_WORLD_ALIGNED


class PerComponentContact6D(crocoddyl.ContactModelAbstract):
    def __init__(self, state, fid, pref, nu, Ax_diag, Av_diag):
        super().__init__(state, _LWA, 6, nu)
        self.id = fid
        self.Ax = np.asarray(Ax_diag, float)            # placement-error gains [lin(3), ang(3)]
        self.Av = np.asarray(Av_diag, float)            # velocity gains [lin(3), ang(3)]
        self._d = crocoddyl.ContactModel6D(state, fid, pref, _LWA, nu, np.array([0., 0.]))  # drift
        self._p = crocoddyl.ContactModel6D(state, fid, pref, _LWA, nu, np.array([1., 0.]))  # + pos_err
        self._v = crocoddyl.ContactModel6D(state, fid, pref, _LWA, nu, np.array([0., 1.]))  # + vel

    @property
    def reference(self):
        return self._d.reference

    @reference.setter
    def reference(self, pref):
        self._d.reference = pref
        self._p.reference = pref
        self._v.reference = pref

    def createData(self, collector):
        return self._d.createData(collector)

    def calc(self, data, x):
        self._d.calc(data, x); a0d = np.array(data.a0)
        self._p.calc(data, x); a0p = np.array(data.a0)
        self._v.calc(data, x); a0v = np.array(data.a0)   # last call leaves data.Jc set (same frame)
        np.asarray(data.a0)[:] = a0d + self.Ax * (a0p - a0d) + self.Av * (a0v - a0d)

    def calcDiff(self, data, x):
        self._d.calc(data, x); self._d.calcDiff(data, x); a0d = np.array(data.a0); Dd = np.array(data.da0_dx)
        self._p.calc(data, x); self._p.calcDiff(data, x); a0p = np.array(data.a0); Dp = np.array(data.da0_dx)
        self._v.calc(data, x); self._v.calcDiff(data, x); a0v = np.array(data.a0); Dv = np.array(data.da0_dx)
        np.asarray(data.a0)[:] = a0d + self.Ax * (a0p - a0d) + self.Av * (a0v - a0d)
        np.asarray(data.da0_dx)[:] = Dd + self.Ax[:, None] * (Dp - Dd) + self.Av[:, None] * (Dv - Dd)

    def updateForce(self, data, force):
        self._d.updateForce(data, force)


def stance_gains(cfg):
    """OCS2 getStanceFootConstraint diagonals from config (= t1 task.info foot_constraint)."""
    Ax = np.array([0., 0., cfg.foot_pos_err_gain_z,
                   cfg.foot_ori_err_gain, cfg.foot_ori_err_gain, cfg.foot_ori_err_gain], float)
    Av = np.array([cfg.foot_linvel_err_gain_xy, cfg.foot_linvel_err_gain_xy, cfg.foot_linvel_err_gain_z,
                   cfg.foot_angvel_err_gain, cfg.foot_angvel_err_gain, cfg.foot_angvel_err_gain], float)
    return Ax, Av
