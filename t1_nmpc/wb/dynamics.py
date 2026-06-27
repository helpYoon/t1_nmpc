"""cpin symbolic dynamics: FreeFlyer state ops + RNEA over the 8 corner frames."""
from __future__ import annotations

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin


class WBDynamics:
    def __init__(self, model: pin.Model, ee_frames):
        self.cmodel = cpin.Model(model)
        self.cdata = self.cmodel.createData()
        self.nq = self.cmodel.nq
        self.nv = self.cmodel.nv
        self.nj = self.nq - 7
        self.ee_frames = tuple(ee_frames)
        self.nf = 3 * len(self.ee_frames)

    def state_integrate(self) -> ca.Function:
        x = ca.SX.sym("x", self.nq + self.nv)
        dx = ca.SX.sym("dx", self.nv + self.nv)
        q_next = cpin.integrate(self.cmodel, x[:self.nq], dx[:self.nv])
        v_next = x[self.nq:] + dx[self.nv:]
        return ca.Function("integrate", [x, dx], [ca.vertcat(q_next, v_next)])

    def state_difference(self) -> ca.Function:
        x0 = ca.SX.sym("x0", self.nq + self.nv)
        x1 = ca.SX.sym("x1", self.nq + self.nv)
        dq = cpin.difference(self.cmodel, x0[:self.nq], x1[:self.nq])
        dv = x1[self.nq:] - x0[self.nq:]
        return ca.Function("difference", [x0, x1], [ca.vertcat(dq, dv)])

    def rnea_dynamics(self) -> ca.Function:
        q = ca.SX.sym("q", self.nq); v = ca.SX.sym("v", self.nv)
        a = ca.SX.sym("a", self.nv); forces = ca.SX.sym("forces", self.nf)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.cmodel.njoints)]
        for idx, fid in enumerate(self.ee_frames):
            jid = self.cmodel.frames[fid].parentJoint
            trans = self.cmodel.frames[fid].placement.translation
            R_w2j = self.cdata.oMi[jid].rotation.T
            f_lin = R_w2j @ forces[idx*3:(idx+1)*3]
            f_ang = ca.cross(trans, f_lin)
            f_ext[jid] = cpin.Force(f_ext[jid].vector + ca.vertcat(f_lin, f_ang))  # ACCUMULATE
        tau = cpin.rnea(self.cmodel, self.cdata, q, v, a, f_ext)
        return ca.Function("rnea_dyn", [q, v, a, forces], [tau])

    def frame_velocity(self, fid: int) -> ca.Function:
        q = ca.SX.sym("q", self.nq); v = ca.SX.sym("v", self.nv)
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        vel = cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).vector
        return ca.Function(f"vel_{fid}", [q, v], [vel])

    def frame_position(self, fid: int) -> ca.Function:
        q = ca.SX.sym("q", self.nq)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        pos = self.cdata.oMf[fid].translation
        return ca.Function(f"pos_{fid}", [q], [pos])
