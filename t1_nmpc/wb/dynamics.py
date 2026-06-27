"""cpin symbolic primitives for the aligator whole_body_rnea backend.

RNEA residual = RNEA(q, v, a, f_ext(W))[:6] (base underactuation, paper Eq. 5), with one
6D foot wrench W_foot per foot expressed in its sole frame, transformed to the parent ankle
joint by the constant frame placement jMf. All Jacobians use the manifold dx-trick
(see Global Constraints)."""
from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import aligator
from aligator import manifolds, dynamics as ali_dyn

from ..robot.config import MPCConfig
from ..robot.model import RobotModel


class _DoubleIntegratorODE(ali_dyn.ODEAbstract):
    """xdot = (v, a) on MultibodyPhaseSpace. Pure kinematics; physics is the RNEA constraint."""
    def __init__(self, space, nu):
        super().__init__(space, nu)
        self._space, self._nu = space, nu
        self.nv = space.model.nv

    def __deepcopy__(self, memo):
        return _DoubleIntegratorODE(self._space, self._nu)

    def forward(self, x, u, data):
        nv = self.nv
        data.xdot[:nv] = x[self._space.model.nq:self._space.model.nq + nv]   # qdot tangent = v
        data.xdot[nv:] = u[:nv]                                              # vdot = a

    def dForward(self, x, u, data):
        nv = self.nv
        data.Jx[:, :] = 0.0
        data.Jx[:nv, nv:] = np.eye(nv)
        data.Ju[:, :] = 0.0
        data.Ju[nv:, :nv] = np.eye(nv)


class WBDynamics:
    def __init__(self, rm: RobotModel, cfg: MPCConfig):
        self.cmodel = cpin.Model(rm.model)
        self.cdata = self.cmodel.createData()
        self.nq, self.nv = self.cmodel.nq, self.cmodel.nv
        self.ndx, self.nu = cfg.ndx, cfg.nu
        self.space = manifolds.MultibodyPhaseSpace(rm.model)
        # constant sole->joint placements as cpin SE3 (jMf is config-independent)
        self._feet = [(jid, cpin.SE3(jMf)) for (jid, jMf) in rm.foot_joint_placements]
        self.DoubleIntegratorODE = _DoubleIntegratorODE

    def _rnea_expr(self, q, v, a, W):
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.cmodel.njoints)]
        for k, (jid, jMf) in enumerate(self._feet):
            Wk = W[6 * k:6 * k + 6]
            f_ext[jid] = cpin.Force(f_ext[jid].vector + jMf.act(cpin.Force(Wk)).vector)
        return cpin.rnea(self.cmodel, self.cdata, q, v, a, f_ext)

    def rnea_funcs(self, base_only: bool = True):
        x = ca.SX.sym("x", self.nq + self.nv)
        u = ca.SX.sym("u", self.nu)
        dx = ca.SX.sym("dx", self.ndx)
        q = cpin.integrate(self.cmodel, x[:self.nq], dx[:self.nv])
        v = x[self.nq:] + dx[self.nv:]
        a = u[:self.nv]
        W = u[self.nv:]
        tau = self._rnea_expr(q, v, a, W)
        r = tau[:6] if base_only else tau
        val = ca.Function("rnea_val", [x, u, dx], [r])
        Jx = ca.Function("rnea_Jx", [x, u, dx], [ca.jacobian(r, dx)])
        Ju = ca.Function("rnea_Ju", [x, u, dx], [ca.jacobian(r, u)])
        return val, Jx, Ju

    def joint_torque_fn(self):
        x = ca.SX.sym("x", self.nq + self.nv)
        u = ca.SX.sym("u", self.nu)
        q, v = x[:self.nq], x[self.nq:]
        tau = self._rnea_expr(q, v, u[:self.nv], u[self.nv:])
        return ca.Function("joint_tau", [x, u], [tau[6:]])
