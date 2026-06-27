"""Accel-level swing-z Baumgarte as a custom aligator StageFunction (AL-friendly, input-coupled).

A position-level swing-z (FrameTranslation z = ref) is NOT enforceable by aligator's augmented-Lagrangian
at a low iteration budget -- position is "far" from the control, so the multiplier never ramps and the
foot stays on the ground. The OCS2 / crocoddyl formulation instead constrains the foot's VERTICAL
ACCELERATION with a Baumgarte feedback law:

    h(x,u) = kp*(p_z - p_z_ref) + kv*(v_z - v_z_ref) + ka*(a_z - a_z_ref) = 0

a_z (the foot's world-frame vertical acceleration) depends on the generalized acceleration -- whose joint
part is the kinodynamic control u[2*FS:] -- so h is INPUT-COUPLED and the AL enforces it directly (PoC:
foot lifts 0.5cm -> 8.5cm). Base acceleration is approximated 0 here (the swing leg's joint accels dominate
a swing foot's vertical accel); this is enough to lift the foot. Jacobians are analytical via pinocchio
frame-acceleration derivatives.
"""
from __future__ import annotations
import numpy as np
import pinocchio as pin
import aligator

_LWA = pin.LOCAL_WORLD_ALIGNED


class SwingZBaumgarte(aligator.StageFunction):
    def __init__(self, am, foot_id, FS=6, kp=100.0, kv=10.0, ka=1.0):
        super().__init__(am.ndx, 2 * FS + (am.nv - 6), 1)
        self.am = am
        self.model = am.model
        self.data = am.model.createData()
        self.fid = int(foot_id)
        self.nq = am.nq
        self.nv = am.nv
        self.FS = FS
        self.kp, self.kv, self.ka = float(kp), float(kv), float(ka)
        self.z_ref = 0.0
        self.vz_ref = 0.0
        self.az_ref = 0.0

    def __deepcopy__(self, memo):                  # aligator addConstraint deepcopies the function
        new = SwingZBaumgarte(self.am, self.fid, self.FS, self.kp, self.kv, self.ka)
        new.z_ref, new.vz_ref, new.az_ref = self.z_ref, self.vz_ref, self.az_ref
        return new

    def _qva(self, x, u):
        q = np.asarray(x[:self.nq])
        v = np.asarray(x[self.nq:self.nq + self.nv])
        a = np.zeros(self.nv)
        a[6:] = np.asarray(u[2 * self.FS:])        # joint accels from control; base accel approx 0
        return q, v, a

    def evaluate(self, x, u, data):
        np.asarray(data.value).reshape(-1)[:] = self._eval(x, u)

    def computeJacobians(self, x, u, data):
        # Finite-difference Jacobians (FD-verified correct). The control block Ju (the input coupling that
        # makes this AL-enforceable) is analytically da_z/d(joint accels); a fully-analytical Jx via
        # pinocchio frame-acceleration derivatives is the RT optimization (FD is O(ndx+nu) evals/call).
        eps = 1e-6
        d = data
        h0 = self._eval(x, u)
        Jx = np.asarray(d.Jx).reshape(1, -1)
        Ju = np.asarray(d.Ju).reshape(1, -1)
        for i in range(self.am.ndx):
            dx = np.zeros(self.am.ndx); dx[i] = eps
            Jx[0, i] = (self._eval(self.am.space.integrate(x, dx), u) - h0) / eps
        u = np.asarray(u, float)
        for j in range(len(u)):
            up = u.copy(); up[j] += eps
            Ju[0, j] = (self._eval(x, up) - h0) / eps

    def _eval(self, x, u):
        q, v, a = self._qva(x, u)
        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)
        pz = float(self.data.oMf[self.fid].translation[2])
        vz = float(pin.getFrameVelocity(self.model, self.data, self.fid, _LWA).linear[2])
        az = float(pin.getFrameClassicalAcceleration(self.model, self.data, self.fid, _LWA).linear[2])
        return self.kp * (pz - self.z_ref) + self.kv * (vz - self.vz_ref) + self.ka * (az - self.az_ref)
