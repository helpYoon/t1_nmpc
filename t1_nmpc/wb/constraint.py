"""aligator constraint builders for the whole_body_rnea OCP. Each cites the paper.

Conventions: u = [a(33), W_L(6), W_R(6)]; wrench slice for foot k is u[33+6k : 33+6k+6]
= [f(3), tau(3)] in the sole frame (f_z = surface normal)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import aligator

EQ = aligator.constraints.EqualityConstraintSet
NEG = aligator.constraints.NegativeOrthant

_WRENCH0 = 33  # start index of W in u


class RneaBaseResidual(aligator.StageFunction):
    """RNEA(q,v,a,f_ext(W))[:6] = 0 — floating-base underactuation (paper Eq. 5)."""
    def __init__(self, ndx, nu, funcs):
        super().__init__(ndx, nu, 6)
        self._ndx, self._nu, self._funcs = ndx, nu, funcs
        self._zero = np.zeros(ndx)

    def __deepcopy__(self, memo):
        return RneaBaseResidual(self._ndx, self._nu, self._funcs)  # share compiled funcs

    def evaluate(self, x, u, data):
        data.value[:] = np.asarray(self._funcs[0](x, u, self._zero)).flatten()

    def computeJacobians(self, x, u, data):
        data.Jx[:, :] = np.asarray(self._funcs[1](x, u, self._zero))
        data.Ju[:, :] = np.asarray(self._funcs[2](x, u, self._zero))


class WrenchConeResidual(aligator.StageFunction):
    """Flat-foot contact-wrench cone on W_foot, rows <= 0 (NegativeOrthant).
    friction (paper Eq. 6) + unilateral + CoP + yaw bound (Caron et al. 2015, flat-foot adapt).
    Rows: [ -f_z, f_x^2+f_y^2 - mu^2 f_z^2, |tau_y|-X f_z (x2), |tau_x|-Y f_z (x2),
            |tau_z| - mu(X+Y) f_z (x2) ] -> 8 rows (abs split into +/-)."""
    def __init__(self, ndx, nu, foot_index, mu, X, Y):
        super().__init__(ndx, nu, 8)
        self._ndx, self._nu = ndx, nu
        self._i = _WRENCH0 + 6 * foot_index
        self._mu, self._X, self._Y = mu, X, Y

    def __deepcopy__(self, memo):
        return WrenchConeResidual(self._ndx, self._nu, (self._i - _WRENCH0) // 6,
                                  self._mu, self._X, self._Y)

    def _rows(self, u):
        fx, fy, fz, tx, ty, tz = u[self._i:self._i + 6]
        mu, X, Y = self._mu, self._X, self._Y
        return np.array([
            -fz,
            fx * fx + fy * fy - mu * mu * fz * fz,
            ty - X * fz,  -ty - X * fz,
            tx - Y * fz,  -tx - Y * fz,
            tz - mu * (X + Y) * fz,  -tz - mu * (X + Y) * fz,
        ])

    def evaluate(self, x, u, data):
        data.value[:] = self._rows(u)

    def computeJacobians(self, x, u, data):
        i, mu, X, Y = self._i, self._mu, self._X, self._Y
        fx, fy, fz = u[i], u[i + 1], u[i + 2]
        J = np.zeros((8, self._nu))
        J[0, i + 2] = -1.0
        J[1, i] = 2 * fx; J[1, i + 1] = 2 * fy; J[1, i + 2] = -2 * mu * mu * fz
        J[2, i + 4] = 1.0;  J[2, i + 2] = -X
        J[3, i + 4] = -1.0; J[3, i + 2] = -X
        J[4, i + 3] = 1.0;  J[4, i + 2] = -Y
        J[5, i + 3] = -1.0; J[5, i + 2] = -Y
        J[6, i + 5] = 1.0;  J[6, i + 2] = -mu * (X + Y)
        J[7, i + 5] = -1.0; J[7, i + 2] = -mu * (X + Y)
        data.Jx[:, :] = 0.0
        data.Ju[:, :] = J


class SwingWrenchResidual(aligator.StageFunction):
    """W_foot = 0 for a swing foot (paper §IV-B2)."""
    def __init__(self, ndx, nu, foot_index):
        super().__init__(ndx, nu, 6)
        self._ndx, self._nu = ndx, nu
        self._i = _WRENCH0 + 6 * foot_index
        self._sel = np.zeros((6, nu)); self._sel[np.arange(6), self._i + np.arange(6)] = 1.0

    def __deepcopy__(self, memo):
        return SwingWrenchResidual(self._ndx, self._nu, (self._i - _WRENCH0) // 6)

    def evaluate(self, x, u, data):
        data.value[:] = u[self._i:self._i + 6]

    def computeJacobians(self, x, u, data):
        data.Jx[:, :] = 0.0
        data.Ju[:, :] = self._sel


def contact_velocity_residual(rm, ndx, nu, foot_index):
    """Stance: 6D foot spatial velocity = 0 (paper §IV-B2, per foot; 6D = flat-foot adapt)."""
    fid = rm.sole_frame_ids[foot_index]
    return aligator.FrameVelocityResidual(ndx, nu, rm.model, pin.Motion.Zero(), fid,
                                          pin.LOCAL_WORLD_ALIGNED)


def swing_z_residual(rm, ndx, nu, foot_index):
    """Swing: foot z-velocity = v_z_ref (paper §IV-B2, hard, velocity-level).
    Returns (z_slice_function, base_residual) — caller sets `base.vref = pin.Motion(...)`
    per tick (vref is the non-deprecated reference setter in aligator 0.19.0)."""
    fid = rm.sole_frame_ids[foot_index]
    base = aligator.FrameVelocityResidual(ndx, nu, rm.model, pin.Motion.Zero(), fid,
                                          pin.LOCAL_WORLD_ALIGNED)
    return base[2:3], base
