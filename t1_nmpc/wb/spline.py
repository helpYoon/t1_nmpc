"""Cubic swing-height spline (z-velocity reference). Ported from wb-mpc-locoman (OCS2 form)."""
from __future__ import annotations
import casadi as ca


class CubicSpline:
    def __init__(self, t0, t1, pos0, vel0, pos1, vel1):
        self.t0, self.t1, self.dt = t0, t1, t1 - t0
        dpos = pos1 - pos0
        dvel = vel1 - vel0
        self.c0 = pos0
        self.c1 = vel0 * self.dt
        self.c2 = -(3.0 * vel0 + dvel) * self.dt + 3.0 * dpos
        self.c3 = (2.0 * vel0 + dvel) * self.dt - 2.0 * dpos

    def velocity(self, t):
        tn = (t - self.t0) / self.dt
        return (3.0 * self.c3 * tn**2 + 2.0 * self.c2 * tn + self.c1) / self.dt


def get_spline_vel_z(swing_phase, swing_period, h_max=0.08, v_liftoff=0.05, v_touchdown=-0.05):
    mid = swing_period / 2.0
    s1 = CubicSpline(0.0, mid, 0.0, v_liftoff, h_max, 0.0)
    s2 = CubicSpline(mid, swing_period, h_max, 0.0, 0.0, v_touchdown)
    return ca.if_else(swing_phase < 0.5,
                      s1.velocity(swing_phase * swing_period),
                      s2.velocity(swing_phase * swing_period))
