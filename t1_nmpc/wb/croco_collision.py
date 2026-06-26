# t1_nmpc/wb/croco_collision.py
"""Foot/knee self-collision avoidance = t1_controller FootCollisionConstraint (dropped in the port).

Without it the swing-foot XY is emergent and the placement/CoP costs pull it toward the support
centre, so the swing foot crosses into the stance foot and TRIPS the robot (verified: foot-foot
distance hits 2r at ~1.76s, exactly when it topples). Soft barrier on the pairwise sphere distances
between the two feet (and the two knees), penalised below 2r + margin. State-only (distance(q)) so it
is a COST, not a hard equality (a state-only equality would be Hu-rank-deficient and crash SolverIntro).
"""
from __future__ import annotations

import casadi as cs
import numpy as np
import crocoddyl

# collision_pts_fun columns: 0:3 = left-foot pts, 3:6 = right-foot pts, 6:8 = ankles, 8:10 = knees
_FOOT_PAIRS = [(a, b) for a in range(3) for b in range(3, 6)]   # 9 left-foot x right-foot
_KNEE_PAIR = (8, 9)


def build_collision_fn(wb):
    q = cs.SX.sym("q", wb.nq)
    P = wb.collision_pts_fun(q)                                  # 3 x 10
    d = cs.vertcat(*[cs.norm_2(P[:, a] - P[:, b]) for a, b in _FOOT_PAIRS],
                   cs.norm_2(P[:, _KNEE_PAIR[0]] - P[:, _KNEE_PAIR[1]]))
    return cs.Function("coll", [q], [d, cs.jacobian(d, q)])


def collision_lb(fr_foot, fr_knee, margin):
    """Lower bound per pair = 2*radius + margin (repel before contact)."""
    return np.array([2 * fr_foot + margin] * len(_FOOT_PAIRS) + [2 * fr_knee + margin], float)


class CollisionResidual(crocoddyl.ResidualModelAbstract):
    """r = pairwise sphere-centre distances (10,). q-dependent only."""

    def __init__(self, state, nu, fn, nq, nv):
        super().__init__(state, len(_FOOT_PAIRS) + 1, nu, True, False, False)  # nr, q-dep only
        self._fn = fn
        self._nq = nq
        self._nv = nv

    def calc(self, data, x, u=None):
        d, _ = self._fn(np.asarray(x[:self._nq], float))
        np.asarray(data.r)[:] = np.asarray(d).ravel()

    def calcDiff(self, data, x, u=None):
        _, J = self._fn(np.asarray(x[:self._nq], float))
        Rx = data.Rx
        Rx[:, :self._nv] = np.asarray(J)          # dd/dq (euler-base tangent is naive, verified)
        Rx[:, self._nv:] = 0.0                     # no velocity dependence
