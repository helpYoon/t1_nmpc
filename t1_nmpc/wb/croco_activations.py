# t1_nmpc/wb/croco_activations.py
"""OCS2 RelaxedBarrierPenalty as a crocoddyl activation (U7/U8 — friction/CoP).

The QuadraticBarrier penalizes only OUTSIDE the feasible set (zero gradient inside), so the optimizer
rides the constraint boundary -- in single support that means shedding contact force at the cone/CoP
edge -> trunk dips. OCS2 uses an INTERIOR-POINT relaxed log barrier: repulsion grows as the residual
approaches the bound from INSIDE, keeping the force off the boundary. Per component with a finite
bound, on the margin h = (ub - r) [upper] or (r - lb) [lower]:

    h > delta:  B = -mu*ln(h)                                   (log barrier)
    h <= delta: B = mu/(2 delta^2)*(h-delta)^2 - mu/delta*(h-delta) - mu*ln(delta)   (C2 quadratic ext)

The quadratic extension is the 2nd-order Taylor of the log at h=delta, so value, slope and curvature
are continuous and the penalty stays FINITE for h<=0 (infeasible) -- this is what makes it usable in a
DDP/SQP step. mu, delta from task.info (friction 0.2/5.0, CoP 0.1/0.03).
"""
from __future__ import annotations

import numpy as np
import crocoddyl


class ActivationModelRelaxedBarrier(crocoddyl.ActivationModelAbstract):
    def __init__(self, bounds, mu: float, delta: float):
        lb = np.asarray(bounds.lb, float)
        super().__init__(lb.shape[0])
        self.lb = lb
        self.ub = np.asarray(bounds.ub, float)
        self.mu = float(mu)
        self.delta = float(delta)
        self._fub = np.isfinite(self.ub)
        self._flb = np.isfinite(self.lb)

    def _bar(self, h):
        """value, dB/dh, d2B/dh2 of the relaxed log barrier on margin h (>0 = interior)."""
        mu, d = self.mu, self.delta
        big = h > d
        hs = np.where(big, h, d)                       # avoid log of small/neg in the inactive branch
        val = np.where(big, -mu * np.log(hs),
                       mu / (2 * d * d) * (h - d) ** 2 - mu / d * (h - d) - mu * np.log(d))
        g = np.where(big, -mu / hs, mu / (d * d) * (h - d) - mu / d)
        hh = np.where(big, mu / (hs * hs), mu / (d * d) * np.ones_like(h))
        return val, g, hh

    def calc(self, data, r):
        r = np.asarray(r, float)
        a = 0.0
        if self._fub.any():
            v, _, _ = self._bar(self.ub[self._fub] - r[self._fub]); a += float(v.sum())
        if self._flb.any():
            v, _, _ = self._bar(r[self._flb] - self.lb[self._flb]); a += float(v.sum())
        data.a_value = a

    def calcDiff(self, data, r):
        r = np.asarray(r, float); nr = r.shape[0]
        Ar = np.zeros(nr); Arr = np.zeros(nr)
        if self._fub.any():
            _, g, hh = self._bar(self.ub[self._fub] - r[self._fub])
            Ar[self._fub] += -g; Arr[self._fub] += hh        # dh/dr = -1 -> d/dr = -g, d2/dr2 = hh
        if self._flb.any():
            _, g, hh = self._bar(r[self._flb] - self.lb[self._flb])
            Ar[self._flb] += g; Arr[self._flb] += hh
        np.asarray(data.Ar)[:] = Ar
        A = np.asarray(data.Arr)
        if A.ndim == 2:
            A[:] = np.diag(Arr)
        else:
            A[:] = Arr
