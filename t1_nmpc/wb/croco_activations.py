"""Custom crocoddyl activations for the faithful T1 walking costs."""
from __future__ import annotations
import numpy as np
import crocoddyl


class RelaxedBarrier(crocoddyl.ActivationModelAbstract):
    """OCS2 RelaxedBarrierPenalty on a BOUNDED residual ``lb <= r <= ub``.

    crocoddyl cone residuals (FrictionCone, CoPSupport, WrenchCone) encode each
    inequality as a one-sided bound on the residual ``r`` -- e.g. a friction-pyramid
    row is feasible for ``r <= 0`` (ub=0, lb=-inf) and the normal-force row for
    ``r >= 0`` (lb=0, ub=+inf) -- NOT as ``h = r >= 0``. We convert ``r`` to the OCS2
    feasibility margin ``h >= 0`` per row, using whichever bound is finite::

        ub finite -> h = ub - r   (feasible r <= ub),  dh/dr = -1
        lb finite -> h = r - lb   (feasible r >= lb),  dh/dr = +1

    then apply the relaxed-barrier shape (penalise h small/negative, push r inward)::

        value = -mu*log(h)                              for h > delta
              = mu*(0.5*((h-2d)/d)^2 - 0.5 - log(d))    for h <= delta   (C2 at h=delta)

    Applying the barrier to the raw residual ``r`` (the previous behaviour) treated the
    FEASIBLE region of every ub=0 cone row as a massive violation, producing ~1e3
    gradients at a healthy operating point and an ill-conditioned OCP.
    """

    def __init__(self, lb, ub, mu, delta):
        lb = np.asarray(lb, float).ravel()
        ub = np.asarray(ub, float).ravel()
        assert lb.shape == ub.shape, "lb/ub shape mismatch"
        crocoddyl.ActivationModelAbstract.__init__(self, len(lb))
        self.lb = lb
        self.ub = ub
        self.mu = float(mu)
        self.delta = float(delta)
        self._ub_fin = np.isfinite(ub)                    # else lb is the finite (active) bound
        self._sign = np.where(self._ub_fin, -1.0, 1.0)    # dh/dr per row

    def _margin(self, r):
        """Per-row feasibility margin h>=0 (h=0 at the active bound)."""
        r = np.asarray(r).ravel()
        return np.where(self._ub_fin, self.ub - r, r - self.lb)

    def calc(self, data, r):
        h = self._margin(r); mu, d = self.mu, self.delta
        v = np.where(h > d, -mu * np.log(np.maximum(h, 1e-12)),
                     mu * (0.5 * ((h - 2 * d) / d) ** 2 - 0.5 - np.log(d)))
        data.a_value = float(np.sum(v))

    def calcDiff(self, data, r):
        h = self._margin(r); mu, d = self.mu, self.delta
        dvdh = np.where(h > d, -mu / np.maximum(h, 1e-12), mu * (h - 2 * d) / d ** 2)
        d2 = np.where(h > d, mu / np.maximum(h, 1e-12) ** 2, mu / d ** 2 * np.ones_like(h))
        data.Ar[:] = dvdh * self._sign        # chain rule: dv/dr = v'(h) * dh/dr
        data.Arr = np.diag(d2)                 # d2v/dr2 = v''(h) * (dh/dr)^2, (dh/dr)^2 = 1
