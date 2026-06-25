"""Custom crocoddyl activations for the faithful T1 walking costs."""
from __future__ import annotations
import numpy as np
import crocoddyl


class RelaxedBarrier(crocoddyl.ActivationModelAbstract):
    """OCS2 RelaxedBarrierPenalty: per element, h>=0 desired.
    value = -mu*log(h)               for h > delta
          = mu*(0.5*((h-2d)/d)^2 - log(d))   for h <= delta   (quadratic continuation, C1 at h=delta).
    Penalizes constraint VIOLATION (h small/negative); gradient pushes h up."""
    def __init__(self, nr, mu, delta):
        crocoddyl.ActivationModelAbstract.__init__(self, nr)
        self.mu = float(mu); self.delta = float(delta)

    def calc(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        v = np.where(h > d, -mu * np.log(np.maximum(h, 1e-12)),
                     mu * (0.5 * ((h - 2 * d) / d) ** 2 - np.log(d)))
        data.a_value = float(np.sum(v))

    def calcDiff(self, data, r):
        h = np.asarray(r).ravel(); mu, d = self.mu, self.delta
        dv = np.where(h > d, -mu / np.maximum(h, 1e-12), mu * (h - 2 * d) / d ** 2)
        d2 = np.where(h > d, mu / np.maximum(h, 1e-12) ** 2, mu / d ** 2 * np.ones_like(h))
        data.Ar[:] = dv
        data.Arr = np.diag(d2)
