"""Biped contact scheduling. Iteration 1: STAND only (all 8 corners always in contact).
Walking (2 swing groups -> 8 corner flags) is deferred; the interface is shaped for it."""
from __future__ import annotations

import numpy as np


class StandGait:
    def __init__(self, n_corners: int = 8):
        self.n_corners = n_corners

    def contact_schedule(self, t_current: float, dts, nodes: int) -> np.ndarray:
        return np.ones((self.n_corners, nodes), dtype=np.float64)

    def swing_schedule(self, t_current: float, dts, nodes: int) -> np.ndarray:
        return np.zeros((self.n_corners, nodes), dtype=np.float64)
