"""Biped contact scheduling for the Fatrop whole_body_rnea walk (cycle 1.4s, t1_controller)."""
from __future__ import annotations
import numpy as np
from ..robot.config import MPCConfig


class WalkGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        self.cycle = cfg.gait_cycle
        self.t_lf_end, self.t_d1_end, self.t_rf_end = cfg.switching_times[1:4]  # 0.6,0.7,1.3
        self.swing_period = self.t_lf_end                                        # 0.6

    def mode_at(self, t: float):
        tp = t % self.cycle
        if tp < self.t_lf_end:   return (False, True)    # LF swing
        if tp < self.t_d1_end:   return (True, True)
        if tp < self.t_rf_end:   return (True, False)    # RF swing
        return (True, True)

    def _swing_phase(self, t: float, foot: int):
        tp = t % self.cycle
        if foot == 0 and tp < self.t_lf_end:
            return tp / self.swing_period
        if foot == 1 and self.t_d1_end <= tp < self.t_rf_end:
            return (tp - self.t_d1_end) / self.swing_period
        return None

    def schedules(self, t0: float):
        N = self.cfg.nodes
        contact = np.zeros((2, N)); swing = np.zeros((2, N))
        for i in range(N):
            t = t0 + i * self.cfg.dt_min
            m = self.mode_at(t)
            for f in (0, 1):
                contact[f, i] = 1.0 if m[f] else 0.0
                ph = self._swing_phase(t, f)
                if ph is not None:
                    swing[f, i] = ph
        return contact, swing


class StandGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    def schedules(self, t0: float = 0.0):
        N = self.cfg.nodes
        return np.ones((2, N)), np.zeros((2, N))
