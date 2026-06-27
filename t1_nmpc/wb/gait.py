"""Biped contact scheduling: walk (cycle 1.4 s, t1_controller gait.info) + stand."""
from __future__ import annotations

from ..robot.config import MPCConfig

FootMode = tuple  # (lf_contact: bool, rf_contact: bool)


def v_z_ref(phase: float, cfg: MPCConfig) -> float:
    """Time-derivative of a cubic swing-height spline: +liftoff -> 0 at apex -> -touchdown.
    Implemented as a symmetric cubic in [0,1] whose derivative is zero at phase=0.5."""
    p = min(max(phase, 0.0), 1.0)
    # height h(p) = swing_height * (3p^2 - 2p^3) blended for up/down would not return to 0;
    # use a velocity profile directly: v(p) = A * (1 - 2p) * 6 ... choose simple shape:
    # v(0)=+v_liftoff, v(0.5)=0, v(1)=-v_touchdown (touchdown stored negative).
    if p <= 0.5:
        s = p / 0.5
        return cfg.v_liftoff * (1.0 - s)            # linear rise->0; (cubic optional refinement)
    s = (p - 0.5) / 0.5
    return cfg.v_touchdown * s                       # 0 -> v_touchdown (negative)


class StandGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    def mode_at(self, t: float) -> FootMode:
        return (True, True)

    def swing_phase(self, t: float, foot_index: int):
        return None

    def horizon_modes(self, t0: float):
        return [(True, True)] * self.cfg.nodes


class WalkGait:
    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        # switching_times = (0, 0.6, 0.7, 1.3, 1.4): LF, double, RF, double
        self.t_lf_end = cfg.switching_times[1]      # 0.6
        self.t_d1_end = cfg.switching_times[2]      # 0.7
        self.t_rf_end = cfg.switching_times[3]      # 1.3
        self.cycle = cfg.gait_cycle                 # 1.4

    def _phase_time(self, t: float) -> float:
        return t % self.cycle

    def mode_at(self, t: float) -> FootMode:
        tp = self._phase_time(t)
        if tp < self.t_lf_end:
            return (False, True)                    # LF swing
        if tp < self.t_d1_end:
            return (True, True)
        if tp < self.t_rf_end:
            return (True, False)                    # RF swing
        return (True, True)

    def swing_phase(self, t: float, foot_index: int):
        tp = self._phase_time(t)
        if foot_index == 0 and tp < self.t_lf_end:
            return tp / self.t_lf_end
        if foot_index == 1 and self.t_d1_end <= tp < self.t_rf_end:
            return (tp - self.t_d1_end) / (self.t_rf_end - self.t_d1_end)
        return None

    def horizon_modes(self, t0: float):
        return [self.mode_at(t0 + i * self.cfg.dt) for i in range(self.cfg.nodes)]
