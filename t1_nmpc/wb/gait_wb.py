# t1_nmpc/wb/gait_wb.py
"""Walking gait schedule for the WB MPC — faithful port of t1_controller's
humanoid_common_mpc Gait/GaitSchedule + MotionPhaseDefinition (slow_walk).

Mode enum (MotionPhaseDefinition.h:47-52): FLY=0, RF=1, LF=2, STANCE=3.
NOTE the inversion: mode RF means the RIGHT foot is in CONTACT (left swings);
LF means the LEFT foot is in contact (right swings). Contact flags are [left, right].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLY, RF, LF, STANCE = 0, 1, 2, 3

# Swing trajectory constants (task.info:91-101)
SWING_HEIGHT = 0.08          # task.info:93
LIFTOFF_VEL = 0.05           # task.info:91
TOUCHDOWN_VEL = -0.05        # task.info:92
TOUCHDOWN_OFFSET = -0.001    # task.info:94
SWING_TIME_SCALE = 0.4       # task.info:95
_IMPACT_LIFTOFF_VEL = -0.15  # task.info:98
_IMPACT_TOUCHDOWN_VEL = 0.3  # task.info:99
_IMPACT_MIDPOINT = 0.005     # task.info:100


def mode_to_stance(mode: int) -> tuple[bool, bool]:
    """modeNumber2StanceLeg (MotionPhaseDefinition.h:57-84), order [left, right]."""
    return {
        FLY: (False, False),
        RF: (False, True),     # right contact, left swing
        LF: (True, False),     # left contact, right swing
        STANCE: (True, True),
    }[int(mode)]


def _cubic(t, t0, p0, v0, t1, p1, v1):
    """Hermite cubic (CubicSpline.cpp:38-80) -> (pos, vel, acc) at time t in [t0,t1]."""
    dt = t1 - t0
    dp, dv = p1 - p0, v1 - v0
    c0 = p0
    c1 = v0 * dt
    c2 = -(3 * v0 + dv) * dt + 3 * dp
    c3 = (2 * v0 + dv) * dt - 2 * dp
    tn = (t - t0) / dt
    pos = c3 * tn ** 3 + c2 * tn ** 2 + c1 * tn + c0
    vel = (3 * c3 * tn ** 2 + 2 * c2 * tn + c1) / dt
    acc = (6 * c3 * tn + 2 * c2) / dt ** 2
    return pos, vel, acc


def _splinecpg(t, t0, p0, v0, t1, p1, v1, pmid):
    """Two-segment lift->apex(vel 0)->touchdown spline (SplineCpg.cpp:38-62)."""
    tmid = 0.5 * (t0 + t1)
    if t < tmid:
        return _cubic(t, t0, p0, v0, tmid, pmid, 0.0)
    return _cubic(t, tmid, pmid, 0.0, t1, p1, v1)


def _swing_window(gait: Gait, t: float, side: int):
    """Absolute [t_start, t_end] of the swing of `side` containing t, or None if stance.
    Scans the single cycle's mode intervals (no wrap for slow_walk/walk)."""
    dur = gait.duration
    cycle = (t // dur) * dur
    phase = (t - cycle) / dur
    if mode_to_stance(gait.mode_at(t))[side]:
        return None
    edges = np.concatenate(([0.0], gait.event_phases, [1.0])) * dur
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= (phase * dur) < hi and not mode_to_stance(gait.mode_sequence[i])[side]:
            return cycle + lo, cycle + hi
    return None


@dataclass(frozen=True)
class Gait:
    duration: float
    event_phases: np.ndarray      # strictly in (0,1), len = len(mode_sequence)-1
    mode_sequence: np.ndarray     # mode enum ints

    def mode_at(self, t: float) -> int:
        phase = (t / self.duration) % 1.0          # wrapPhase -> [0,1)
        idx = int(np.searchsorted(self.event_phases, phase, side="right"))
        return int(self.mode_sequence[idx])

    def contact_flags(self, t: float) -> tuple[bool, bool]:
        return mode_to_stance(self.mode_at(t))

    def swing_z(self, t: float, side: int) -> tuple[float, float, float]:
        """Vertical swing trajectory (z, zdot, zddot) via 2-segment Hermite cubic.
        Returns (0,0,0) for stance foot. side: 0=left, 1=right."""
        win = _swing_window(self, t, side)
        if win is None:
            return 0.0, 0.0, 0.0
        t0, t1 = win
        scaling = min(1.0, (t1 - t0) / SWING_TIME_SCALE)
        midheight = min(0.0, TOUCHDOWN_OFFSET) + scaling * SWING_HEIGHT
        return _splinecpg(t, t0, 0.0, scaling * LIFTOFF_VEL,
                          t1, TOUCHDOWN_OFFSET, scaling * TOUCHDOWN_VEL, midheight)

    def impact_proximity(self, t: float, side: int) -> float:
        """Impact proximity scaler in [0,1]: ~1 near liftoff/touchdown, ~0 at mid-swing.
        Returns 1.0 for stance foot."""
        win = _swing_window(self, t, side)
        if win is None:
            return 1.0
        t0, t1 = win
        scaling = min(1.0, (t1 - t0) / SWING_TIME_SCALE)
        return _splinecpg(t, t0, 1.0, scaling * _IMPACT_LIFTOFF_VEL,
                          t1, 1.0, scaling * _IMPACT_TOUCHDOWN_VEL, _IMPACT_MIDPOINT)[0]


def _gait_from_template(mode_sequence, switching_times) -> Gait:
    """loadGaitSchedule (GaitSchedule.cpp:187-196): duration=last switch; event_phases =
    interior switches / duration."""
    st = np.asarray(switching_times, dtype=np.float64)
    dur = float(st[-1])
    return Gait(duration=dur, event_phases=st[1:-1] / dur,
               mode_sequence=np.asarray(mode_sequence, dtype=int))


# gait.info:172-189
# NOTE: cadence is load-bearing for solver feasibility. A 1.0s cycle (0.40s single-support) was tried
# 2026-06-24 to cut the lateral rock — it FAILED catastrophically (tilt 1.97, z->0.13, 337 fallbacks,
# QP MINSTEP->NaN). Cause: cycle < horizon (1.085s) makes the horizon wrap a FULL gait cycle, i.e. a
# near-periodic multi-step reference, which is not a feasible QP point for the few-iter reduced-KD
# solver (same "periodic horizon backfired" wall seen on t1_kd). Keep cycle > horizon (single-window).
SLOW_WALK = _gait_from_template([LF, STANCE, RF, STANCE], [0.0, 0.65, 0.85, 1.5, 1.7])
# gait.info:134-151 (faster M1b cadence; not used for the first walk)
WALK = _gait_from_template([LF, STANCE, RF, STANCE], [0.0, 0.6, 0.7, 1.3, 1.4])
# Standing gait: always double-stance, no swing events.
STANCE_GAIT = Gait(duration=0.5, event_phases=np.array([]), mode_sequence=np.array([STANCE]))
