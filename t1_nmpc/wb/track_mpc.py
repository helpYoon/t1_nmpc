"""TrackingMPC: build PickupOCP once, sample the plan each tick, warm-solve, emit a JointCommand."""
from __future__ import annotations

import time

import numpy as np

from ..robot.config import MPCConfig
from ..robot.model import RobotModel
from .reference import MotionPlanReference
from .track_ocp import PickupOCP
from .state import extract_command
from .mpc import WBResult


class TrackingMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, plan_path: str,
                 x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
        self.cfg, self.rm = cfg, rm
        self.ref = MotionPlanReference(plan_path, cfg, rm, x0=x0, y0=y0, yaw0=yaw0)
        self.ocp = PickupOCP(cfg, rm); self.ocp.set_weights()
        # TWO compiled solvers: reset converges fully; per-tick is RTI-capped (bounded worst case).
        # Validated 2026-06-28: cap=3 falls at the left-release transition (under-converges there);
        # cap>=6 completes the full motion. cold/reset MUST converge (full iters) for a clean start.
        self._solve_cold = self.ocp.solve_function(cfg.fatrop_max_iter)
        self._solve_warm = self.ocp.solve_function(cfg.track_warm_iters)
        self._warm = None
        self.duration_wall = self.ref.duration_phase * cfg.time_scale

    def _call(self, fn, x, xr, hr, gg, warm):
        return np.array(fn(x, self.cfg.Q_diag, self.cfg.R_diag, xr, hr, gg, warm)).flatten()

    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        xr, hr, gg = self.ref.sample(0.0)
        self.ocp.set_refs(x0, xr, hr, gg)
        self._warm = self._call(self._solve_cold, x0, xr, hr, gg, self.ocp.x_initial())

    def step(self, x_meas, t_wall: float) -> WBResult:
        x = np.asarray(x_meas, dtype=np.float64)
        xr, hr, gg = self.ref.sample(t_wall)
        self.ocp.set_refs(x, xr, hr, gg)
        warm = self._warm if self._warm is not None else self.ocp.x_initial()
        t0 = time.perf_counter()
        sol = self._call(self._solve_warm, x, xr, hr, gg, warm)
        dt = time.perf_counter() - t0
        self._warm = sol
        out = self.ocp.retract(sol, x)
        node1_x = np.concatenate([out["q_sol"][1], out["v_sol"][1]])
        return WBResult(command=extract_command(out, self.cfg),
                        forces0=np.asarray(out["forces_sol"][0], dtype=np.float64),
                        solve_time=dt, constr_viol=0.0, num_iters=0,
                        node1_x=node1_x, planned=out)
