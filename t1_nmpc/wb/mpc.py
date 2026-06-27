"""WholeBodyMPC: build the whole_body_rnea OCP once, run warm-started Fatrop each tick."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..robot.config import MPCConfig, JointCommand
from ..robot.model import RobotModel
from .ocp import StandOCP
from .state import extract_command


@dataclass
class WBResult:
    command: JointCommand
    forces0: np.ndarray
    solve_time: float
    constr_viol: float


class WholeBodyMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, uniform_width: bool = False):
        self.cfg = cfg
        self.rm = rm
        self.ocp = StandOCP(cfg, rm, uniform_width=uniform_width)
        self.ocp.set_weights()
        # ONE solver_function built once, reused for reset + every tick (warm param = opti.x).
        # A single max_iter cap is correct: warm-started ticks converge early and stop well under it.
        self._solve = self.ocp.solve_function(max_iter=cfg.fatrop_max_iter)
        self._gdata = self.ocp.g_data()
        self._warm = None        # last opti.x solution vector (warm start)

    def reset(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        self.ocp.set_x_init(x0)
        sol = np.array(self._solve(x0, self.cfg.Q_diag, self.cfg.R_diag,
                                   self.ocp.x_initial())).flatten()
        self._warm = sol

    def step(self, x_meas) -> WBResult:
        x = np.asarray(x_meas, dtype=np.float64)
        self.ocp.set_x_init(x)
        warm = self._warm if self._warm is not None else self.ocp.x_initial()
        t0 = time.perf_counter()
        sol = np.array(self._solve(x, self.cfg.Q_diag, self.cfg.R_diag, warm)).flatten()
        dt = time.perf_counter() - t0
        self._warm = sol
        g, lbg, ubg = self._gdata(sol, self.ocp.opti.value(self.ocp.opti.p))
        cv = StandOCP.constr_viol_inf(np.array(g).flatten(), np.array(lbg).flatten(),
                                      np.array(ubg).flatten())
        out = self.ocp.retract(sol)
        return WBResult(command=extract_command(out, self.cfg),
                        forces0=np.asarray(out["forces_sol"][0], dtype=np.float64),
                        solve_time=dt, constr_viol=cv)
