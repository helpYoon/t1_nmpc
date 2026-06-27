"""WholeBodyMPC: build the WalkOCP once, run warm-started Fatrop each tick.

The single compiled solver_fn takes the per-tick gait schedules, base-velocity command
and footstep targets as ARGUMENTS, so the same function is reused across MPC ticks.
`gait` selects the contact/swing schedule source; default StandGait preserves the M0
all-stance stand path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from ..robot.config import MPCConfig, JointCommand
from ..robot.model import RobotModel
from .ocp import WalkOCP
from .gait import StandGait
from .state import extract_command


@dataclass
class WBResult:
    command: JointCommand
    forces0: np.ndarray
    solve_time: float
    constr_viol: float
    num_iters: int = 0
    node1_x: np.ndarray | None = None     # planned node-1 full state (idealized receding test)
    planned: dict | None = None           # full retracted solution (q/v/a/forces/tau per node)


class WholeBodyMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, gait=None):
        self.cfg = cfg
        self.rm = rm
        self.gait = gait if gait is not None else StandGait(cfg)
        self.base_vx = float(cfg.base_vx_des)
        self.ocp = WalkOCP(cfg, rm)
        self.ocp.set_weights()
        # ONE solver_function (8-arg) built once, reused for reset + every tick.
        # A single max_iter cap is correct: warm-started ticks converge early.
        self._solve = self.ocp.solve_function(max_iter=cfg.fatrop_max_iter)
        self._gdata = self.ocp.g_data()
        self._warm = None                 # last opti.x solution vector (warm start)
        # numeric pinocchio FK for footstep targets (foot-center xy)
        self._fk_data = rm.model.createData()
        self._foot_center_ids = rm.foot_center_frame_ids

    # --- per-tick inputs -------------------------------------------------
    def _foot_center_xy(self, x) -> list[np.ndarray]:
        q = np.asarray(x[: self.cfg.nq], dtype=np.float64)
        pin.framesForwardKinematics(self.rm.model, self._fk_data, q)
        return [self._fk_data.oMf[fid].translation[:2].copy() for fid in self._foot_center_ids]

    def _footstep_targets(self, x_meas) -> np.ndarray:
        """Raibert footstep target per foot, broadcast across the horizon.

        target_xy = stance_xy + 0.5*swing_period*v_des + k*(v_meas - v_des).
        For the in-place/default command (base_vx=0, v_meas~0) this collapses to the
        current foot-center xy, so the (soft) footstep cost is ~inert."""
        v_des = np.array([self.base_vx, 0.0])
        v_meas = np.asarray(x_meas[self.cfg.nq: self.cfg.nq + 2], dtype=np.float64)  # base local lin xy
        offset = 0.5 * self.ocp.swing_period * v_des + self.cfg.footstep_k * (v_meas - v_des)
        centers = self._foot_center_xy(x_meas)
        tgt = np.zeros((2 * self.cfg.n_feet, self.cfg.nodes))
        for f in range(self.cfg.n_feet):
            tgt[2 * f: 2 * f + 2, :] = (centers[f] + offset)[:, None]
        return tgt

    def _sync_params(self, x, contact, swing, footstep) -> None:
        """Mirror the solver_fn arguments into the opti parameters so g_data() /
        opti.value(opti.p) report the matching vector for the CV / retract reads."""
        self.ocp.set_x_init(x)
        self.ocp.set_schedules(contact, swing)
        self.ocp.set_base_vx(self.base_vx)
        self.ocp.set_footstep_targets(footstep)

    # --- API -------------------------------------------------------------
    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        contact, swing = self.gait.schedules(0.0)
        footstep = self._footstep_targets(x0)
        self._sync_params(x0, contact, swing, footstep)
        sol = np.array(self._solve(x0, self.cfg.Q_diag, self.cfg.R_diag, contact, swing,
                                   self.base_vx, footstep, self.ocp.x_initial())).flatten()
        self._warm = sol

    def step(self, x_meas, t: float = 0.0) -> WBResult:
        x = np.asarray(x_meas, dtype=np.float64)
        contact, swing = self.gait.schedules(t)
        footstep = self._footstep_targets(x)
        self._sync_params(x, contact, swing, footstep)
        warm = self._warm if self._warm is not None else self.ocp.x_initial()
        t0 = time.perf_counter()
        sol = np.array(self._solve(x, self.cfg.Q_diag, self.cfg.R_diag, contact, swing,
                                   self.base_vx, footstep, warm)).flatten()
        dt = time.perf_counter() - t0
        self._warm = sol
        g, lbg, ubg = self._gdata(sol, self.ocp.opti.value(self.ocp.opti.p))
        cv = WalkOCP.constr_viol_inf(np.array(g).flatten(), np.array(lbg).flatten(),
                                     np.array(ubg).flatten())
        out = self.ocp.retract(sol)
        node1_x = np.concatenate([out["q_sol"][1], out["v_sol"][1]])
        return WBResult(command=extract_command(out, self.cfg),
                        forces0=np.asarray(out["forces_sol"][0], dtype=np.float64),
                        solve_time=dt, constr_viol=cv, num_iters=0,
                        node1_x=node1_x, planned=out)
