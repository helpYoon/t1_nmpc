"""Solver-agnostic MPC result container (was in the now-deleted t1_nmpc.acados_mpc.mpc)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MPCResult:
    x_traj: np.ndarray        # (N+1, nx)
    u_traj: np.ndarray        # (N,   nu)
    feasible: bool
    solve_time: float         # seconds (full step)
    mode_schedule: object
    status: int               # acados solve status (0 == success)
    node_times: np.ndarray = None  # (N+1,) absolute wall-clock times of each state node; None = uniform cfg.dt
    u_phys_traj: np.ndarray | None = None    # physical (projected) inputs P@u+Q@x+u_p for execution
