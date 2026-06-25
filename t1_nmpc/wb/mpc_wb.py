"""WholeBodyMPC — closed-loop single-RTI wrapper around the whole-body acados OCP.

M0: STAND (double stance, constant nominal reference).
M1: per-node walking rollout — per-node contact flags + swing-Z + impact proximity
    sampled from the gait schedule at t + k*dt.

Mirrors acados_mpc/mpc.py interface: set_command / reset / step(x_meas, t)->MPCResult.
"""
from __future__ import annotations

import time

import numpy as np

from .config_wb import WBConfig
from .cost_wb import N_PARAM_WB, P_XREF, P_UREF, P_CONTACT, P_SWINGZ, P_IMPACT, P_DT
from .gait_wb import SLOW_WALK, STANCE_GAIT
from .grid_wb import event_aligned_grid
from .reference_wb import build_reference, filter_command
from .ocp_wb import make_ocp, build_solver
from ..mpc_result import MPCResult


def build_node_params(x_meas, node_times, comm_filt, gait, cfg, model, xg, ug, proj_funcs) -> np.ndarray:
    """Per-node acados parameter matrix (N+1, N_PARAM_WB) on the given (possibly non-uniform)
    node_times: folded reference + contact flags + swing-Z + impact + per-stage dt (P_DT)
    + per-node affine projector (P_PROJ_P/Q/UP) linearized at the warm-start (xg, ug both N+1 rows)."""
    from .cost_wb import P_PROJ_P, P_PROJ_Q, P_PROJ_UP
    from .projection_wb import compute_projector
    node_times = np.asarray(node_times, float)
    x_ref, u_ref = build_reference(x_meas, comm_filt, gait, node_times[0], node_times, cfg, model)
    P = np.zeros((cfg.N + 1, N_PARAM_WB))
    for k, tk in enumerate(node_times):
        P[k, P_XREF] = x_ref[k]
        if k < len(u_ref):
            P[k, P_UREF] = u_ref[k]
        lf, rf = gait.contact_flags(tk)
        P[k, P_CONTACT] = [float(lf), float(rf)]
        zL = gait.swing_z(tk, 0); zR = gait.swing_z(tk, 1)
        P[k, P_SWINGZ] = [zL[0], zL[1], zL[2], zR[0], zR[1], zR[2]]
        P[k, P_IMPACT] = [gait.impact_proximity(tk, 0), gait.impact_proximity(tk, 1)]
    dts = np.diff(node_times)
    P[:cfg.N, P_DT] = dts
    P[cfg.N, P_DT] = dts[-1]                  # terminal slot unused by dynamics
    # projector per node, linearized at the warm-start (xg,ug both N+1 rows) -- the SAME point acados linearizes at.
    for k in range(cfg.N + 1):
        Pk, Qk, upk = compute_projector(xg[k], ug[k], P[k], proj_funcs, cfg)
        P[k, P_PROJ_P] = Pk.flatten(order="F")
        P[k, P_PROJ_Q] = Qk.flatten(order="F")
        P[k, P_PROJ_UP] = upk
    return P


def shift_warmstart(x_prev, u_prev, node_times_prev, node_times_now, cfg):
    """Interpolate the previous primal (defined on node_times_prev) onto node_times_now by absolute
    time; hold-last past the previous horizon end. Generalizes the old uniform time-shift to the
    D4 event-aligned (non-uniform, per-tick-varying) grid.

    node_times_prev / node_times_now: 1-D arrays of length N+1 with absolute wall-clock times of
    each state node.
    """
    tp = np.asarray(node_times_prev, float)
    tn = np.asarray(node_times_now, float)
    xg = np.empty((cfg.N + 1, cfg.nx))
    for j in range(cfg.nx):
        xg[:, j] = np.interp(tn, tp, x_prev[:, j])          # np.interp holds-last past tp[-1]
    # u defined on intervals: sample at the START of each new interval from prev interval-starts
    up_t = tp[:cfg.N]
    ug = np.empty((cfg.N, cfg.nu))
    for j in range(cfg.nu):
        ug[:, j] = np.interp(tn[:cfg.N], up_t, u_prev[:, j])
    return xg, ug


class WholeBodyMPC:
    def __init__(self, cfg: WBConfig, model, solver=None, max_iter: int = 1):
        self.cfg = cfg; self.model = model
        from .projection_wb import build_projector_funcs
        self._proj_funcs = build_projector_funcs(cfg, model)
        self.ocp, self.bundle = make_ocp(cfg)                   # DISCRETE raw-u; contact equalities = con_h toggled by per-node bounds
        self.solver = solver if solver is not None else build_solver(self.ocp)
        self.solver.options_set("max_iter", int(max_iter))      # RUNTIME SQP iters (lowers the ceiling baked in ocp_wb);
        # changing this NEVER rebuilds. Default 1 = single-RTI; pass max_iter>1 (<=ceiling) for a convergence probe.
        self._cmd = np.zeros(5)                     # [vx, vy, wz, dheight, dpitch]
        self._gait = STANCE_GAIT
        self._comm_filt = np.array([0.0, 0.0, cfg.nominal_base_height, 0.0])
        self._x_nom = model.nominal_state()
        self._x_prev = self._u_prev = None; self._t_prev = None   # warm-start carry (trajectorySpread)
        self._node_times_prev = None
        self._last_warmstart_x = self._last_warmstart_u = None     # last linearization point (diagnostics)

    def set_command(self, cmd) -> None:
        self._cmd = np.asarray(cmd, dtype=np.float64).copy()
        speed = abs(self._cmd[0]) + abs(self._cmd[1]) + abs(self._cmd[2])
        self._gait = SLOW_WALK if speed > 1e-3 else STANCE_GAIT

    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        self._comm_filt = np.array([0.0, 0.0, self.cfg.nominal_base_height, 0.0])
        for k in range(self.cfg.N + 1):
            self.solver.set(k, "x", x0)
        u0 = np.zeros(self.cfg.nu)
        u0[2] = u0[8] = self.model.total_mass() * 9.81 / 2.0
        for k in range(self.cfg.N):
            self.solver.set(k, "u", u0)
        self._x_prev = self._u_prev = None; self._t_prev = None   # no stale warm-start across a reset
        self._node_times_prev = None

    def step(self, x_meas, t) -> MPCResult:
        cfg = self.cfg; t0 = time.perf_counter()
        x_meas = np.asarray(x_meas, dtype=np.float64)
        comm = np.array([self._cmd[0], self._cmd[1],
                         cfg.nominal_base_height + self._cmd[3], self._cmd[2]])
        # OCS2 bounds the command UPSTREAM (reference.info maxDisplacementVelocity 1.0/0.6, maxRotation
        # 1.0) before the 0.8 EMA; clamp here so out-of-range commands match t1_controller.
        comm[0] = np.clip(comm[0], -cfg.max_vel_x, cfg.max_vel_x)
        comm[1] = np.clip(comm[1], -cfg.max_vel_y, cfg.max_vel_y)
        comm[3] = np.clip(comm[3], -cfg.max_yaw_rate, cfg.max_yaw_rate)
        self._comm_filt = filter_command(self._comm_filt, comm)
        node_times = event_aligned_grid(t, self._gait, cfg)
        if self._x_prev is not None:
            xg, ug = shift_warmstart(self._x_prev, self._u_prev, self._node_times_prev, node_times, cfg)
        else:
            u0 = np.zeros(cfg.nu); u0[2] = u0[8] = self.model.total_mass() * 9.81 / 2.0
            xg = np.tile(x_meas, (cfg.N + 1, 1)); ug = np.tile(u0, (cfg.N, 1))
        ug_full = np.vstack([ug, ug[-1]])                       # N+1 rows for the terminal projector node
        P = build_node_params(x_meas, node_times, self._comm_filt, self._gait, cfg, self.model, xg, ug_full, self._proj_funcs)
        for k in range(cfg.N + 1):
            self.solver.set(k, "x", xg[k])
        for k in range(cfg.N):
            self.solver.set(k, "u", ug[k])
        for k in range(cfg.N + 1):
            self.solver.set(k, "p", P[k])
        self._last_warmstart_x, self._last_warmstart_u = xg, ug
        self.solver.constraints_set(0, "lbx", x_meas)
        self.solver.constraints_set(0, "ubx", x_meas)
        status = self.solver.solve()
        x_traj = np.array([self.solver.get(k, "x") for k in range(cfg.N + 1)])
        u_traj = np.array([self.solver.get(k, "u") for k in range(cfg.N)])
        from .cost_wb import P_PROJ_P, P_PROJ_Q, P_PROJ_UP
        u_phys_traj = np.empty((cfg.N, cfg.nu))
        for k in range(cfg.N):
            Pk = P[k, P_PROJ_P].reshape(cfg.nu, cfg.nu, order="F")
            Qk = P[k, P_PROJ_Q].reshape(cfg.nu, cfg.nx, order="F")
            u_phys_traj[k] = Pk @ u_traj[k] + Qk @ x_traj[k] + P[k, P_PROJ_UP]
        self._x_prev, self._u_prev, self._t_prev = x_traj, u_traj, t
        self._node_times_prev = node_times
        return MPCResult(x_traj=x_traj, u_traj=u_traj, feasible=(status == 0),
                         solve_time=time.perf_counter() - t0, mode_schedule=None, status=int(status),
                         node_times=node_times, u_phys_traj=u_phys_traj)
