# t1_nmpc/wb/croco_mpc.py
"""CrocoMPC: receding-horizon single-RTI driver. Drop-in for the acados WholeBodyMPC at
the loop interface (.cfg/.model/.reset/.step -> MPCResult/.last_solve_s). Speaks 66-dim
crocoddyl state internally, 68-dim at the boundary; emits an acados-layout u_traj so the
existing execution_wb/joint_torque path is reused."""
from __future__ import annotations

import time
import numpy as np
import pinocchio as pin
import crocoddyl

from ..mpc_result import MPCResult
from .croco_problem import T1ProblemBuilder

_REG = 1e-9


class CrocoMPC:
    def __init__(self, cfg, wb, max_iter: int = 1, gait=None):
        self.cfg = cfg
        self.model = wb                                  # has .joint_torque (used by execution_wb)
        self.max_iter = int(max_iter)
        self.builder = T1ProblemBuilder(cfg, wb)
        self.nv = wb.model.nv
        self.N = int(cfg.N)
        self._x0 = self._nominal66()
        self.problem = self.builder.build_stand_problem(self._x0)
        self.solver = crocoddyl.SolverIntro(self.problem)
        self.solver.setCallbacks([])
        self._xs = [self._x0.copy() for _ in range(self.N + 1)]
        self._us = list(self.problem.quasiStatic([self._x0.copy() for _ in range(self.N)]))
        self._node_times = np.arange(self.N + 1) * float(cfg.dt)
        self.last_solve_s = 0.0
        # Walk mode (gait=None -> M0 stand path unchanged)
        self.gait = gait
        self._t_gait = 0.0
        self._comm = np.array([0., 0., float(cfg.nominal_base_height), 0.])

    def _nominal66(self):
        q0 = pin.neutral(self.builder.model); q0[2] = self.cfg.nominal_base_height
        q0[6:6 + self.cfg.n_joints] = self.cfg.nominal_joint_pos
        return np.concatenate([q0, np.zeros(self.nv)])

    def reset(self, x0_68):
        x66 = np.asarray(x0_68, float)[:66]
        self._x0 = x66.copy()
        self.problem.x0 = x66
        self._xs = [x66.copy() for _ in range(self.N + 1)]
        self._us = list(self.problem.quasiStatic([x66.copy() for _ in range(self.N)]))

    def step(self, x_meas_68, t, command=None) -> MPCResult:
        # Walk mode path (gait is set)
        if self.gait is not None:
            return self._step_walk(x_meas_68, t, command)
        # M0 stand path (unchanged)
        x66 = np.asarray(x_meas_68, float)[:66]
        self.problem.x0 = x66
        # warm-start shift
        xs = self._xs[1:] + [self._xs[-1]]; xs[0] = x66.copy()
        us = self._us[1:] + [self._us[-1]]
        t0 = time.perf_counter()
        self.solver.solve(xs, us, self.max_iter, False, _REG)
        self.last_solve_s = time.perf_counter() - t0
        self._xs = list(self.solver.xs); self._us = list(self.solver.us)

        xs_arr = np.asarray(self.solver.xs)               # (N+1, 66)
        ok = bool(np.all(np.isfinite(xs_arr)) and self.solver.isFeasible)
        x_traj = np.zeros((self.N + 1, 68)); x_traj[:, :66] = xs_arr
        u_traj = self._acados_layout(self.solver.us)
        if not ok:                                        # degrade safely (ZOH-able): flat plan
            # Emits a freeze-in-place plan (flat state, zero forces) rather than replaying the
            # last command; never hit in practice while status==0 (isFeasible + finite xs).
            x_traj[:] = x_traj[0]
            u_traj = np.zeros((self.N, 40))
        return MPCResult(
            x_traj=x_traj, u_traj=u_traj, feasible=ok, solve_time=self.last_solve_s,
            mode_schedule=None, status=0 if ok else 1,
            node_times=t + self._node_times, u_phys_traj=None)

    def _step_walk(self, x_meas_68, t, command=None) -> MPCResult:
        x66 = np.asarray(x_meas_68, float)[:66]
        if command is not None:
            from .reference_wb import filter_command
            self._comm = filter_command(self._comm, command)
        self._t_gait += float(self.cfg.dt)               # advance gait clock one control dt
        prob = self.builder.build_walk_problem(x66, self._t_gait, self._comm, self.gait, np.asarray(x_meas_68, float))
        self.solver = crocoddyl.SolverIntro(prob); self.solver.setCallbacks([])
        # re-quasiStatic: dimension-safe across nu changes (single-support vs double-support)
        us = list(prob.quasiStatic([x66.copy() for _ in range(self.N)]))
        # rollout produces a gap-free (feasible) xs from the quasiStatic us, so we can
        # pass isFeasible=True to the solver and get isFeasible=True after 1 RTI step
        xs = list(prob.rollout(us))                      # N+1 feasible states
        t0 = time.perf_counter(); self.solver.solve(xs, us, self.max_iter, True, _REG)
        self.last_solve_s = time.perf_counter() - t0
        self._xs = list(self.solver.xs); self._us = list(self.solver.us)
        xs_arr = np.asarray(self.solver.xs)
        ok = bool(np.all(np.isfinite(xs_arr)) and self.solver.isFeasible)
        x_traj = np.zeros((self.N + 1, 68)); x_traj[:, :66] = xs_arr
        u_traj = self._acados_layout_walk(self.solver.us, self._t_gait)
        if not ok:
            x_traj[:] = x_traj[0]; u_traj = np.zeros((self.N, 40))
        return MPCResult(x_traj=x_traj, u_traj=u_traj, feasible=ok, solve_time=self.last_solve_s,
                         mode_schedule=None, status=0 if ok else 1, node_times=t + self._node_times, u_phys_traj=None)

    def _acados_layout(self, us):
        """crocoddyl us[k]=[a(nv); forces...] -> [W_l(6); W_r(6); a_joints(27); vdot_s=0]."""
        out = np.zeros((self.N, 40))
        for k, u in enumerate(us):
            u = np.asarray(u, float)
            a = u[:self.nv]
            forces = u[self.nv:]
            out[k, 12:39] = a[6:33]                        # qdd joints
            # double-support contact order in make_node = [L, R]
            if forces.size >= 6:
                out[k, 0:6] = forces[0:6]                  # W_l
            if forces.size >= 12:
                out[k, 6:12] = forces[6:12]                # W_r
        return out

    def _acados_layout_walk(self, us, t_gait):
        """Stance-aware mapping: force order matches make_node's enumerate(stance_fids) (left before right).
        Per node query gait.contact_flags to determine which slots to fill."""
        out = np.zeros((self.N, 40)); dt = float(self.cfg.dt)
        for k, u in enumerate(us):
            u = np.asarray(u, float); a = u[:self.nv]; forces = u[self.nv:]
            out[k, 12:39] = a[6:33]
            flags = self.gait.contact_flags(t_gait + k * dt)      # (left, right)
            stance = [i for i in (0, 1) if flags[i]]              # contact order matches make_node enumerate
            for j, side in enumerate(stance):
                sl = slice(0, 6) if side == 0 else slice(6, 12)   # left->W_l, right->W_r
                out[k, sl] = forces[6*j:6*j+6]
        return out
