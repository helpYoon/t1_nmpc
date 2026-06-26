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
from .croco_walk import WalkOCP

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
        # Walk mode (gait=None -> M0 stand path unchanged). The walk OCP is a persistent
        # both-contacts-every-node problem solved by a persistent solver, warm-started from the
        # previous solution (OCS2-faithful); see croco_walk.WalkOCP.
        self.gait = gait
        self._t_gait = 0.0
        self._comm = np.array([0., 0., float(cfg.nominal_base_height), 0.])
        if gait is not None:
            self._walk_ocp = WalkOCP(cfg, wb)
            self._walk_solver = crocoddyl.SolverIntro(self._walk_ocp.problem)
            self._walk_solver.setCallbacks([])

    def _nominal66(self):
        q0 = pin.neutral(self.builder.model); q0[2] = self.cfg.nominal_base_height
        q0[6:6 + self.cfg.n_joints] = self.cfg.nominal_joint_pos
        return np.concatenate([q0, np.zeros(self.nv)])

    def reset(self, x0_68):
        x66 = np.asarray(x0_68, float)[:66]
        self._x0 = x66.copy()
        prob = self._walk_ocp.problem if self.gait is not None else self.problem
        if self.gait is not None:
            self._comm = np.array([0., 0., float(self.cfg.nominal_base_height), 0.])
        prob.x0 = x66
        self._xs = [x66.copy() for _ in range(self.N + 1)]
        self._us = list(prob.quasiStatic([x66.copy() for _ in range(self.N)]))

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
        t_gait = float(t)                                 # gait clock tracks the real sim time
        self._t_gait = t_gait
        # Mutate the persistent walk OCP in place (retarget references, toggle stance/swing per node)
        # and re-solve the persistent solver warm-started from the previous solution -- the crocoddyl
        # analog of OCS2's coldStart=false primalSolution_ + trajectorySpread (see croco_walk.WalkOCP).
        self._walk_ocp.update(x66, t_gait, self._comm, self.gait, np.asarray(x_meas_68, float))
        us = self._us[1:] + [self._us[-1]]                # receding-horizon shift of the previous solution
        xs = self._xs[1:] + [self._xs[-1]]; xs[0] = x66.copy()
        t0 = time.perf_counter(); self._walk_solver.solve(xs, us, self.max_iter, False, _REG)
        self.last_solve_s = time.perf_counter() - t0
        self._xs = list(self._walk_solver.xs); self._us = list(self._walk_solver.us)
        xs_arr = np.asarray(self._walk_solver.xs)
        # Single-RTI applies the first control regardless of crocoddyl's gap-feasibility flag (one
        # iteration legitimately leaves small dynamic gaps -> isFeasible=False is normal); only a
        # non-finite solve is a genuine failure.
        ok = bool(np.all(np.isfinite(xs_arr)))
        x_traj = np.zeros((self.N + 1, 68)); x_traj[:, :66] = xs_arr
        u_traj = self._acados_layout_walk(self._walk_solver.us, t_gait)
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
        """Fixed both-contacts layout: u forces are always [c0=left(6), c1=right(6)]. Zero the
        wrench of a swing foot (its contact is inactive that node) so execution applies no ground
        reaction for a foot in the air."""
        out = np.zeros((self.N, 40)); dt = float(self.cfg.dt)
        for k, u in enumerate(us):
            u = np.asarray(u, float); a = u[:self.nv]; forces = u[self.nv:]
            out[k, 12:39] = a[6:33]
            flags = self.gait.contact_flags(t_gait + k * dt)      # (left, right)
            if flags[0]:
                out[k, 0:6] = forces[0:6]                         # left contact c0 -> W_l
            if flags[1]:
                out[k, 6:12] = forces[6:12]                       # right contact c1 -> W_r
        return out
