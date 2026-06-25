# t1_nmpc/wb/croco_mpc.py
"""CrocoMPC: receding-horizon single-RTI driver. Drop-in for the acados WholeBodyMPC at
the loop interface (.cfg/.model/.reset/.step -> MPCResult/.last_solve_s). Speaks 66-dim
crocoddyl state internally, 68-dim at the boundary; emits an acados-layout u_traj so the
existing execution_wb/joint_torque path is reused."""
from __future__ import annotations

import time
import numpy as np
import crocoddyl

from ..mpc_result import MPCResult
from .croco_problem import T1ProblemBuilder

_REG = 1e-9


class CrocoMPC:
    def __init__(self, cfg, wb, max_iter: int = 1):
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
        self._foot_l, self._foot_r = self.builder.foot_fids

    def _nominal66(self):
        import pinocchio as pin
        q0 = pin.neutral(self.builder.model); q0[2] = self.cfg.nominal_base_height
        q0[6:6 + self.cfg.n_joints] = self.cfg.nominal_joint_pos
        return np.concatenate([q0, np.zeros(self.nv)])

    def reset(self, x0_68):
        x66 = np.asarray(x0_68, float)[:66]
        self._x0 = x66.copy()
        self.problem.x0 = x66
        self._xs = [x66.copy() for _ in range(self.N + 1)]
        self._us = list(self.problem.quasiStatic([x66.copy() for _ in range(self.N)]))

    def step(self, x_meas_68, t) -> MPCResult:
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
            x_traj[:] = x_traj[0]
        return MPCResult(
            x_traj=x_traj, u_traj=u_traj, feasible=ok, solve_time=self.last_solve_s,
            mode_schedule=None, status=0 if ok else 1,
            node_times=t + self._node_times, u_phys_traj=None)

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
