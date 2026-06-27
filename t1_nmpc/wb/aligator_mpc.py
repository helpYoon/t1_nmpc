"""Persistent warm-started SolverProxDDP for the kinodynamic walk. Phase 1: fixed double-support
stand problem (gait=None). Phase 2: rolling contact schedule via replaceStageCircular+cycleProblem
when gait is set (Task 8). Mirrors CrocoMPC's reset/step interface so the runner + diagnostics
swap in directly."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import aligator
from .aligator_model import make_ode, nominal_stand_x
from .aligator_walk import build_problem, build_gait_cycle, build_problem_from_stages

@dataclass
class AligatorResult:
    xs: list
    us: list
    status: int
    num_iters: int

class AligatorMPC:
    def __init__(self, wb_cfg, al_cfg, am, gait=None):
        self.cfg = wb_cfg; self.al = al_cfg; self.am = am; self.gait = gait
        self.model = am  # exec uses .am-like fields; alias for runner compatibility
        self.last_solve_s = 0.0
        self._built = False

    def _configure(self, problem):
        s = aligator.SolverProxDDP(self.al.tol, self.al.mu_init, max_iters=self.al.max_iters, verbose=aligator.QUIET)
        try: s.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
        except Exception: pass
        try: s.setNumThreads(self.al.num_threads)
        except Exception: pass
        for attr, val in [("max_al_iters", self.al.max_al_iters),
                          ("rollout_type", getattr(aligator, "ROLLOUT_LINEAR", None)),
                          ("sa_strategy", getattr(aligator, "SA_FILTER", None))]:
            if val is not None:
                try: setattr(s, attr, val)
                except Exception: pass
        s.setup(problem)
        return s

    def reset(self, x0):
        x0 = np.asarray(x0, float)
        N = self.al.N
        self._x_ref = nominal_stand_x(self.am, self.cfg)

        if self.gait is not None:
            # ---- Walk path: build a full gait-period ring of stage models ----
            # n_cycle models cover one full gait period so the ring always contains
            # the correct contact schedule at every future horizon knot.
            n_cycle = max(int(round(self.gait.duration / self.cfg.dt)), N)
            node_times = np.arange(n_cycle) * self.cfg.dt
            self._cycle_models, _ = build_gait_cycle(
                self.am, self.cfg, self.al, self.gait, self._x_ref, node_times)
            # Pre-allocate StageData for each ring model (needed by cycleProblem)
            self._cycle_datas = [m.createData() for m in self._cycle_models]
            # First N models go into the initial problem; next to add is index N
            self._ci = N
            self.problem = build_problem_from_stages(
                self.am, self.al, x0, self._x_ref, self._cycle_models[:N])
        else:
            # ---- Stand path: fixed double-support horizon ----
            sched = [[True, True]] * N
            self.problem = build_problem(self.am, self.cfg, self.al, x0, self._x_ref, sched, [[] for _ in range(N)])

        self.solver = self._configure(self.problem)
        ode = make_ode(self.am, [True, True]); mg = self.am.mass * 9.81
        u_grav = np.zeros(ode.nu); u_grav[2] = u_grav[8] = mg / 2
        self.xs = [x0.copy() for _ in range(N + 1)]; self.us = [u_grav.copy() for _ in range(N)]
        self.vs = []; self.lams = []
        self._built = True

    def _recede(self, x_meas, t):
        """Advance the rolling horizon by one knot: drop the oldest stage, append the next
        cycle model. Order is critical: replaceStageCircular FIRST (updates problem structure),
        then cycleProblem (shifts solver internal arrays to match new structure)."""
        n = len(self._cycle_models)
        ci = self._ci % n
        m = self._cycle_models[ci]
        d = self._cycle_datas[ci]
        self.problem.replaceStageCircular(m)
        self.solver.cycleProblem(self.problem, d)
        self._ci += 1
        self.problem.x0_init = x_meas
        # Shift primal warm-start: drop first (solved) state/control, extrapolate at horizon end
        self.xs = self.xs[1:] + [self.xs[-1].copy()]
        self.xs[0] = x_meas.copy()
        self.us = self.us[1:] + [self.us[-1].copy()]

    def step(self, x_meas, t, command=None) -> AligatorResult:
        x_meas = np.asarray(x_meas, float)
        if not self._built:
            self.reset(x_meas)
        if self.gait is not None:
            self._recede(x_meas, t)
        else:
            # Stand path: only update x0; no horizon shift needed
            self.problem.x0_init = x_meas; self.xs[0] = x_meas.copy()
        t0 = time.perf_counter()
        ok = self.solver.run(self.problem, self.xs, self.us, self.vs, self.lams)
        self.last_solve_s = time.perf_counter() - t0
        R = self.solver.results
        self.xs = [np.asarray(a).copy() for a in R.xs]; self.us = [np.asarray(a).copy() for a in R.us]
        self.vs = [np.asarray(a).copy() for a in R.vs]; self.lams = [np.asarray(a).copy() for a in R.lams]
        finite = all(np.all(np.isfinite(a)) for a in self.xs)
        return AligatorResult(self.xs, self.us, 0 if finite else 1, R.num_iters)
