"""Persistent warm-started SolverProxDDP for the kinodynamic walk. Phase 1: fixed double-support
stand problem (no contact cycling yet -- the receding cycle is added in Task 8). Mirrors CrocoMPC's
reset/step interface so the runner + diagnostics swap in directly."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import aligator
from .aligator_model import make_ode, nominal_stand_x
from .aligator_walk import build_problem

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
        sched = [[True, True]] * N
        self._x_ref = nominal_stand_x(self.am, self.cfg)
        self.problem = build_problem(self.am, self.cfg, self.al, x0, self._x_ref, sched, [[] for _ in range(N)])
        self.solver = self._configure(self.problem)
        ode = make_ode(self.am, [True, True]); mg = self.am.mass * 9.81
        u_grav = np.zeros(ode.nu); u_grav[2] = u_grav[8] = mg / 2
        self.xs = [x0.copy() for _ in range(N + 1)]; self.us = [u_grav.copy() for _ in range(N)]
        self.vs = []; self.lams = []
        self._built = True

    def step(self, x_meas, t, command=None) -> AligatorResult:
        x_meas = np.asarray(x_meas, float)
        if not self._built:
            self.reset(x_meas)
        self.problem.x0_init = x_meas; self.xs[0] = x_meas.copy()
        t0 = time.perf_counter()
        ok = self.solver.run(self.problem, self.xs, self.us, self.vs, self.lams)
        self.last_solve_s = time.perf_counter() - t0
        R = self.solver.results
        self.xs = [np.asarray(a).copy() for a in R.xs]; self.us = [np.asarray(a).copy() for a in R.us]
        self.vs = [np.asarray(a).copy() for a in R.vs]; self.lams = [np.asarray(a).copy() for a in R.lams]
        finite = all(np.all(np.isfinite(a)) for a in self.xs)
        return AligatorResult(self.xs, self.us, 0 if finite else 1, R.num_iters)
