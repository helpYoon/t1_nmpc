"""AligatorMPC: SolverProxDDP over the whole_body_rnea OCP with cyclic warm-start carry."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import aligator

from ..robot.config import MPCConfig, JointCommand
from ..robot.model import RobotModel, nominal_x
from .dynamics import WBDynamics
from .ocp import OCPBuilder
from .gait import v_z_ref
from .state import extract_command


@dataclass
class MPCResult:
    command: JointCommand
    forces0: np.ndarray
    solve_time: float
    constr_viol: float
    num_iters: int


class AligatorMPC:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, gait):
        self.cfg, self.rm, self.gait = cfg, rm, gait
        self.dyn = WBDynamics(rm, cfg)
        self.builder = OCPBuilder(cfg, rm, self.dyn)
        self.tau_fn = self.dyn.joint_torque_fn()
        self._is_walk = hasattr(gait, "t_lf_end")  # WalkGait has phase boundaries

        modes = gait.horizon_modes(0.0)
        x0 = nominal_x(cfg, rm.model)
        self.problem, self.handles = self.builder.build_problem(modes, x0)
        self.solver = aligator.SolverProxDDP(cfg.al_tol, cfg.mu_init,
                                             cfg.cold_max_iters, aligator.QUIET)
        self.solver.setup(self.problem)
        self._warm = None      # (xs, us, vs, lams)

    def _refresh_refs(self, t: float):
        # Reach each swing-z residual THROUGH the problem (addConstraint deep-copied it; the
        # original builder handle is disconnected). funcs[cidx] is the slice; .func is the
        # wrapped FrameVelocityResidual; set its vref property (NOT deprecated setReference).
        for i, handles in enumerate(self.handles):
            for (foot_index, cidx) in handles["swing"]:
                phase = self.gait.swing_phase(t + i * self.cfg.dt, foot_index)
                vz = v_z_ref(phase, self.cfg) if phase is not None else 0.0
                self.problem.stages[i].constraints.funcs[cidx].func.vref = \
                    pin.Motion(np.array([0, 0, vz, 0, 0, 0.0]))

    def reset(self, x0) -> None:
        x0 = np.asarray(x0, dtype=np.float64)
        self.problem.x0_init = x0
        self.solver.max_iters = self.cfg.cold_max_iters
        self._refresh_refs(0.0)
        xs = [x0.copy() for _ in range(self.cfg.nodes + 1)]
        us = [self.builder.u_des.copy() for _ in range(self.cfg.nodes)]
        self.solver.run(self.problem, xs, us)
        r = self.solver.results
        self._warm = (list(r.xs), list(r.us), list(r.vs), list(r.lams))

    def step(self, x_meas, t: float) -> MPCResult:
        x = np.asarray(x_meas, dtype=np.float64)
        if self._is_walk:
            # advance the ring by one knot: replaceStageCircular + self.handles rotation +
            # cycleProblem shift the problem and the solver's internal data; the carried
            # self._warm lists are passed to run as-is (their alignment across cycled
            # heterogeneous stages is a follow-up-walk-plan concern).
            tip_t = t + self.cfg.nodes * self.cfg.dt
            tip_stage, tip_handles = self.builder.build_stage(self.gait.mode_at(tip_t))
            self.problem.replaceStageCircular(tip_stage)
            self.handles = self.handles[1:] + [tip_handles]
            self.solver.cycleProblem(self.problem, self.problem.stages[-1].createData())
        self.problem.x0_init = x
        self._refresh_refs(t)
        self.solver.max_iters = self.cfg.warm_max_iters
        xs, us, vs, lams = self._warm
        t0 = time.perf_counter()
        self.solver.run(self.problem, xs, us, vs, lams)
        dt = time.perf_counter() - t0
        r = self.solver.results
        self._warm = (list(r.xs), list(r.us), list(r.vs), list(r.lams))

        x1 = np.asarray(r.xs[1], dtype=np.float64)
        u0 = np.asarray(r.us[0], dtype=np.float64)
        tau = np.asarray(self.tau_fn(np.asarray(r.xs[0]), u0)).flatten()
        cmd = extract_command(x1, tau, self.cfg, self.rm)
        return MPCResult(command=cmd, forces0=u0[33:].copy(), solve_time=dt,
                         constr_viol=float(r.primal_infeas), num_iters=int(r.num_iters))
