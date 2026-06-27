"""Thin assembler: build aligator StageModels from gait flags. No physics lives here."""
from __future__ import annotations

import numpy as np
import aligator
from aligator import dynamics as ali_dyn

from ..robot.config import MPCConfig
from ..robot.model import RobotModel, nominal_x
from .dynamics import WBDynamics
from . import cost as K
from . import constraint as C


class OCPBuilder:
    def __init__(self, cfg: MPCConfig, rm: RobotModel, dyn: WBDynamics):
        self.cfg, self.rm, self.dyn = cfg, rm, dyn
        self.space = dyn.space
        self.x_des = nominal_x(cfg, rm.model)
        self._rnea_funcs = dyn.rnea_funcs(base_only=True)   # built once, shared across stages
        self.u_des = K.gravity_comp_u_des(rm, n_support=2)

    def _discrete_dynamics(self):
        ode = self.dyn.DoubleIntegratorODE(self.space, self.cfg.nu)
        return ali_dyn.IntegratorEuler(ode, self.cfg.dt)

    def _cost(self):
        comps = [
            ("state", K.state_tracking(self.space, self.cfg.nu, self.x_des, self.cfg.Q_diag), 1.0),
            ("input", K.input_reg(self.space, self.cfg.nu, self.u_des, self.cfg.R_diag), 1.0),
            ("arms", K.arm_to_nominal(self.space, self.cfg.nu, self.x_des, self.cfg), 1.0),
        ]
        return K.make_cost_stack(self.space, self.cfg.nu, comps)

    def build_stage(self, mode):
        cfg, rm = self.cfg, self.rm
        stage = aligator.StageModel(self._cost(), self._discrete_dynamics())
        stage.addConstraint(C.RneaBaseResidual(cfg.ndx, cfg.nu, self._rnea_funcs), C.EQ())
        handles = {"swing": []}
        cidx = 1                                   # rnea_base occupies constraint-stack index 0
        for k, in_contact in enumerate(mode):
            if in_contact:
                stage.addConstraint(
                    C.WrenchConeResidual(cfg.ndx, cfg.nu, k, cfg.friction_mu,
                                         cfg.half_len, cfg.half_width), C.NEG()); cidx += 1
                stage.addConstraint(C.contact_velocity_residual(rm, cfg.ndx, cfg.nu, k), C.EQ()); cidx += 1
            else:
                stage.addConstraint(C.SwingWrenchResidual(cfg.ndx, cfg.nu, k), C.EQ()); cidx += 1
                sliced, _ = C.swing_z_residual(rm, cfg.ndx, cfg.nu, k)   # discard the disconnected base
                stage.addConstraint(sliced, C.EQ())
                handles["swing"].append((k, cidx)); cidx += 1            # record swing-z stack index
        return stage, handles

    def terminal_cost(self):
        return K.state_tracking(self.space, self.cfg.nu, self.x_des, self.cfg.Q_diag)

    def build_problem(self, modes, x0):
        stages = aligator.StdVec_StageModel()
        all_handles = []
        for mode in modes:
            stage, handles = self.build_stage(mode)
            stages.append(stage)
            all_handles.append(handles)
        problem = aligator.TrajOptProblem(np.asarray(x0, dtype=np.float64), stages,
                                          self.terminal_cost())
        return problem, all_handles
