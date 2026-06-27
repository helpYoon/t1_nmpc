"""Cost builders for the whole_body_rnea OCP (paper Eq. 3: ||x-x_des||^2_Q + ||u-u_des||^2_R)."""
from __future__ import annotations

import numpy as np
import aligator

from ..robot.config import MPCConfig, ARM_JOINT_SLICE


def state_tracking(space, nu, x_des, Q_diag):
    res = aligator.StateErrorResidual(space, nu, np.asarray(x_des, dtype=np.float64))
    return aligator.QuadraticStateCost(res, np.diag(np.asarray(Q_diag, dtype=np.float64)))


def input_reg(space, nu, u_des, R_diag):
    return aligator.QuadraticControlCost(space, np.asarray(u_des, dtype=np.float64),
                                         np.diag(np.asarray(R_diag, dtype=np.float64)))


def arm_to_nominal(space, nu, x_des, cfg: MPCConfig):
    """High-weight state cost on arm joint positions only (M1 holds arms to nominal)."""
    w = np.zeros(cfg.ndx)
    # arm joint_pos delta indices: base_pos(6) + ARM_JOINT_SLICE within the 27 joints
    arm = np.arange(6 + ARM_JOINT_SLICE.start, 6 + ARM_JOINT_SLICE.stop)
    w[arm] = cfg.arm_weight_scale
    res = aligator.StateErrorResidual(space, nu, np.asarray(x_des, dtype=np.float64))
    return aligator.QuadraticStateCost(res, np.diag(w))


def gravity_comp_u_des(rm, n_support: int) -> np.ndarray:
    u = np.zeros(45, dtype=np.float64)
    if n_support > 0:
        fz = rm.mass * 9.81 / n_support
        for k in range(2):                    # both feet entries; swing feet get fz too as a
            u[33 + 6 * k + 2] = fz            # mild prior (overridden by swing W=0 constraint)
    return u


def make_cost_stack(space, nu, components_with_weights):
    stack = aligator.CostStack(space, nu)
    for name, cost, weight in components_with_weights:
        stack.addCost(name, cost, weight)
    return stack
